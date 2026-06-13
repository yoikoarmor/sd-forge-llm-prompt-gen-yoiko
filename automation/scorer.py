"""Scoring phase: CLIP score + VLM judge + optional distilled discriminator.

VRAM lifecycle rules:
- Every scorer is a context manager. Models are loaded on __enter__ and are
  guaranteed to be released on __exit__ (del -> gc -> empty_cache -> ipc_collect),
  including on exceptions.
- Before the VLM phase the Forge image checkpoint can be unloaded over the API
  (config: scoring.unload_forge_checkpoint_during_vlm) and is reloaded after.

Usage:
    python scorer.py
"""

import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

AUTOMATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTOMATION_DIR))

import db
from forge_client import ForgeClient

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

VLM_JUDGE_TEMPLATE = """You are a strict quality inspector for AI-generated images.
Score the attached image on each axis as an integer from 0 (worst) to 10 (best).

- "prompt_fidelity": how well the image matches this intent: "{gen_prompt}"
- "anatomy": correctness of bodies, hands, faces (10 = no issues; if the image contains no characters or creatures, score 10)
- "artifacts": freedom from noise, distortion, broken geometry (10 = clean)
- "aesthetics": overall visual appeal and composition
- "overall": overall quality

Respond with ONLY a JSON object, no other text:
{{"prompt_fidelity": n, "anatomy": n, "artifacts": n, "aesthetics": n, "overall": n, "comment": "<one short sentence>"}}"""


def _vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**2


def _free_vram(label):
    """Force VRAM release after the caller has dropped all model references."""
    before = _vram_mb()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"  [{label}] VRAM (this process): {before:.0f}MB -> {_vram_mb():.0f}MB")


class ClipScorer:
    """CLIP similarity between gen_prompt and image. Also yields image embeddings."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def __enter__(self):
        from transformers import CLIPModel, CLIPProcessor

        print(f"Loading CLIP: {self.model_id}")
        self.model = (
            CLIPModel.from_pretrained(self.model_id, dtype=torch.float16)
            .to("cuda" if torch.cuda.is_available() else "cpu")
            .eval()
        )
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.model = None
        self.processor = None
        _free_vram("clip unload")
        return False

    @staticmethod
    def _as_features(out):
        # transformers >= 5 returns a model output whose pooler_output holds
        # the projected features; transformers 4.x returned the tensor directly.
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def score(self, image_path, gen_prompt):
        """Returns (clip_score, image_embedding ndarray float32, normalized)."""
        image = Image.open(image_path).convert("RGB")
        device = self.model.device
        img_inputs = self.processor(images=image, return_tensors="pt").to(device)
        txt_inputs = self.processor(
            text=[gen_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        img_emb = self._as_features(self.model.get_image_features(**img_inputs))
        txt_emb = self._as_features(self.model.get_text_features(**txt_inputs))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        score = float((img_emb @ txt_emb.T).squeeze().item())
        return score, img_emb.squeeze(0).float().cpu().numpy()


class VlmJudge:
    """4bit VLM judge with guaranteed VRAM release."""

    def __init__(self, model_id, load_in_4bit=True, max_new_tokens=256):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def __enter__(self):
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        print(f"Loading VLM: {self.model_id} (4bit={self.load_in_4bit})")
        kwargs = {"device_map": "auto"}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"  VLM loaded, VRAM (this process): {_vram_mb():.0f}MB")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.model = None
        self.processor = None
        _free_vram("vlm unload")
        return False

    @torch.no_grad()
    def judge(self, image_path, gen_prompt):
        """Returns (scores_dict_or_None, raw_text)."""
        image = Image.open(image_path).convert("RGB")
        prompt = VLM_JUDGE_TEMPLATE.format(gen_prompt=gen_prompt.replace('"', "'"))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return self._parse(text), text

    @staticmethod
    def _parse(text):
        m = JSON_BLOCK_RE.search(text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        keys = ["prompt_fidelity", "anatomy", "artifacts", "aesthetics", "overall"]
        if not all(isinstance(data.get(k), (int, float)) for k in keys):
            return None
        for k in keys:
            data[k] = max(0.0, min(10.0, float(data[k])))
        return data


def _clip_phase(conn, config):
    rows = db.rows_missing_clip(conn)
    if not rows:
        print("CLIP phase: nothing to score.")
        return
    print(f"CLIP phase: {len(rows)} images")
    with ClipScorer(config["scoring"]["clip_model"]) as clip:
        for run_id, gen_prompt, image_path in rows:
            if not Path(image_path).exists():
                print(f"  run {run_id}: image missing, skipped")
                continue
            score, emb = clip.score(image_path, gen_prompt)
            db.update_clip(conn, run_id, score, emb.astype(np.float32).tobytes())
            print(f"  run {run_id}: clip={score:.4f}")


def _disc_phase(conn, config):
    disc_path = AUTOMATION_DIR / config["paths"]["output_dir"] / "discriminator.pt"
    if not config["scoring"].get("use_discriminator_if_available", True):
        return
    if not disc_path.exists():
        print("Discriminator phase: no discriminator.pt yet (train with distill.py).")
        return
    rows = db.rows_missing_disc(conn)
    if not rows:
        print("Discriminator phase: nothing to score.")
        return

    from distill import build_mlp

    ckpt = torch.load(disc_path, map_location="cpu", weights_only=True)
    mlp = build_mlp(ckpt["input_dim"])
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()
    print(f"Discriminator phase: {len(rows)} rows (val MAE {ckpt['val_mae_0_10']:.2f})")
    with torch.no_grad():
        for run_id, emb_bytes in rows:
            emb = torch.from_numpy(
                np.frombuffer(emb_bytes, dtype=np.float32).copy()
            ).unsqueeze(0)
            score = float(mlp(emb).squeeze().item()) * 10.0
            db.update_disc(conn, run_id, max(0.0, min(10.0, score)))
    del mlp
    gc.collect()


def _vlm_phase(conn, config):
    scoring = config["scoring"]
    if not scoring.get("enable_vlm", True):
        print("VLM phase: disabled in config.")
        return
    rows = db.rows_missing_vlm(conn)
    if not rows:
        print("VLM phase: nothing to score.")
        return

    client = ForgeClient(config["forge"]["base_url"])
    forge_unloaded = False
    if scoring.get("unload_forge_checkpoint_during_vlm", True) and client.alive():
        forge_unloaded = client.unload_checkpoint()
        print(f"Forge checkpoint unload: {'ok' if forge_unloaded else 'unavailable'}")

    print(f"VLM phase: {len(rows)} images")
    try:
        with VlmJudge(
            scoring["vlm_model"],
            load_in_4bit=scoring.get("vlm_load_in_4bit", True),
            max_new_tokens=scoring.get("vlm_max_new_tokens", 256),
        ) as judge:
            for run_id, gen_prompt, image_path in rows:
                if not Path(image_path).exists():
                    print(f"  run {run_id}: image missing, skipped")
                    continue
                scores, raw = judge.judge(image_path, gen_prompt)
                if scores is None:
                    print(f"  run {run_id}: VLM output not parseable, stored raw")
                    db.update_vlm(
                        conn, run_id, json.dumps({"raw": raw}, ensure_ascii=False), None
                    )
                    continue
                db.update_vlm(
                    conn,
                    run_id,
                    json.dumps(scores, ensure_ascii=False),
                    scores["overall"],
                )
                print(
                    f"  run {run_id}: overall={scores['overall']:.0f} "
                    f"fidelity={scores['prompt_fidelity']:.0f} "
                    f"anatomy={scores['anatomy']:.0f}"
                )
    finally:
        if forge_unloaded:
            ok = client.reload_checkpoint()
            print(f"Forge checkpoint reload: {'ok' if ok else 'FAILED (reload manually)'}")


def run_scoring(config):
    output_dir = AUTOMATION_DIR / config["paths"]["output_dir"]
    conn = db.connect_db(output_dir / "runs.sqlite3")
    try:
        _clip_phase(conn, config)
        _disc_phase(conn, config)
        _vlm_phase(conn, config)
    finally:
        conn.close()
    print("Scoring done.")


def main():
    with open(AUTOMATION_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    run_scoring(config)


if __name__ == "__main__":
    main()
