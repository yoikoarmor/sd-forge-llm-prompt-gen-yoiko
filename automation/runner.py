"""Generation phase: random Gen Prompt -> Forge txt2img via API.

Usage (from the automation/ directory, with the Forge venv python):
    python runner.py --count 20
    python runner.py --count 20 --score   # run scoring right after
"""

import argparse
import base64
import datetime
import json
import re
import sys
from pathlib import Path

AUTOMATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTOMATION_DIR))

from db import connect_db, insert_run
from forge_client import ForgeClient
from sampler import GenPromptSampler

LLM_MODE_RE = re.compile(r"LLM Prompt Gen: ([^,\n]+)")


def load_config():
    with open(AUTOMATION_DIR / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_payload(config, gen_prompt):
    gen = config["generation"]
    ext = config["llm_extension"]
    return {
        "prompt": "",
        "negative_prompt": gen["negative_prompt"],
        "steps": gen["steps"],
        "width": gen["width"],
        "height": gen["height"],
        "cfg_scale": gen["cfg_scale"],
        "sampler_name": gen["sampler_name"],
        "seed": gen.get("seed", -1),
        "batch_size": 1,
        "n_iter": 1,
        "alwayson_scripts": {
            ext["script_title"]: {
                "args": [
                    ext["enabled"],
                    ext["model"],
                    ext["load_mode"],
                    ext["weight_mode"],
                    ext["max_new_tokens"],
                    ext["num_candidates"],
                    gen_prompt,
                    "",
                ]
            }
        },
    }


def save_image(b64_data, image_dir, stem):
    image_dir.mkdir(parents=True, exist_ok=True)
    if "," in b64_data and b64_data.strip().startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]
    path = image_dir / f"{stem}.png"
    path.write_bytes(base64.b64decode(b64_data))
    return path


def parse_llm_mode(info):
    infotexts = info.get("infotexts") or [""]
    m = LLM_MODE_RE.search(infotexts[0])
    return m.group(1).strip() if m else None


def run_generation(config, count, sampler_seed=None):
    client = ForgeClient(
        config["forge"]["base_url"],
        timeout=config["forge"]["request_timeout_seconds"],
    )
    if not client.wait_alive():
        print(f"ERROR: Forge API not reachable at {config['forge']['base_url']}.")
        print("Start Forge with the --api flag and try again.")
        return False

    output_dir = AUTOMATION_DIR / config["paths"]["output_dir"]
    conn = connect_db(output_dir / "runs.sqlite3")
    sampler = GenPromptSampler(
        AUTOMATION_DIR / config["paths"]["wordpool"], seed=sampler_seed
    )
    day = datetime.datetime.now().strftime("%Y%m%d")
    image_dir = output_dir / "images" / day

    fallback_count = 0
    for i in range(count):
        gen_prompt, parts = sampler.sample()
        ts = datetime.datetime.now().strftime("%H%M%S")
        print(f"[{i + 1}/{count}] gen_prompt: {gen_prompt}")

        result = client.txt2img(build_payload(config, gen_prompt))
        info = json.loads(result["info"])
        llm_mode = parse_llm_mode(info)
        if llm_mode == "llm fallback":
            fallback_count += 1

        image_path = save_image(result["images"][0], image_dir, f"{day}_{ts}_{i:04d}")
        run_id = insert_run(
            conn,
            created_at=datetime.datetime.now().isoformat(timespec="seconds"),
            gen_prompt=gen_prompt,
            parts_json=json.dumps(parts, ensure_ascii=False),
            final_prompt=info.get("prompt"),
            negative_prompt=info.get("negative_prompt"),
            llm_mode=llm_mode,
            seed=info.get("seed"),
            image_path=str(image_path),
            info_json=result["info"],
        )
        print(f"  -> run {run_id} | mode={llm_mode} | {image_path.name}")

    conn.close()
    print(f"Done. {count} images, llm fallback rate: {fallback_count}/{count}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--score", action="store_true", help="run scorer afterwards")
    parser.add_argument("--sampler-seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    count = args.count if args.count is not None else config["generation"]["count"]
    ok = run_generation(config, count, sampler_seed=args.sampler_seed)
    if ok and args.score:
        import scorer

        scorer.run_scoring(config)


if __name__ == "__main__":
    main()
