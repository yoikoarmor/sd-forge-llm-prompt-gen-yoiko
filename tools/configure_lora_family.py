"""Configure verified local adapters and/or one published merged model.

This command does not upload, download, merge, log in, or modify model files.
Existing registry entries are preserved. Run from any working directory.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure(registry_path, *, adapter_root=None, base_model=None,
              merged_model=None, merged_revision=None, gguf_model=None,
              gguf_revision=None):
    family = load_json(ROOT / "configs/lora_family.json")
    example = load_json(ROOT / "configs/model_registry.example.json")
    registry_path = Path(registry_path)
    data = load_json(registry_path) if registry_path.exists() else example
    if not isinstance(data.get("models"), dict):
        raise ValueError("Registry must contain a models object.")
    if not adapter_root and not merged_model and not gguf_model:
        raise ValueError("Provide --adapter-root, --merged-model, and/or --gguf-model.")

    configured = []
    if adapter_root:
        adapter_root = Path(adapter_root).expanduser().resolve(strict=True)
        if base_model and base_model != family["base_repo"]:
            base_path = Path(base_model).expanduser()
            if not base_path.is_dir() or not (base_path / "config.json").is_file():
                raise ValueError("--base-model must be the Qwen/Qwen3.5-9B ID or its existing local snapshot.")
            base_model = str(base_path.resolve())
        # Validate the complete family before changing any config.
        for item in family["models"]:
            adapter = adapter_root / item["version"] / "best_eval_adapter"
            for filename in ("adapter_config.json", "adapter_model.safetensors",
                             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                if not (adapter / filename).is_file():
                    raise ValueError(f"Missing adapter file: {adapter / filename}")
            if sha256(adapter / "adapter_model.safetensors") != item["adapter_sha256"]:
                raise ValueError(f"SHA256 mismatch for {item['version']}; expected best-eval epoch 3.")
            if sha256(adapter / "chat_template.jinja") != family["template_sha256"]:
                raise ValueError(f"Chat-template SHA256 mismatch for {item['version']}.")
            entry = dict(example["models"][item["key"]])
            entry.update(enabled=True, adapter_path=str(adapter),
                         base_model_name_or_path=str(base_model or family["base_repo"]),
                         local_files_only=bool(base_model and Path(base_model).exists()))
            data["models"][item["key"]] = entry
            configured.append(item["key"])

    if merged_model:
        merged_path = Path(merged_model).expanduser()
        remote = bool(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", str(merged_model)))
        if merged_path.is_dir():
            merged_model = str(merged_path.resolve())
            remote = False
            for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                if not (merged_path / filename).is_file():
                    raise ValueError(f"Missing merged model file: {filename}")
            if not any(merged_path.glob("model*.safetensors")):
                raise ValueError("Merged model directory contains no model safetensors.")
            if (merged_path / "adapter_config.json").exists():
                raise ValueError("Expected a merged model, not a PEFT adapter directory.")
        elif not remote:
            raise ValueError("Merged model must be an existing directory or owner/repository HF ID.")
        if remote and not re.fullmatch(r"[0-9a-f]{40}", merged_revision or ""):
            raise ValueError("Remote merged models require --merged-revision with the verified 40-character HF commit SHA.")
        key = "qwen3.5-9b-lora7-merged"
        entry = dict(example["models"][key])
        entry.update(enabled=True, base_model_name_or_path=str(merged_model),
                     base_model_revision=merged_revision if remote else None,
                     local_files_only=not remote)
        data["models"][key] = entry
        configured.append(key)

    if gguf_model:
        key = "qwen3.5-9b-lora-7-q4km"
        entry = dict(example["models"][key])
        model_path = Path(gguf_model).expanduser()
        remote = bool(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", str(gguf_model)))
        if model_path.is_file():
            if model_path.suffix.lower() != ".gguf":
                raise ValueError("--gguf-model must point to a GGUF file.")
            if sha256(model_path) != entry["gguf_sha256"]:
                raise ValueError("GGUF SHA256 mismatch; expected the verified Instruct 7 Q4_K_M file.")
            gguf_model = str(model_path.resolve())
            remote = False
        elif not remote:
            raise ValueError("--gguf-model must be an existing GGUF file or owner/repository HF ID.")
        if remote and not re.fullmatch(r"[0-9a-f]{40}", gguf_revision or ""):
            raise ValueError("Remote GGUF models require --gguf-revision with the verified 40-character HF commit SHA.")
        entry.update(enabled=True, gguf_path=None if remote else str(gguf_model),
                     gguf_repo_id=str(gguf_model) if remote else None,
                     gguf_revision=gguf_revision if remote else None, local_files_only=not remote)
        data["models"][key] = entry
        configured.append(key)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after all validation has succeeded; never leave partial JSON.
    fd, temp_name = tempfile.mkstemp(prefix=".registry-", suffix=".json", dir=registry_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, registry_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return configured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/model_registry.json")
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--base-model", help="Existing local fixed-revision snapshot, or the base HF ID")
    parser.add_argument("--merged-model", help="Published merged HF repo ID, or existing local directory")
    parser.add_argument("--merged-revision", help="Verified 40-character HF commit SHA")
    parser.add_argument("--gguf-model", help="Instruct 7 Q4_K_M GGUF file or published HF repository")
    parser.add_argument("--gguf-revision", help="Verified 40-character GGUF repository commit SHA")
    args = parser.parse_args()
    try:
        configured = configure(args.registry, adapter_root=args.adapter_root,
                               base_model=args.base_model, merged_model=args.merged_model,
                               merged_revision=args.merged_revision, gguf_model=args.gguf_model,
                               gguf_revision=args.gguf_revision)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Configured {', '.join(configured)} in {args.registry}")


if __name__ == "__main__":
    main()
