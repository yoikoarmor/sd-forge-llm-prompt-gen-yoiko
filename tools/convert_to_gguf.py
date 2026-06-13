import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = EXTENSION_ROOT / "backend"
CONFIG_DIR = EXTENSION_ROOT / "configs"
MODEL_REGISTRY_PATH = CONFIG_DIR / "model_registry.json"
MODEL_REGISTRY_EXAMPLE_PATH = CONFIG_DIR / "model_registry.example.json"
DEFAULT_LLAMA_CPP_REF = "1593d5684d077c07fc788e9527ec1bd52287de7f"


def log(message):
    print(f"[convert_to_gguf] {message}", flush=True)


def run(cmd, *, cwd=None, dry_run=False):
    log("run " + " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, check=True)


def load_registry():
    path = MODEL_REGISTRY_PATH if MODEL_REGISTRY_PATH.exists() else MODEL_REGISTRY_EXAMPLE_PATH
    return path, json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_model_entry(model_key):
    path, registry = load_registry()
    models = registry.get("models") or {}
    if model_key not in models:
        raise SystemExit(f"model key not found in {path}: {model_key}")
    return path, models[model_key]


def import_runtime_deps():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"missing conversion dependency: {exc}") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def merge_model(base, adapter, tokenizer_source, merged_dir, dtype, local_files_only):
    torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = import_runtime_deps()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        raise SystemExit(f"unsupported dtype: {dtype}")

    merged_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading base model: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="cpu",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if adapter:
        log(f"loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(
            model,
            adapter,
            is_trainable=False,
            local_files_only=local_files_only,
        )
        log("merging LoRA into base model")
        model = model.merge_and_unload()

    log(f"saving merged model: {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer_ref = tokenizer_source or adapter or base
    log(f"saving tokenizer/template from: {tokenizer_ref}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(merged_dir)

    template_path = Path(str(tokenizer_ref)) / "chat_template.jinja"
    if template_path.exists():
        shutil.copy2(template_path, merged_dir / "chat_template.jinja")


def ensure_llama_cpp(llama_cpp_dir, outdir, llama_cpp_ref=DEFAULT_LLAMA_CPP_REF, dry_run=False):
    if llama_cpp_dir:
        path = Path(llama_cpp_dir).resolve()
    else:
        path = (outdir / "_llama.cpp").resolve()

    if (path / "convert_hf_to_gguf.py").exists():
        if llama_cpp_ref:
            run(["git", "fetch", "--tags", "--depth", "1", "origin", llama_cpp_ref], cwd=path, dry_run=dry_run)
            run(["git", "checkout", llama_cpp_ref], cwd=path, dry_run=dry_run)
        return path

    if llama_cpp_dir:
        raise SystemExit(f"convert_hf_to_gguf.py not found under --llama-cpp-dir: {path}")

    run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", path], dry_run=dry_run)
    if llama_cpp_ref:
        run(["git", "checkout", llama_cpp_ref], cwd=path, dry_run=dry_run)
    req = path / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", req], dry_run=dry_run)
    return path


def convert_hf_to_gguf(llama_cpp_dir, merged_dir, f16_gguf, outtype, dry_run=False):
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    run(
        [
            sys.executable,
            converter,
            merged_dir,
            "--outfile",
            f16_gguf,
            "--outtype",
            outtype,
        ],
        dry_run=dry_run,
    )


def find_quantize_binary(llama_cpp_dir, explicit):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"quantize binary not found: {path}")
        return path

    candidates = [
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "quantize.exe",
        llama_cpp_dir / "build" / "bin" / "quantize.exe",
        llama_cpp_dir / "llama-quantize.exe",
        llama_cpp_dir / "quantize.exe",
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def quantize_gguf(llama_cpp_dir, f16_gguf, final_gguf, quant, quantize_bin, dry_run=False):
    quantize = find_quantize_binary(llama_cpp_dir, quantize_bin)
    if quantize is None:
        if dry_run:
            quantize = "<llama-quantize>"
        else:
            raise SystemExit(
                "llama-quantize binary was not found. Build llama.cpp or pass --quantize-bin. "
                "For f16/q8_0-only output, use --quant f16 or --quant q8_0."
            )
    run([quantize, f16_gguf, final_gguf, quant], dry_run=dry_run)


def update_registry(model_key, gguf_path, quant, n_ctx, n_gpu_layers, n_batch, dry_run=False):
    registry_path = MODEL_REGISTRY_PATH
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    else:
        registry = {"models": {}}

    quant_suffix = str(quant).lower().replace("_", "")
    entry_key = f"{model_key}-gguf-{quant_suffix}"
    entry = {
        "enabled": True,
        "backend": "llama_cpp",
        "description": f"GGUF model generated from {model_key} by tools/convert_to_gguf.py",
        "gguf_path": str(Path(gguf_path).resolve()),
        "n_ctx": int(n_ctx),
        "n_gpu_layers": int(n_gpu_layers),
        "n_batch": int(n_batch),
        "flash_attn": True,
    }
    registry.setdefault("models", {})[entry_key] = entry

    log("registry snippet:")
    print(json.dumps({entry_key: entry}, ensure_ascii=False, indent=2))
    if dry_run:
        return

    if registry_path.exists():
        backup = registry_path.with_suffix(f".json.bak-{int(time.time())}")
        shutil.copy2(registry_path, backup)
        log(f"backup written: {backup}")
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"registry updated: {registry_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a HF base+LoRA model and convert it to GGUF.")
    parser.add_argument("--model-key", help="Model key from configs/model_registry.json or example.")
    parser.add_argument("--base", help="Base model path or Hugging Face repo ID.")
    parser.add_argument("--adapter", help="LoRA adapter path or Hugging Face repo ID.")
    parser.add_argument("--tokenizer-source", help="Tokenizer/template source. Defaults to adapter, then base.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--quant", default="Q4_K_M", help="f16, q8_0, Q4_K_M, Q5_K_M, etc.")
    parser.add_argument("--dtype", default="float16", help="Merge dtype: float16, bfloat16, float32.")
    parser.add_argument("--llama-cpp-dir", help="Existing llama.cpp checkout.")
    parser.add_argument(
        "--llama-cpp-ref",
        default=DEFAULT_LLAMA_CPP_REF,
        help=f"llama.cpp git tag/commit to use. Default: {DEFAULT_LLAMA_CPP_REF}",
    )
    parser.add_argument("--quantize-bin", help="Path to llama-quantize binary.")
    parser.add_argument("--register", action="store_true", help="Append a llama_cpp entry to configs/model_registry.json.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download from Hugging Face.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model_key and not args.base:
        raise SystemExit("Specify --model-key or --base.")

    model_key = args.model_key or "custom"
    base = args.base
    adapter = args.adapter
    tokenizer_source = args.tokenizer_source
    if args.model_key:
        _, entry = resolve_model_entry(args.model_key)
        base = base or entry.get("base_model_name_or_path")
        adapter = adapter if adapter is not None else entry.get("adapter_path")
        tokenizer_source = tokenizer_source or entry.get("tokenizer_name_or_path") or adapter or base

    if not base:
        raise SystemExit("Base model is required.")

    outdir = Path(args.outdir).resolve()
    merged_dir = outdir / f"{model_key}-merged-hf"
    quant_normalized = args.quant.lower()
    final_gguf = outdir / f"{model_key}-merged-{args.quant}.gguf"
    f16_gguf = final_gguf if quant_normalized in {"f16", "q8_0"} else outdir / f"{model_key}-merged-f16.gguf"

    if args.dry_run:
        log(f"dry-run merge source base={base} adapter={adapter} tokenizer_source={tokenizer_source}")
        log(f"dry-run merged model dir={merged_dir}")
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        merge_model(base, adapter, tokenizer_source, merged_dir, args.dtype, args.local_files_only)

    llama_cpp_dir = ensure_llama_cpp(
        args.llama_cpp_dir,
        outdir,
        llama_cpp_ref=args.llama_cpp_ref,
        dry_run=args.dry_run,
    )
    convert_hf_to_gguf(llama_cpp_dir, merged_dir, f16_gguf, "q8_0" if quant_normalized == "q8_0" else "f16", dry_run=args.dry_run)

    if quant_normalized not in {"f16", "q8_0"}:
        quantize_gguf(llama_cpp_dir, f16_gguf, final_gguf, args.quant, args.quantize_bin, dry_run=args.dry_run)

    if args.register:
        update_registry(model_key, final_gguf, args.quant, args.n_ctx, args.n_gpu_layers, args.n_batch, dry_run=args.dry_run)

    log(f"done gguf={final_gguf}")


if __name__ == "__main__":
    main()
