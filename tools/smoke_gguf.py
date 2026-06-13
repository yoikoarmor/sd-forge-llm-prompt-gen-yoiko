import argparse
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from backend.generator import generate_prompt_candidates
from backend.loader import MissingDependencyError, ModelLoadError, load_model_bundle
from backend.registry import ModelSpec, get_generation_defaults, get_model_spec


def log(message):
    print(f"[smoke_gguf] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test a llama.cpp GGUF prompt generator.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-key", help="llama_cpp model key from configs/model_registry.json or example.")
    source.add_argument("--gguf-path", help="Local GGUF file path.")
    source.add_argument("--repo-id", help="Hugging Face repo ID that contains a GGUF file.")
    parser.add_argument("--filename", help="GGUF filename when --repo-id is used.")
    parser.add_argument("--prompt", default="女性、高身長、ビジネススーツ", help="Prompt text to rewrite.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=256)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--local-files-only", action="store_true", help="Use only local Hugging Face cache files.")
    parser.add_argument("--no-flash-attn", action="store_true")
    return parser.parse_args()


def build_spec(args):
    if args.model_key:
        spec = get_model_spec(args.model_key)
        if spec.backend != "llama_cpp":
            raise SystemExit(f"model key is not a llama_cpp backend: {args.model_key}")
        return spec

    if args.repo_id and not args.filename:
        raise SystemExit("--filename is required with --repo-id")

    return ModelSpec(
        key="smoke-gguf",
        backend="llama_cpp",
        gguf_path=args.gguf_path,
        gguf_repo_id=args.repo_id,
        gguf_filename=args.filename,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
        flash_attn=not args.no_flash_attn,
        local_files_only=args.local_files_only,
    )


def main():
    args = parse_args()
    spec = build_spec(args)
    defaults = get_generation_defaults()
    defaults["max_new_tokens"] = args.max_new_tokens
    defaults["seed_mode"] = "fixed"
    defaults["llm_seed"] = 1
    defaults["enable_thinking"] = False

    def logger(message):
        log(message)

    bundle = None
    try:
        bundle = load_model_bundle(spec, logger=logger)
        result = generate_prompt_candidates(
            bundle,
            args.prompt,
            "",
            "",
            defaults,
            num_candidates=1,
            logger=logger,
        )
        log(f"fallback_used={result.fallback_used} fallback_reason={result.fallback_reason}")
        print(result.selected_positive.strip())
    except MissingDependencyError as exc:
        raise SystemExit(f"missing dependency: {exc}") from exc
    except ModelLoadError as exc:
        raise SystemExit(f"model load failed: {exc}") from exc
    finally:
        if bundle is not None:
            close = getattr(bundle.model, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
