import copy
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXTENSION_ROOT / "configs"
MODEL_REGISTRY_PATH = CONFIG_DIR / "model_registry.json"
MODEL_REGISTRY_EXAMPLE_PATH = CONFIG_DIR / "model_registry.example.json"
GENERATION_DEFAULTS_PATH = CONFIG_DIR / "generation_defaults.json"


class RegistryError(Exception):
    pass


@dataclass
class ModelSpec:
    key: str
    base_model_name_or_path: str = ""
    adapter_path: str | None = None
    tokenizer_name_or_path: str | None = None
    cache_dir: str | None = None
    fallback_base_model_name_or_path: str | None = None
    fallback_adapter_path: str | None = None
    fallback_tokenizer_name_or_path: str | None = None
    allow_auto_download_missing: bool = True
    registry_source: str = "model_registry.json"
    enabled: bool = True
    load_in_4bit: bool = True
    merge_lora_for_inference: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    local_files_only: bool = True
    tokenizer_source: str = "adapter"
    chat_template_source: str = "adapter"
    use_fast_tokenizer: bool = True
    runtime_weight_mode: str = "auto"
    backend: str = "transformers"
    gguf_path: str | None = None
    gguf_repo_id: str | None = None
    gguf_filename: str | None = None
    gguf_lora_path: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_threads: int | None = None
    flash_attn: bool = True
    thinking_suppression: str = "auto"
    description: str = ""

    def signature(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)


@lru_cache(maxsize=16)
def _read_json_cached(path_str: str, mtime_ns: int):
    path = Path(path_str)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Failed to parse JSON config: {path}: {exc}") from exc


def _read_json_file(path: Path):
    if not path.exists():
        raise RegistryError(
            f"Config file not found: {path}. "
            f"Copy {MODEL_REGISTRY_EXAMPLE_PATH.name} to {MODEL_REGISTRY_PATH.name} and set local paths or Hugging Face repo IDs."
        )

    mtime_ns = path.stat().st_mtime_ns
    cached = _read_json_cached(str(path), mtime_ns)
    return copy.deepcopy(cached)


def _looks_like_remote_id(value: str):
    normalized = value.replace("\\", "/")
    return (
        "/" in normalized
        and not normalized.startswith("./")
        and not normalized.startswith("../")
        and not normalized.startswith("<")
        and normalized.count("/") == 1
    )


def _resolve_path(value: str | None, *, allow_remote_id=True):
    if not value:
        return None

    value = str(value).strip()
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)

    extension_candidate = (EXTENSION_ROOT / candidate).resolve()
    if extension_candidate.exists():
        return str(extension_candidate)

    if allow_remote_id and _looks_like_remote_id(value):
        return value

    return str(extension_candidate)


def _validate_models_object(data, registry_source: Path):
    models = data.get("models")
    if not isinstance(models, dict):
        raise RegistryError(f"{registry_source} must contain a top-level 'models' object.")
    return models


def _merge_model_entries(base_entry, override_entry):
    if isinstance(base_entry, dict) and isinstance(override_entry, dict):
        merged = copy.deepcopy(base_entry)
        merged.update(copy.deepcopy(override_entry))
        return merged
    return copy.deepcopy(override_entry)


def _get_registry_data():
    if not MODEL_REGISTRY_EXAMPLE_PATH.exists() and not MODEL_REGISTRY_PATH.exists():
        raise RegistryError(
            f"Config file not found: {MODEL_REGISTRY_PATH}. "
            f"Also missing fallback example config: {MODEL_REGISTRY_EXAMPLE_PATH}."
        )

    merged_models = {}
    registry_sources = []

    if MODEL_REGISTRY_EXAMPLE_PATH.exists():
        example_data = _read_json_file(MODEL_REGISTRY_EXAMPLE_PATH)
        example_models = _validate_models_object(example_data, MODEL_REGISTRY_EXAMPLE_PATH)
        merged_models.update(copy.deepcopy(example_models))
        registry_sources.append(MODEL_REGISTRY_EXAMPLE_PATH.name)

    if MODEL_REGISTRY_PATH.exists():
        user_data = _read_json_file(MODEL_REGISTRY_PATH)
        user_models = _validate_models_object(user_data, MODEL_REGISTRY_PATH)
        for key, entry in user_models.items():
            merged_models[key] = _merge_model_entries(merged_models.get(key), entry)
        registry_sources.append(MODEL_REGISTRY_PATH.name)

    return {
        "models": merged_models,
        "_registry_source": "+".join(registry_sources),
    }


def get_ui_model_choices(default_choices=None):
    default_choices = default_choices or ["none", "qwen2.5-7b-instruct", "qwen3.5-4b", "qwen3.5-9b"]
    try:
        registry_data = _get_registry_data()
        models = registry_data.get("models", {})
        model_keys = [
            key for key, entry in models.items()
            if isinstance(entry, dict) and entry.get("enabled", True)
        ]
        ordered = ["none"] + [key for key in model_keys if key != "none"]
        return ordered or default_choices
    except RegistryError:
        return default_choices


def get_model_spec(model_key: str):
    if not model_key or model_key == "none":
        return None

    registry_data = _get_registry_data()
    entry = registry_data.get("models", {}).get(model_key)
    if entry is None:
        raise RegistryError(
            f"Model key '{model_key}' is not defined in {MODEL_REGISTRY_PATH.name}."
        )

    if not entry.get("enabled", True):
        raise RegistryError(
            f"Model key '{model_key}' is present but disabled in {MODEL_REGISTRY_PATH.name}."
        )

    backend = str(entry.get("backend", "transformers") or "transformers").strip().lower()
    if backend not in {"transformers", "llama_cpp"}:
        raise RegistryError(
            f"Model key '{model_key}' has unsupported backend '{backend}' in {MODEL_REGISTRY_PATH.name}."
        )

    base_model_name_or_path = entry.get("base_model_name_or_path", "")
    if backend == "transformers" and not base_model_name_or_path:
        raise RegistryError(
            f"Model key '{model_key}' is missing 'base_model_name_or_path' in {MODEL_REGISTRY_PATH.name}."
        )

    gguf_path = _resolve_path(entry.get("gguf_path"), allow_remote_id=False)
    gguf_repo_id = entry.get("gguf_repo_id")
    gguf_filename = entry.get("gguf_filename")
    if backend == "llama_cpp" and not gguf_path and not (gguf_repo_id and gguf_filename):
        raise RegistryError(
            f"Model key '{model_key}' uses backend='llama_cpp' but has no 'gguf_path' "
            "or 'gguf_repo_id' + 'gguf_filename'."
        )

    thinking_suppression = str(entry.get("thinking_suppression", "auto") or "auto").strip().lower()
    if thinking_suppression not in {"auto", "no_think", "none"}:
        raise RegistryError(
            f"Model key '{model_key}' has unsupported thinking_suppression '{thinking_suppression}' "
            "in model_registry.json. Use auto, no_think, or none."
        )

    return ModelSpec(
        key=model_key,
        base_model_name_or_path=_resolve_path(base_model_name_or_path),
        adapter_path=_resolve_path(entry.get("adapter_path")),
        tokenizer_name_or_path=_resolve_path(entry.get("tokenizer_name_or_path")),
        cache_dir=_resolve_path(entry.get("cache_dir"), allow_remote_id=False),
        fallback_base_model_name_or_path=_resolve_path(entry.get("fallback_base_model_name_or_path")),
        fallback_adapter_path=_resolve_path(entry.get("fallback_adapter_path")),
        fallback_tokenizer_name_or_path=_resolve_path(entry.get("fallback_tokenizer_name_or_path")),
        allow_auto_download_missing=entry.get("allow_auto_download_missing", True),
        registry_source=registry_data.get("_registry_source", MODEL_REGISTRY_PATH.name),
        enabled=entry.get("enabled", True),
        load_in_4bit=entry.get("load_in_4bit", True),
        merge_lora_for_inference=entry.get("merge_lora_for_inference", False),
        bnb_4bit_quant_type=entry.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=entry.get("bnb_4bit_compute_dtype", "bfloat16"),
        use_double_quant=entry.get("use_double_quant", True),
        device_map=entry.get("device_map", "auto"),
        torch_dtype=entry.get("torch_dtype", "bfloat16"),
        trust_remote_code=entry.get("trust_remote_code", False),
        local_files_only=entry.get("local_files_only", False if backend == "llama_cpp" and gguf_repo_id else True),
        tokenizer_source=entry.get("tokenizer_source", "adapter"),
        chat_template_source=entry.get("chat_template_source", "adapter"),
        use_fast_tokenizer=entry.get("use_fast_tokenizer", True),
        runtime_weight_mode=entry.get("runtime_weight_mode", "auto"),
        backend=backend,
        gguf_path=gguf_path,
        gguf_repo_id=gguf_repo_id,
        gguf_filename=gguf_filename,
        gguf_lora_path=_resolve_path(entry.get("gguf_lora_path"), allow_remote_id=False),
        n_ctx=int(entry.get("n_ctx", 4096)),
        n_gpu_layers=int(entry.get("n_gpu_layers", -1)),
        n_batch=int(entry.get("n_batch", 512)),
        n_threads=entry.get("n_threads", None),
        flash_attn=bool(entry.get("flash_attn", True)),
        thinking_suppression=thinking_suppression,
        description=entry.get("description", ""),
    )


def get_generation_defaults():
    data = _read_json_file(GENERATION_DEFAULTS_PATH)
    return {
        "max_new_tokens": int(data.get("max_new_tokens", 128)),
        "num_candidates": int(data.get("num_candidates", 1)),
        "do_sample": bool(data.get("do_sample", True)),
        "temperature": float(data.get("temperature", 0.7)),
        "top_p": float(data.get("top_p", 0.9)),
        "top_k": data.get("top_k", None),
        "repetition_penalty": float(data.get("repetition_penalty", 1.0)),
        "input_template_mode": str(data.get("input_template_mode", "simple_chat_template")),
        "cache_implementation": str(data.get("cache_implementation", "dynamic")),
        "use_cache": bool(data.get("use_cache", True)),
        "enable_thinking": data.get("enable_thinking", False),
        "seed_mode": str(data.get("seed_mode", "random")),
        "llm_seed": int(data.get("llm_seed", 42)),
        "debug_compare_input_variants": bool(data.get("debug_compare_input_variants", False)),
    }
