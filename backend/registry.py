import json
from dataclasses import asdict, dataclass
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
    base_model_name_or_path: str
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
    description: str = ""

    def signature(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)


def _read_json_file(path: Path):
    if not path.exists():
        raise RegistryError(
            f"Config file not found: {path}. "
            f"Copy {MODEL_REGISTRY_EXAMPLE_PATH.name} to {MODEL_REGISTRY_PATH.name} and set local paths or Hugging Face repo IDs."
        )

    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Failed to parse JSON config: {path}: {exc}") from exc


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


def _get_registry_data():
    registry_source = MODEL_REGISTRY_PATH
    if not MODEL_REGISTRY_PATH.exists():
        if not MODEL_REGISTRY_EXAMPLE_PATH.exists():
            raise RegistryError(
                f"Config file not found: {MODEL_REGISTRY_PATH}. "
                f"Also missing fallback example config: {MODEL_REGISTRY_EXAMPLE_PATH}."
            )
        registry_source = MODEL_REGISTRY_EXAMPLE_PATH

    data = _read_json_file(registry_source)
    models = data.get("models")
    if not isinstance(models, dict):
        raise RegistryError(f"{registry_source} must contain a top-level 'models' object.")
    data["_registry_source"] = registry_source.name
    return data


def get_ui_model_choices(default_choices=None):
    default_choices = default_choices or ["none", "qwen2.5-7b-instruct", "qwen3.5-4b", "qwen3.5-9b"]
    try:
        registry_data = _get_registry_data()
        model_keys = list(registry_data.get("models", {}).keys())
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

    base_model_name_or_path = entry.get("base_model_name_or_path")
    if not base_model_name_or_path:
        raise RegistryError(
            f"Model key '{model_key}' is missing 'base_model_name_or_path' in {MODEL_REGISTRY_PATH.name}."
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
        local_files_only=entry.get("local_files_only", True),
        tokenizer_source=entry.get("tokenizer_source", "adapter"),
        chat_template_source=entry.get("chat_template_source", "adapter"),
        use_fast_tokenizer=entry.get("use_fast_tokenizer", True),
        description=entry.get("description", ""),
    )


def get_generation_defaults():
    data = _read_json_file(GENERATION_DEFAULTS_PATH)
    return {
        "max_new_tokens": int(data.get("max_new_tokens", 128)),
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
