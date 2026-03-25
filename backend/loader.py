import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path


class MissingDependencyError(Exception):
    pass


class ModelLoadError(Exception):
    pass


@dataclass
class LoadedModelBundle:
    spec: object
    tokenizer: object
    model: object
    load_debug: dict = field(default_factory=dict)


@dataclass
class ResolvedReference:
    original_reference: str | None
    source: str
    resolved_reference: str | None = None
    resolved_local_path: str | None = None
    effective_reference: str | None = None
    fallback_reference: str | None = None
    missing_local_fallback_used: bool = False
    local_cache_hit: bool | None = None
    download_started: bool = False
    download_finished: bool = False
    download_failed: bool = False


ADAPTER_REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]
BASE_MODEL_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
]
TOKENIZER_ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "*.model",
    "chat_template.jinja",
]


def _import_torch():
    try:
        import torch
    except Exception as exc:
        raise MissingDependencyError(f"torch is required: {exc}") from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise MissingDependencyError(f"transformers is required: {exc}") from exc
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _import_peft():
    try:
        from peft import LoraConfig, PeftModel
    except Exception as exc:
        raise MissingDependencyError(f"peft is required: {exc}") from exc
    return LoraConfig, PeftModel


def _import_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
    except Exception as exc:
        raise MissingDependencyError(f"huggingface_hub is required for remote model IDs: {exc}") from exc
    return snapshot_download, try_to_load_from_cache


def _resolve_dtype(torch, value: str):
    normalized = (value or "bfloat16").strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ModelLoadError(f"Unsupported dtype value: {value}")
    return mapping[normalized]


def _safe_log(logger, message):
    if logger:
        logger(message)


def _looks_like_remote_id(value: str):
    normalized = str(value).replace("\\", "/")
    return (
        "/" in normalized
        and not normalized.startswith("./")
        and not normalized.startswith("../")
        and normalized.count("/") == 1
    )


def _read_probe(path: Path, *, binary=False):
    try:
        if binary:
            with path.open("rb") as handle:
                handle.read(16)
        else:
            with path.open("r", encoding="utf-8") as handle:
                handle.read(256)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _inspect_adapter_dir(adapter_local_path: str | None):
    if not adapter_local_path:
        return {}

    adapter_dir = Path(adapter_local_path)
    targets = {
        "adapter_config.json": {"binary": False},
        "adapter_model.safetensors": {"binary": True},
        "chat_template.jinja": {"binary": False},
        "tokenizer.json": {"binary": False},
        "tokenizer_config.json": {"binary": False},
    }

    report = {}
    for filename, options in targets.items():
        file_path = adapter_dir / filename
        exists = file_path.exists()
        readable = False
        error = ""
        if exists:
            readable, error = _read_probe(file_path, binary=options["binary"])
        report[filename] = {
            "path": str(file_path),
            "exists": exists,
            "readable": readable,
            "error": error,
        }
    return report


def _log_adapter_report(report, logger):
    if not report:
        _safe_log(logger, "adapter_file_presence=<no adapter configured>")
        return

    joined = " ".join(f"{name}={meta['exists']}" for name, meta in report.items())
    _safe_log(logger, f"adapter_file_presence {joined}")

    for name, meta in report.items():
        if not meta["exists"]:
            _safe_log(logger, f"adapter_file_warning missing={name} path={meta['path']}")
        elif not meta["readable"]:
            _safe_log(
                logger,
                f"adapter_file_warning unreadable={name} path={meta['path']} error={meta['error']}",
            )


def _format_hf_error(exc, label: str, reference: str):
    name = type(exc).__name__
    message = str(exc)
    lowered = message.lower()

    if name == "RepositoryNotFoundError" or "repository not found" in lowered:
        return f"{label} Hugging Face repo not found: {reference}"
    if name == "GatedRepoError" or "gated" in lowered:
        return f"{label} Hugging Face repo requires access approval/authentication: {reference}"
    if name == "RevisionNotFoundError":
        return f"{label} Hugging Face revision not found: {reference}"
    if name == "EntryNotFoundError":
        return f"{label} required file not found in Hugging Face repo: {reference}"
    if name == "LocalEntryNotFoundError":
        return (
            f"{label} was not found in the local Hugging Face cache and network download was not available: "
            f"{reference}"
        )
    if "401" in lowered or "403" in lowered or "authentication" in lowered or "authorized" in lowered:
        return f"{label} Hugging Face access failed (auth may be required): {reference}"
    if "no space left" in lowered or "disk full" in lowered or "not enough space" in lowered:
        return f"{label} download failed because the disk appears to be full: {reference}"
    if "connection" in lowered or "timed out" in lowered or "temporary failure" in lowered:
        return f"{label} download failed due to a network error: {reference}"
    return f"{label} download failed for {reference}: {message}"


def _probe_cache_hit(try_to_load_from_cache, reference: str, cache_dir: str | None, probe_filenames):
    for filename in probe_filenames or []:
        try:
            cached_path = try_to_load_from_cache(
                reference,
                filename,
                cache_dir=cache_dir,
                repo_type="model",
            )
        except Exception:
            cached_path = None
        if isinstance(cached_path, str) and Path(cached_path).exists():
            return True
    return False


def _get_staging_root(cache_dir: str | None):
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "huggingface"


def _build_staging_dir(cache_dir: str | None, label: str, reference: str):
    safe_reference = reference.replace("\\", "--").replace("/", "--").replace(":", "_")
    return str((_get_staging_root(cache_dir) / "sd_forge_llm_prompt_gen" / label / safe_reference).resolve())


def _resolve_reference(
    reference: str | None,
    *,
    label: str,
    cache_dir: str | None,
    local_files_only: bool,
    logger=None,
    allow_patterns=None,
    probe_filenames=None,
    download_strategy="snapshot",
    fallback_reference: str | None = None,
    allow_missing_fallback=False,
):
    if not reference:
        return ResolvedReference(original_reference=reference, source="none")

    reference = str(reference).strip()
    candidate = Path(reference)
    if candidate.exists():
        resolved_local = str(candidate.resolve())
        _safe_log(logger, f"{label}_source=local")
        return ResolvedReference(
            original_reference=reference,
            source="local",
            resolved_reference=resolved_local,
            resolved_local_path=resolved_local,
            effective_reference=reference,
            local_cache_hit=True,
        )

    if _looks_like_remote_id(reference):
        snapshot_download, try_to_load_from_cache = _import_huggingface_hub()
        local_cache_hit = _probe_cache_hit(
            try_to_load_from_cache,
            reference,
            cache_dir,
            probe_filenames or ["config.json"],
        )
        _safe_log(logger, f"{label}_source=huggingface")
        if download_strategy == "defer_to_loader":
            return ResolvedReference(
                original_reference=reference,
                source="huggingface",
                resolved_reference=reference,
                resolved_local_path=None,
                effective_reference=reference,
                local_cache_hit=local_cache_hit,
            )

        staging_dir = _build_staging_dir(cache_dir, label, reference)
        _safe_log(
            logger,
            f"{label}_download_started repo_id={reference} cache_dir={cache_dir} "
            f"local_files_only={local_files_only} local_cache_hit={local_cache_hit} "
            f"staging_dir={staging_dir}",
        )
        try:
            snapshot_path = snapshot_download(
                repo_id=reference,
                repo_type="model",
                cache_dir=cache_dir,
                local_dir=staging_dir,
                local_files_only=local_files_only,
                allow_patterns=allow_patterns,
            )
        except Exception as exc:
            _safe_log(logger, f"{label}_download_failed repo_id={reference} error={exc}")
            raise ModelLoadError(_format_hf_error(exc, label, reference)) from exc

        _safe_log(logger, f"{label}_download_finished repo_id={reference} snapshot_path={snapshot_path}")
        return ResolvedReference(
            original_reference=reference,
            source="huggingface",
            resolved_reference=snapshot_path,
            resolved_local_path=snapshot_path,
            effective_reference=reference,
            local_cache_hit=local_cache_hit,
            download_started=True,
            download_finished=True,
        )

    if allow_missing_fallback and fallback_reference:
        _safe_log(
            logger,
            f"{label}_missing_local_primary reference={reference} fallback_reference={fallback_reference}",
        )
        fallback_result = _resolve_reference(
            fallback_reference,
            label=label,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            logger=logger,
            allow_patterns=allow_patterns,
            probe_filenames=probe_filenames,
            download_strategy=download_strategy,
        )
        fallback_result.original_reference = reference
        fallback_result.fallback_reference = fallback_reference
        fallback_result.missing_local_fallback_used = True
        if fallback_result.effective_reference is None:
            fallback_result.effective_reference = fallback_reference
        return fallback_result

    if candidate.is_absolute():
        resolved_local = str(candidate)
    else:
        resolved_local = str(candidate.resolve())
    raise ModelLoadError(f"{label} path not found: {resolved_local}")


def _validate_resolved_paths(base_reference, adapter_reference, adapter_report):
    if base_reference.source == "local" and (
        not base_reference.resolved_local_path or not Path(base_reference.resolved_local_path).exists()
    ):
        raise ModelLoadError(f"Base model path not found: {base_reference.resolved_reference}")

    if adapter_reference and adapter_reference.source != "none":
        if not adapter_reference.resolved_local_path or not Path(adapter_reference.resolved_local_path).exists():
            raise ModelLoadError(f"Adapter path not found: {adapter_reference.resolved_reference}")

        config_report = adapter_report.get("adapter_config.json", {})
        weight_report = adapter_report.get("adapter_model.safetensors", {})
        if not config_report.get("exists"):
            raise ModelLoadError(
                f"adapter_config.json not found under adapter path: {adapter_reference.resolved_reference}"
            )
        if not config_report.get("readable"):
            raise ModelLoadError(
                f"adapter_config.json could not be read under adapter path: {adapter_reference.resolved_reference}"
            )
        if not weight_report.get("exists"):
            raise ModelLoadError(
                f"adapter_model.safetensors not found under adapter path: {adapter_reference.resolved_reference}"
            )
        if not weight_report.get("readable"):
            raise ModelLoadError(
                f"adapter_model.safetensors could not be read under adapter path: {adapter_reference.resolved_reference}"
            )


def _load_compatible_lora_config(adapter_local_path: str, LoraConfig):
    adapter_config_path = Path(adapter_local_path) / "adapter_config.json"
    try:
        raw_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelLoadError(f"Failed to read adapter config: {adapter_config_path}: {exc}") from exc

    signature = inspect.signature(LoraConfig.__init__)
    valid_keys = set(signature.parameters.keys()) - {"self"}
    filtered_config = {key: value for key, value in raw_config.items() if key in valid_keys}

    try:
        config = LoraConfig(**filtered_config)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to build compatible LoRA config from {adapter_config_path}: {exc}"
        ) from exc

    return config, raw_config


def _resolve_tokenizer_reference(spec, base_reference, adapter_reference, adapter_report, logger=None):
    if spec.tokenizer_name_or_path:
        effective_tokenizer_reference = spec.tokenizer_name_or_path
        if (
            spec.allow_auto_download_missing
            and spec.fallback_tokenizer_name_or_path
            and not _looks_like_remote_id(spec.tokenizer_name_or_path)
            and not Path(spec.tokenizer_name_or_path).exists()
        ):
            effective_tokenizer_reference = spec.fallback_tokenizer_name_or_path

        if spec.adapter_path and effective_tokenizer_reference == spec.adapter_path and adapter_reference:
            return adapter_reference.resolved_reference, "adapter"
        if effective_tokenizer_reference == spec.base_model_name_or_path:
            return base_reference.resolved_reference, "base"

        tokenizer_reference = _resolve_reference(
            effective_tokenizer_reference,
            label="tokenizer",
            cache_dir=spec.cache_dir,
            local_files_only=spec.local_files_only,
            logger=logger,
            allow_patterns=TOKENIZER_ALLOW_PATTERNS,
            probe_filenames=["tokenizer.json", "tokenizer_config.json"],
        )
        return tokenizer_reference.resolved_reference, "custom"

    if spec.tokenizer_source == "adapter" and adapter_reference and adapter_report.get("tokenizer.json", {}).get("exists"):
        return adapter_reference.resolved_reference, "adapter"

    return base_reference.resolved_reference, "base"


def _load_tokenizer(AutoTokenizer, spec, adapter_report, base_reference, adapter_reference, logger=None):
    tokenizer_load_path, tokenizer_source = _resolve_tokenizer_reference(
        spec,
        base_reference,
        adapter_reference,
        adapter_report,
        logger=logger,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            use_fast=bool(spec.use_fast_tokenizer),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=spec.local_files_only,
            cache_dir=spec.cache_dir,
        )
    except Exception as exc:
        raise ModelLoadError(f"Failed to load tokenizer from '{tokenizer_load_path}': {exc}") from exc

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, tokenizer_load_path, tokenizer_source


def _load_tokenizer_for_template(AutoTokenizer, path, spec):
    try:
        return AutoTokenizer.from_pretrained(
            path,
            use_fast=bool(spec.use_fast_tokenizer),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=spec.local_files_only,
            cache_dir=spec.cache_dir,
        )
    except Exception:
        return None


def _apply_chat_template_source(
    tokenizer,
    AutoTokenizer,
    spec,
    adapter_report,
    base_reference,
    adapter_reference,
    logger=None,
):
    requested_source = (spec.chat_template_source or "adapter").strip().lower()
    selected_source = "prompt_builder"
    chat_template_path = None
    chat_template_loaded = False

    if requested_source == "prompt_builder":
        try:
            tokenizer.chat_template = None
        except Exception:
            pass
        return {
            "requested_chat_template_source": requested_source,
            "chat_template_source": selected_source,
            "chat_template_path": chat_template_path,
            "chat_template_loaded": chat_template_loaded,
            "use_adapter_chat_template": False,
            "use_base_tokenizer_chat_template": False,
        }

    if requested_source == "adapter" and adapter_reference and adapter_reference.resolved_local_path:
        adapter_template = Path(adapter_reference.resolved_local_path) / "chat_template.jinja"
        if adapter_report.get("chat_template.jinja", {}).get("readable"):
            tokenizer.chat_template = adapter_template.read_text(encoding="utf-8")
            selected_source = "adapter"
            chat_template_path = str(adapter_template)
            chat_template_loaded = True
        elif getattr(tokenizer, "chat_template", None):
            selected_source = "adapter_tokenizer"
            chat_template_path = adapter_reference.resolved_reference
            chat_template_loaded = True
        else:
            _safe_log(
                logger,
                "chat_template_warning requested adapter template but none was found; using prompt_builder fallback",
            )

    if requested_source == "base" and selected_source == "prompt_builder":
        base_tokenizer = _load_tokenizer_for_template(AutoTokenizer, base_reference.resolved_reference, spec)
        if base_tokenizer is not None and getattr(base_tokenizer, "chat_template", None):
            tokenizer.chat_template = base_tokenizer.chat_template
            selected_source = "base"
            chat_template_path = base_reference.resolved_reference
            chat_template_loaded = True
        else:
            _safe_log(logger, "chat_template_warning requested base template but none was found; using prompt_builder fallback")

    if selected_source == "prompt_builder":
        try:
            tokenizer.chat_template = None
        except Exception:
            pass

    return {
        "requested_chat_template_source": requested_source,
        "chat_template_source": selected_source,
        "chat_template_path": chat_template_path,
        "chat_template_loaded": chat_template_loaded,
        "use_adapter_chat_template": selected_source.startswith("adapter"),
        "use_base_tokenizer_chat_template": selected_source == "base",
    }


def _extract_active_adapter(model):
    active_adapter = getattr(model, "active_adapter", None)
    try:
        if callable(active_adapter):
            active_adapter = active_adapter()
    except Exception:
        active_adapter = getattr(model, "active_adapter", None)

    if active_adapter is None:
        active_adapters = getattr(model, "active_adapters", None)
        try:
            if callable(active_adapters):
                active_adapter = active_adapters()
            else:
                active_adapter = active_adapters
        except Exception:
            active_adapter = None

    return active_adapter


def _inspect_model_state(model, PeftModel):
    peft_config = getattr(model, "peft_config", {})
    if isinstance(peft_config, dict):
        peft_keys = list(peft_config.keys())
    else:
        peft_keys = []

    merged_state = False
    for candidate in (
        getattr(model, "merged", False),
        bool(getattr(model, "merged_adapters", [])),
        getattr(getattr(model, "base_model", None), "merged", False),
        bool(getattr(getattr(model, "base_model", None), "merged_adapters", [])),
    ):
        if candidate:
            merged_state = True
            break

    return {
        "is_peft_model": isinstance(model, PeftModel),
        "peft_adapters": peft_keys,
        "active_adapter": _extract_active_adapter(model),
        "merged_state": merged_state,
    }


def _tokenizer_debug_info(tokenizer, tokenizer_load_path, tokenizer_source, chat_template_info):
    tokenizer_config_path = Path(tokenizer_load_path) / "tokenizer_config.json"
    tokenizer_config_source = str(tokenizer_config_path) if tokenizer_config_path.exists() else "unknown"

    return {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", tokenizer_load_path),
        "tokenizer_source": tokenizer_source,
        "tokenizer_source_path": tokenizer_load_path,
        "tokenizer_config_source": tokenizer_config_source,
        "tokenizer_use_fast": bool(getattr(tokenizer, "is_fast", False)),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        **chat_template_info,
    }


def _log_load_debug(load_debug, logger):
    if not logger:
        return

    logger(f"base_model_source={load_debug['base_model_source']}")
    logger(f"adapter_source={load_debug['adapter_source']}")
    logger(f"base_model_name_or_path={load_debug['base_model_name_or_path']}")
    logger(f"adapter_path={load_debug['adapter_path']}")
    logger(f"resolved_base_model_reference={load_debug['resolved_base_model_reference']}")
    logger(f"resolved_adapter_reference={load_debug['resolved_adapter_reference']}")
    logger(f"effective_base_model_reference={load_debug['effective_base_model_reference']}")
    logger(f"effective_adapter_reference={load_debug['effective_adapter_reference']}")
    logger(f"base_fallback_reference={load_debug['base_fallback_reference']}")
    logger(f"adapter_fallback_reference={load_debug['adapter_fallback_reference']}")
    logger(f"base_missing_local_fallback_used={load_debug['base_missing_local_fallback_used']}")
    logger(f"adapter_missing_local_fallback_used={load_debug['adapter_missing_local_fallback_used']}")
    logger(f"cache_dir={load_debug['cache_dir']}")
    logger(
        "local_cache_hit "
        f"base_model={load_debug['base_local_cache_hit']} "
        f"adapter={load_debug['adapter_local_cache_hit']}"
    )
    logger(f"base_download_started={load_debug['base_download_started']}")
    logger(f"base_download_finished={load_debug['base_download_finished']}")
    logger(f"base_download_failed={load_debug['base_download_failed']}")
    logger(f"adapter_download_started={load_debug['adapter_download_started']}")
    logger(f"adapter_download_finished={load_debug['adapter_download_finished']}")
    logger(f"adapter_download_failed={load_debug['adapter_download_failed']}")
    logger(f"base_model_class={load_debug['base_model_class']}")
    logger(f"final_model_class={load_debug['final_model_class']}")
    logger(f"is_peft_model={load_debug['is_peft_model']}")
    logger(f"peft_adapters={load_debug['peft_adapters']}")
    logger(f"active_adapter={load_debug['active_adapter']}")
    logger(f"merged_state={load_debug['merged_state']}")
    logger(f"adapter_config_loaded={load_debug['adapter_config_loaded']}")
    logger(f"adapter_weights_loaded={load_debug['adapter_weights_loaded']}")
    logger(f"tokenizer_class={load_debug['tokenizer_class']}")
    logger(f"tokenizer_name_or_path={load_debug['tokenizer_name_or_path']}")
    logger(f"tokenizer_source={load_debug['tokenizer_source']}")
    logger(f"tokenizer_source_path={load_debug['tokenizer_source_path']}")
    logger(f"tokenizer_config_source={load_debug['tokenizer_config_source']}")
    logger(f"chat_template_source={load_debug['chat_template_source']}")
    logger(f"requested_chat_template_source={load_debug['requested_chat_template_source']}")
    logger(f"use_adapter_chat_template={load_debug['use_adapter_chat_template']}")
    logger(f"use_base_tokenizer_chat_template={load_debug['use_base_tokenizer_chat_template']}")
    logger(f"chat_template_loaded={load_debug['chat_template_loaded']}")
    logger(f"chat_template_path={load_debug['chat_template_path']}")
    logger(f"eos_token={load_debug['eos_token']}")
    logger(f"bos_token={load_debug['bos_token']}")
    logger(f"pad_token={load_debug['pad_token']}")
    logger(f"tokenizer_use_fast={load_debug['tokenizer_use_fast']}")

    if load_debug["adapter_path"] and not load_debug["is_peft_model"]:
        logger("LORA_NOT_ACTIVE PEFT_MODEL_NOT_ATTACHED base model only inference")


def load_model_bundle(spec, logger=None):
    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_transformers()
    LoraConfig, PeftModel = _import_peft()

    base_reference = _resolve_reference(
        spec.base_model_name_or_path,
        label="base_model",
        cache_dir=spec.cache_dir,
        local_files_only=spec.local_files_only,
        logger=logger,
        allow_patterns=BASE_MODEL_ALLOW_PATTERNS,
        probe_filenames=["config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"],
        download_strategy="snapshot",
        fallback_reference=spec.fallback_base_model_name_or_path,
        allow_missing_fallback=bool(spec.allow_auto_download_missing),
    )
    adapter_reference = _resolve_reference(
        spec.adapter_path,
        label="adapter",
        cache_dir=spec.cache_dir,
        local_files_only=spec.local_files_only,
        logger=logger,
        allow_patterns=ADAPTER_REQUIRED_FILES,
        probe_filenames=["adapter_config.json", "adapter_model.safetensors"],
        fallback_reference=spec.fallback_adapter_path,
        allow_missing_fallback=bool(spec.allow_auto_download_missing),
    )

    adapter_report = _inspect_adapter_dir(adapter_reference.resolved_local_path)
    _log_adapter_report(adapter_report, logger)
    _validate_resolved_paths(base_reference, adapter_reference, adapter_report)

    tokenizer, tokenizer_load_path, tokenizer_source = _load_tokenizer(
        AutoTokenizer,
        spec,
        adapter_report,
        base_reference,
        adapter_reference,
        logger=logger,
    )
    chat_template_info = _apply_chat_template_source(
        tokenizer,
        AutoTokenizer,
        spec,
        adapter_report,
        base_reference,
        adapter_reference,
        logger=logger,
    )

    model_kwargs = {
        "device_map": spec.device_map,
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": True if base_reference.resolved_local_path else spec.local_files_only,
        "low_cpu_mem_usage": True,
        "cache_dir": spec.cache_dir,
    }

    if spec.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=spec.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_resolve_dtype(torch, spec.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=bool(spec.use_double_quant),
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = _resolve_dtype(torch, spec.torch_dtype)
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(torch, spec.torch_dtype)

    base_download_started = False
    base_download_finished = False
    base_download_failed = False

    if base_reference.source == "huggingface":
        base_download_started = bool(base_reference.download_started)
        base_download_finished = bool(base_reference.download_finished)
        base_download_failed = bool(base_reference.download_failed)
        _safe_log(
            logger,
            f"base_download_wait_complete reference={base_reference.original_reference} "
            f"resolved_local_path={base_reference.resolved_local_path} "
            f"cache_dir={spec.cache_dir} local_files_only={spec.local_files_only} "
            f"local_cache_hit={base_reference.local_cache_hit}",
        )
    try:
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_reference.resolved_reference,
            **model_kwargs,
        )
    except Exception as exc:
        if base_reference.source == "huggingface":
            base_download_failed = True
            _safe_log(
                logger,
                f"base_download_failed reference={base_reference.original_reference} error={exc}",
            )
            raise ModelLoadError(
                _format_hf_error(exc, "base_model", spec.base_model_name_or_path)
            ) from exc
        raise ModelLoadError(f"Failed to load base model '{spec.base_model_name_or_path}': {exc}") from exc

    base_model_class = type(base_model_obj).__name__
    model = base_model_obj
    adapter_config_loaded = False
    adapter_weights_loaded = False

    if adapter_reference and adapter_reference.source != "none":
        adapter_config, _raw_config = _load_compatible_lora_config(
            adapter_reference.resolved_local_path,
            LoraConfig,
        )
        adapter_config_loaded = True
        try:
            model = PeftModel.from_pretrained(
                model,
                adapter_reference.resolved_reference,
                is_trainable=False,
                config=adapter_config,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load adapter '{spec.adapter_path}': {exc}"
            ) from exc

    final_model_class = type(model).__name__
    model.eval()

    model_state = _inspect_model_state(model, PeftModel)
    adapter_weights_loaded = bool(
        adapter_reference
        and adapter_reference.source != "none"
        and adapter_report.get("adapter_model.safetensors", {}).get("readable")
        and model_state["is_peft_model"]
    )

    load_debug = {
        "base_model_source": base_reference.source,
        "adapter_source": adapter_reference.source,
        "base_model_name_or_path": spec.base_model_name_or_path,
        "adapter_path": spec.adapter_path,
        "resolved_base_model_reference": base_reference.resolved_reference,
        "resolved_adapter_reference": adapter_reference.resolved_reference,
        "effective_base_model_reference": base_reference.effective_reference,
        "effective_adapter_reference": adapter_reference.effective_reference,
        "base_fallback_reference": base_reference.fallback_reference,
        "adapter_fallback_reference": adapter_reference.fallback_reference,
        "base_missing_local_fallback_used": base_reference.missing_local_fallback_used,
        "adapter_missing_local_fallback_used": adapter_reference.missing_local_fallback_used,
        "cache_dir": spec.cache_dir,
        "base_local_cache_hit": base_reference.local_cache_hit,
        "adapter_local_cache_hit": adapter_reference.local_cache_hit,
        "base_download_started": base_download_started,
        "base_download_finished": base_download_finished,
        "base_download_failed": base_download_failed,
        "adapter_download_started": adapter_reference.download_started,
        "adapter_download_finished": adapter_reference.download_finished,
        "adapter_download_failed": adapter_reference.download_failed,
        "adapter_config_loaded": adapter_config_loaded,
        "adapter_weights_loaded": adapter_weights_loaded,
        "adapter_files": adapter_report,
        "base_model_class": base_model_class,
        "final_model_class": final_model_class,
        **model_state,
        **_tokenizer_debug_info(
            tokenizer,
            tokenizer_load_path,
            tokenizer_source,
            chat_template_info,
        ),
    }

    _log_load_debug(load_debug, logger)

    return LoadedModelBundle(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        load_debug=load_debug,
    )
