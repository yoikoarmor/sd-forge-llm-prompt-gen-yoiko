import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .compat import (
    QWEN35_MIN_TRANSFORMERS,
    TORCH_MIN_FOR_QWEN35,
    apply_runtime_compat_shims,
    format_dependency_versions,
    get_dependency_versions,
    torch_supports_modern_transformers,
    transformers_supports_qwen35,
)


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
        _safe_log(
            logger,
            f"tokenizer_load_warning source={tokenizer_source} path={tokenizer_load_path} error={exc}",
        )
        if tokenizer_source != "base":
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_reference.resolved_reference,
                    use_fast=bool(spec.use_fast_tokenizer),
                    trust_remote_code=spec.trust_remote_code,
                    local_files_only=True if base_reference.resolved_local_path else spec.local_files_only,
                    cache_dir=spec.cache_dir,
                )
                tokenizer_load_path = base_reference.resolved_reference
                tokenizer_source = "base_fallback"
                _safe_log(
                    logger,
                    f"tokenizer_load_fallback source=base path={tokenizer_load_path}",
                )
            except Exception as fallback_exc:
                raise ModelLoadError(
                    f"Failed to load tokenizer from '{tokenizer_load_path}': {exc}. "
                    f"Base tokenizer fallback also failed: {fallback_exc}"
                ) from fallback_exc
        else:
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


def _model_is_loaded_in_4bit(model):
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(getattr(model, "base_model", None), "is_loaded_in_4bit", False)
    )


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
    logger(f"model_load_seconds={load_debug['model_load_seconds']:.3f}")
    logger(
        f"merge_lora_for_inference_requested={load_debug['merge_lora_for_inference_requested']}"
    )
    logger(
        f"merge_lora_for_inference_applied={load_debug['merge_lora_for_inference_applied']}"
    )
    logger(f"merge_lora_seconds={load_debug['merge_lora_seconds']:.3f}")
    logger(f"merge_lora_skipped_reason={load_debug['merge_lora_skipped_reason']}")
    logger(f"quantization_fallback_used={load_debug['quantization_fallback_used']}")
    logger(f"model_load_type_requested={load_debug['model_load_type_requested']}")
    logger(f"model_load_type_effective={load_debug['model_load_type_effective']}")
    logger(f"model_load_type_reason={load_debug['model_load_type_reason']}")
    logger(f"model_load_total_vram_gib={load_debug['model_load_total_vram_gib']}")
    logger(f"cpu_offload_retry_used={load_debug['cpu_offload_retry_used']}")
    logger(f"cpu_offload_folder={load_debug['cpu_offload_folder']}")
    logger(f"cpu_offload_max_memory={load_debug['cpu_offload_max_memory']}")
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

    if (
        load_debug["adapter_path"]
        and not load_debug["is_peft_model"]
        and not load_debug["merged_state"]
    ):
        logger("LORA_NOT_ACTIVE PEFT_MODEL_NOT_ATTACHED base model only inference")


def _load_base_model_with_compat(
    AutoModelForCausalLM,
    reference: str,
    model_kwargs: dict,
    *,
    logger=None,
):
    try:
        return AutoModelForCausalLM.from_pretrained(reference, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in model_kwargs:
            raise
        error_text = str(exc)
        if "dtype" not in error_text or "unexpected keyword argument" not in error_text:
            raise

        legacy_kwargs = dict(model_kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        _safe_log(logger, "base_model_load_retry using torch_dtype compatibility path")
        return AutoModelForCausalLM.from_pretrained(reference, **legacy_kwargs)


def _should_retry_without_4bit(spec, exc: Exception):
    if not spec.load_in_4bit:
        return False
    error_text = str(exc).lower()
    return "bitsandbytes" in error_text and "4-bit" in error_text


def _is_low_vram_dispatch_error(exc: Exception):
    error_text = str(exc).lower()
    markers = (
        "some modules are dispatched on the cpu or the disk",
        "make sure you have enough gpu ram to fit the quantized model",
        "llm_int8_enable_fp32_cpu_offload=true",
    )
    return any(marker in error_text for marker in markers)


def _should_retry_with_cpu_offload(spec, exc: Exception):
    return bool(spec.load_in_4bit) and _is_low_vram_dispatch_error(exc)


def _detect_available_cpu_memory_bytes():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except Exception:
        pass

    return None


def _format_mib_string(num_bytes: int | None):
    if not num_bytes or num_bytes <= 0:
        return None
    mib = max(1, int(num_bytes // (1024 * 1024)))
    return f"{mib}MiB"


def _build_low_vram_max_memory(torch):
    max_memory = {}

    try:
        if torch.cuda.is_available():
            total_vram = int(torch.cuda.get_device_properties(0).total_memory)
            reserve = max(768 * 1024 * 1024, total_vram // 10)
            gpu_budget = max(1024 * 1024 * 1024, total_vram - reserve)
            gpu_budget_string = _format_mib_string(gpu_budget)
            if gpu_budget_string:
                max_memory[0] = gpu_budget_string
    except Exception:
        pass

    cpu_available = _detect_available_cpu_memory_bytes()
    cpu_budget_string = _format_mib_string(cpu_available)
    if cpu_budget_string:
        max_memory["cpu"] = cpu_budget_string

    return max_memory or None


def _build_cpu_offload_retry_kwargs(spec, model_kwargs, BitsAndBytesConfig, torch):
    retry_kwargs = dict(model_kwargs)
    retry_kwargs["device_map"] = "auto"
    retry_kwargs["offload_state_dict"] = True
    retry_kwargs["offload_buffers"] = True

    max_memory = _build_low_vram_max_memory(torch)
    if max_memory:
        retry_kwargs["max_memory"] = max_memory

    offload_folder = Path(_build_staging_dir(spec.cache_dir, "offload", spec.base_model_name_or_path))
    offload_folder.mkdir(parents=True, exist_ok=True)
    retry_kwargs["offload_folder"] = str(offload_folder)

    retry_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=spec.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_resolve_dtype(torch, spec.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=bool(spec.use_double_quant),
        llm_int8_enable_fp32_cpu_offload=True,
    )

    return retry_kwargs, str(offload_folder), max_memory


def _normalize_model_load_type(value: str | None):
    normalized = (value or "GPU_load").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"gpu_load", "prefer_gpu"}:
        return "GPU_load"
    if normalized in {"gpu_cpu_load", "always_cpu_offload"}:
        return "GPU_CPU_load"
    return "GPU_load"


def _detect_total_vram_bytes(torch):
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


def _select_effective_model_load_type(spec, torch):
    requested = _normalize_model_load_type(getattr(spec, "model_load_type", "GPU_load"))
    total_vram_bytes = _detect_total_vram_bytes(torch)
    total_vram_gib = None
    if total_vram_bytes:
        total_vram_gib = total_vram_bytes / (1024 ** 3)

    if requested != "auto":
        return requested, requested, total_vram_gib, "user_selected"

    if not torch.cuda.is_available():
        return requested, "GPU_CPU_load", total_vram_gib, "cuda_unavailable"

    model_ref = str(getattr(spec, "base_model_name_or_path", "") or "")
    if "Qwen/Qwen3.5-9B" in model_ref and total_vram_gib is not None and total_vram_gib <= 10.5:
        return requested, "GPU_CPU_load", total_vram_gib, "auto_low_vram_qwen3.5_9b"

    return requested, "GPU_load", total_vram_gib, "auto_default_gpu_load"


def load_model_bundle(spec, logger=None):
    load_started_at = time.perf_counter()
    apply_runtime_compat_shims(logger=logger)
    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_transformers()
    LoraConfig, PeftModel = _import_peft()

    if "Qwen/Qwen3.5-" in str(spec.base_model_name_or_path) and not transformers_supports_qwen35():
        installed = get_dependency_versions().get("transformers") or "<missing>"
        installed_torch = get_dependency_versions().get("torch") or "<missing>"
        if not torch_supports_modern_transformers():
            raise ModelLoadError(
                "Qwen3.5 models are not supported on this legacy Forge runtime. "
                f"Installed torch={installed_torch}, but the modern Qwen3.5 dependency profile needs torch>={TORCH_MIN_FOR_QWEN35}. "
                f"Installed transformers={installed}, required>={QWEN35_MIN_TRANSFORMERS}. "
                "This extension intentionally keeps older dependencies on such environments so Forge itself does not break. "
                "Use qwen2.5-7b-instruct on this PC, or upgrade Forge/PyTorch first if you need Qwen3.5. "
                f"Current dependency set: {format_dependency_versions()}"
            )
        raise ModelLoadError(
            "Qwen3.5 models require a newer Transformers build. "
            f"Installed transformers={installed}, required>={QWEN35_MIN_TRANSFORMERS}. "
            "Run the extension install step once without --skip-install, or update the Forge venv to the pinned versions. "
            f"Current dependency set: {format_dependency_versions()}"
        )

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

    requested_model_load_type, effective_model_load_type, total_vram_gib, model_load_type_reason = (
        _select_effective_model_load_type(spec, torch)
    )
    _safe_log(
        logger,
        "model_load_type "
        f"requested={requested_model_load_type} "
        f"effective={effective_model_load_type} "
        f"reason={model_load_type_reason} "
        f"total_vram_gib={f'{total_vram_gib:.2f}' if total_vram_gib is not None else 'unknown'}",
    )

    model_kwargs = {
        "device_map": spec.device_map,
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": True if base_reference.resolved_local_path else spec.local_files_only,
        "low_cpu_mem_usage": True,
        "cache_dir": spec.cache_dir,
        "dtype": _resolve_dtype(torch, spec.torch_dtype),
    }

    if spec.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=spec.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_resolve_dtype(torch, spec.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=bool(spec.use_double_quant),
            llm_int8_enable_fp32_cpu_offload=(effective_model_load_type == "GPU_CPU_load"),
        )
        model_kwargs["quantization_config"] = quantization_config

    cpu_offload_retry_used = False
    cpu_offload_folder = None
    cpu_offload_max_memory = None
    if effective_model_load_type == "GPU_CPU_load":
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_state_dict"] = True
        model_kwargs["offload_buffers"] = True
        cpu_offload_max_memory = _build_low_vram_max_memory(torch)
        if cpu_offload_max_memory:
            model_kwargs["max_memory"] = cpu_offload_max_memory
        offload_folder = Path(_build_staging_dir(spec.cache_dir, "offload", spec.base_model_name_or_path))
        offload_folder.mkdir(parents=True, exist_ok=True)
        cpu_offload_folder = str(offload_folder)
        model_kwargs["offload_folder"] = cpu_offload_folder

    base_download_started = False
    base_download_finished = False
    base_download_failed = False
    quantization_fallback_used = False

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
        base_model_obj = _load_base_model_with_compat(
            AutoModelForCausalLM,
            base_reference.resolved_reference,
            model_kwargs,
            logger=logger,
        )
    except Exception as exc:
        if effective_model_load_type != "GPU_CPU_load" and _should_retry_with_cpu_offload(spec, exc):
            cpu_offload_retry_used = True
            retry_model_kwargs, cpu_offload_folder, cpu_offload_max_memory = _build_cpu_offload_retry_kwargs(
                spec,
                model_kwargs,
                BitsAndBytesConfig,
                torch,
            )
            _safe_log(
                logger,
                "low_vram_retry "
                "reason=quantized_model_spilled_to_cpu_or_disk "
                "retry_with_cpu_offload=True "
                f"offload_folder={cpu_offload_folder} "
                f"max_memory={cpu_offload_max_memory}",
            )
            try:
                base_model_obj = _load_base_model_with_compat(
                    AutoModelForCausalLM,
                    base_reference.resolved_reference,
                    retry_model_kwargs,
                    logger=logger,
                )
            except Exception as retry_exc:
                exc = retry_exc
            else:
                exc = None

        if exc is not None and _should_retry_without_4bit(spec, exc):
            quantization_fallback_used = True
            _safe_log(
                logger,
                "quantization_fallback reason=bitsandbytes_4bit_unavailable retry_without_4bit=True",
            )
            fallback_model_kwargs = dict(model_kwargs)
            fallback_model_kwargs.pop("quantization_config", None)
            try:
                base_model_obj = _load_base_model_with_compat(
                    AutoModelForCausalLM,
                    base_reference.resolved_reference,
                    fallback_model_kwargs,
                    logger=logger,
                )
            except Exception as fallback_exc:
                exc = fallback_exc
            else:
                exc = None

        if exc is None:
            pass
        elif base_reference.source == "huggingface":
            base_download_failed = True
            _safe_log(
                logger,
                f"base_download_failed reference={base_reference.original_reference} error={exc}",
            )
            formatted_error = _format_hf_error(exc, "base_model", spec.base_model_name_or_path)
            if _is_low_vram_dispatch_error(exc):
                formatted_error += (
                    " The selected LLM likely exceeds the available VRAM for a clean GPU-only load on this machine. "
                    "On GPUs around 8 GB VRAM, qwen3.5-4b or qwen2.5-7b-instruct is recommended if CPU offload still fails."
                )
            raise ModelLoadError(
                formatted_error
            ) from exc
        else:
            raise ModelLoadError(f"Failed to load base model '{spec.base_model_name_or_path}': {exc}") from exc

    base_model_class = type(base_model_obj).__name__
    model = base_model_obj
    adapter_config_loaded = False
    adapter_weights_loaded = False
    merge_lora_for_inference_requested = bool(spec.merge_lora_for_inference)
    merge_lora_for_inference_applied = False
    merge_lora_seconds = 0.0
    merge_lora_skipped_reason = "not_requested"

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
        if merge_lora_for_inference_requested:
            if not hasattr(model, "merge_and_unload"):
                merge_lora_skipped_reason = "merge_and_unload_unavailable"
            elif _model_is_loaded_in_4bit(model):
                merge_lora_skipped_reason = "quantized_model_loaded"
            else:
                merge_started_at = time.perf_counter()
                try:
                    model = model.merge_and_unload()
                except Exception as exc:
                    merge_lora_skipped_reason = "merge_failed"
                    _safe_log(
                        logger,
                        f"merge_lora_for_inference_failed error={exc}",
                    )
                else:
                    merge_lora_for_inference_applied = True
                    merge_lora_seconds = time.perf_counter() - merge_started_at
                    merge_lora_skipped_reason = None
                    _safe_log(
                        logger,
                        "merge_lora_for_inference_applied "
                        f"seconds={merge_lora_seconds:.3f}",
                    )
    elif merge_lora_for_inference_requested:
        merge_lora_skipped_reason = "no_adapter"

    final_model_class = type(model).__name__
    model.eval()

    model_state = _inspect_model_state(model, PeftModel)
    if merge_lora_for_inference_applied:
        model_state["merged_state"] = True
        model_state["active_adapter"] = "merged"
    adapter_weights_loaded = bool(
        adapter_reference
        and adapter_reference.source != "none"
        and adapter_report.get("adapter_model.safetensors", {}).get("readable")
        and (model_state["is_peft_model"] or model_state["merged_state"])
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
        "model_load_seconds": time.perf_counter() - load_started_at,
        "merge_lora_for_inference_requested": merge_lora_for_inference_requested,
        "merge_lora_for_inference_applied": merge_lora_for_inference_applied,
        "merge_lora_seconds": merge_lora_seconds,
        "merge_lora_skipped_reason": merge_lora_skipped_reason,
        "quantization_fallback_used": quantization_fallback_used,
        "model_load_type_requested": requested_model_load_type,
        "model_load_type_effective": effective_model_load_type,
        "model_load_type_reason": model_load_type_reason,
        "model_load_total_vram_gib": (
            f"{total_vram_gib:.2f}" if total_vram_gib is not None else None
        ),
        "cpu_offload_retry_used": cpu_offload_retry_used,
        "cpu_offload_folder": cpu_offload_folder,
        "cpu_offload_max_memory": cpu_offload_max_memory,
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
