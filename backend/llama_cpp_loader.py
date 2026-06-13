import importlib.util
import os
import time
from pathlib import Path

from .loader import LoadedModelBundle, MissingDependencyError, ModelLoadError
from .runtime_dependencies import ensure_llama_cpp_runtime_dependencies


def _safe_log(logger, message):
    if logger:
        logger(message)


def _import_llama_cpp(logger=None):
    try:
        ensure_llama_cpp_runtime_dependencies(logger=logger)
    except Exception as exc:
        _safe_log(logger, f"llama_cpp_dependency_ensure_failed={exc}")

    try:
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.submodule_search_locations:
            package_dir = Path(next(iter(spec.submodule_search_locations)))
            dll_dir = package_dir / "lib"
            if dll_dir.exists():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise MissingDependencyError(
            "llama-cpp-python is required for GGUF models. "
            "Install the optional GGUF dependency first. "
            "CPU: pip install llama-cpp-python. "
            "CUDA wheels: pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/<cuda-version> "
            "(for example cu121, cu124, cu125, cu130)."
        ) from exc
    return Llama


def _resolve_gguf_path(spec, logger=None):
    if spec.gguf_path:
        gguf_path = Path(spec.gguf_path)
        if not gguf_path.exists():
            raise ModelLoadError(f"GGUF file not found: {gguf_path}")
        return str(gguf_path.resolve()), "local", False

    if spec.gguf_repo_id and spec.gguf_filename:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise MissingDependencyError(
                f"huggingface_hub is required to download GGUF from {spec.gguf_repo_id}: {exc}"
            ) from exc

        _safe_log(
            logger,
            f"gguf_download_started repo_id={spec.gguf_repo_id} filename={spec.gguf_filename} "
            f"cache_dir={spec.cache_dir} local_files_only={spec.local_files_only}",
        )
        try:
            path = hf_hub_download(
                repo_id=spec.gguf_repo_id,
                filename=spec.gguf_filename,
                cache_dir=spec.cache_dir,
                local_files_only=spec.local_files_only,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"GGUF download failed for {spec.gguf_repo_id}/{spec.gguf_filename}: {exc}"
            ) from exc
        _safe_log(logger, f"gguf_download_finished path={path}")
        return path, "huggingface", True

    raise ModelLoadError(
        f"Model key '{spec.key}' uses backend='llama_cpp' but no GGUF reference was configured."
    )


def _estimate_quantization_from_filename(path):
    name = Path(path).name.lower()
    known = [
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q3_k_l",
        "q4_0",
        "q4_1",
        "q4_k_s",
        "q4_k_m",
        "q5_0",
        "q5_1",
        "q5_k_s",
        "q5_k_m",
        "q6_k",
        "q8_0",
        "f16",
        "bf16",
    ]
    for item in known:
        if item in name:
            return item.upper()
    return None


def _extract_metadata_from_model(model):
    candidates = [
        getattr(model, "metadata", None),
        getattr(model, "model_metadata", None),
    ]
    inner_model = getattr(model, "_model", None)
    if inner_model is not None:
        candidates.extend(
            [
                getattr(inner_model, "metadata", None),
                getattr(inner_model, "model_metadata", None),
            ]
        )

    for candidate in candidates:
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:
            continue
        if isinstance(value, dict):
            return {str(key): str(val) for key, val in value.items()}
    return {}


def _get_metadata_value(metadata, *keys):
    for key in keys:
        if key in metadata:
            return metadata.get(key)
    lowered = {str(key).lower(): val for key, val in metadata.items()}
    for key in keys:
        value = lowered.get(str(key).lower())
        if value is not None:
            return value
    return None


def _resolve_thinking_suppression(spec, metadata):
    requested = str(getattr(spec, "thinking_suppression", "auto") or "auto").strip().lower()
    if requested == "no_think":
        return "no_think", "registry_override"
    if requested == "none":
        return "none", "registry_override"

    architecture = str(_get_metadata_value(metadata, "general.architecture") or "").lower()
    chat_template = str(
        _get_metadata_value(
            metadata,
            "tokenizer.chat_template",
            "tokenizer.ggml.chat_template",
            "chat_template",
        )
        or ""
    ).lower()

    is_qwen_architecture = "qwen" in architecture
    thinking_markers = ["enable_thinking", "/no_think", "<think>", "thinking"]
    has_thinking_marker = any(marker in chat_template for marker in thinking_markers)
    if is_qwen_architecture and has_thinking_marker:
        return "no_think", "auto_qwen_thinking_template"
    return "none", "auto_not_detected"


def load_llama_cpp_bundle(spec, logger=None):
    load_started_at = time.perf_counter()
    Llama = _import_llama_cpp(logger=logger)
    gguf_path, gguf_source, gguf_download_finished = _resolve_gguf_path(spec, logger=logger)

    llama_kwargs = {
        "model_path": gguf_path,
        "n_ctx": int(spec.n_ctx),
        "n_gpu_layers": int(spec.n_gpu_layers),
        "n_batch": int(spec.n_batch),
        "flash_attn": bool(spec.flash_attn),
        "verbose": False,
    }
    if spec.n_threads is not None:
        llama_kwargs["n_threads"] = int(spec.n_threads)

    try:
        model = Llama(**llama_kwargs)
    except TypeError as exc:
        if "flash_attn" not in str(exc):
            raise ModelLoadError(f"Failed to load GGUF model '{gguf_path}': {exc}") from exc
        llama_kwargs.pop("flash_attn", None)
        _safe_log(logger, "llama_cpp_load_retry removed_unsupported_flash_attn=True")
        try:
            model = Llama(**llama_kwargs)
        except Exception as retry_exc:
            raise ModelLoadError(f"Failed to load GGUF model '{gguf_path}': {retry_exc}") from retry_exc
    except Exception as exc:
        raise ModelLoadError(f"Failed to load GGUF model '{gguf_path}': {exc}") from exc

    metadata = _extract_metadata_from_model(model)
    architecture = _get_metadata_value(metadata, "general.architecture")
    chat_template = _get_metadata_value(
        metadata,
        "tokenizer.chat_template",
        "tokenizer.ggml.chat_template",
        "chat_template",
    )
    thinking_suppression_resolved, thinking_suppression_reason = _resolve_thinking_suppression(
        spec,
        metadata,
    )

    load_debug = {
        "backend": "llama_cpp",
        "gguf_source": gguf_source,
        "gguf_path": gguf_path,
        "gguf_repo_id": spec.gguf_repo_id,
        "gguf_filename": spec.gguf_filename,
        "gguf_download_finished": gguf_download_finished,
        "gguf_quantization": _estimate_quantization_from_filename(gguf_path),
        "gguf_metadata_architecture": architecture,
        "gguf_chat_template_present": bool(chat_template),
        "gguf_chat_template_thinking_markers": bool(
            chat_template
            and any(
                marker in str(chat_template).lower()
                for marker in ["enable_thinking", "/no_think", "<think>", "thinking"]
            )
        ),
        "n_ctx": int(spec.n_ctx),
        "n_gpu_layers": int(spec.n_gpu_layers),
        "n_batch": int(spec.n_batch),
        "n_threads": spec.n_threads,
        "flash_attn": bool(spec.flash_attn),
        "model_load_seconds": time.perf_counter() - load_started_at,
        "base_model_source": "gguf",
        "adapter_source": "none",
        "base_model_name_or_path": spec.base_model_name_or_path,
        "adapter_path": spec.adapter_path,
        "final_model_class": type(model).__name__,
        "tokenizer_class": None,
        "tokenizer_source": "gguf_chat_template",
        "chat_template_source": "gguf_metadata",
        "thinking_suppression": getattr(spec, "thinking_suppression", "auto"),
        "thinking_suppression_resolved": thinking_suppression_resolved,
        "thinking_suppression_reason": thinking_suppression_reason,
        "quantization_fallback_used": False,
    }

    _log_load_debug(load_debug, logger)
    return LoadedModelBundle(spec=spec, tokenizer=None, model=model, load_debug=load_debug)


def _log_load_debug(load_debug, logger):
    if not logger:
        return
    logger(f"backend={load_debug.get('backend')}")
    logger(f"gguf_source={load_debug.get('gguf_source')}")
    logger(f"gguf_path={load_debug.get('gguf_path')}")
    logger(f"gguf_repo_id={load_debug.get('gguf_repo_id')}")
    logger(f"gguf_filename={load_debug.get('gguf_filename')}")
    logger(f"gguf_download_finished={load_debug.get('gguf_download_finished')}")
    logger(f"gguf_quantization={load_debug.get('gguf_quantization')}")
    logger(f"gguf_metadata_architecture={load_debug.get('gguf_metadata_architecture')}")
    logger(f"gguf_chat_template_present={load_debug.get('gguf_chat_template_present')}")
    logger(f"gguf_chat_template_thinking_markers={load_debug.get('gguf_chat_template_thinking_markers')}")
    logger(f"n_ctx={load_debug.get('n_ctx')}")
    logger(f"n_gpu_layers={load_debug.get('n_gpu_layers')}")
    logger(f"n_batch={load_debug.get('n_batch')}")
    logger(f"n_threads={load_debug.get('n_threads')}")
    logger(f"flash_attn={load_debug.get('flash_attn')}")
    logger(f"model_load_seconds={load_debug['model_load_seconds']:.3f}")
    logger(f"final_model_class={load_debug.get('final_model_class')}")
    logger(f"thinking_suppression={load_debug.get('thinking_suppression')}")
    logger(f"thinking_suppression_resolved={load_debug.get('thinking_suppression_resolved')}")
    logger(f"thinking_suppression_reason={load_debug.get('thinking_suppression_reason')}")
