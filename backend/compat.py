from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from packaging.version import Version, InvalidVersion


EXTENSION_ID = "sd-forge-llm-prompt-gen-yoiko"
LOG_PREFIX = f"[{EXTENSION_ID}]"
QWEN35_MIN_TRANSFORMERS = Version("5.5.0")


def _safe_log(logger, message: str):
    if logger:
        logger(message)


@lru_cache(maxsize=1)
def get_dependency_versions():
    names = [
        "transformers",
        "huggingface_hub",
        "peft",
        "accelerate",
        "bitsandbytes",
        "tokenizers",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parse_version(value: str | None):
    if not value:
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def transformers_supports_qwen35():
    installed = _parse_version(get_dependency_versions().get("transformers"))
    if installed is None:
        return False
    return installed >= QWEN35_MIN_TRANSFORMERS


def format_dependency_versions():
    versions = get_dependency_versions()
    parts = [f"{name}={versions.get(name) or '<missing>'}" for name in versions]
    return " ".join(parts)


def _patch_hf_hffolder_compat():
    try:
        import huggingface_hub
        if hasattr(huggingface_hub, "HfFolder"):
            return False

        class _CompatHfFolder:
            @staticmethod
            def save_token(token):
                try:
                    from huggingface_hub._login import _save_token

                    _save_token(token, "compat-token")
                except Exception:
                    pass

            @staticmethod
            def get_token():
                try:
                    return huggingface_hub.get_token()
                except Exception:
                    return None

            @staticmethod
            def delete_token():
                try:
                    huggingface_hub.logout()
                except Exception:
                    pass

        huggingface_hub.HfFolder = _CompatHfFolder
        if "__all__" in globals() and "HfFolder" not in huggingface_hub.__all__:
            huggingface_hub.__all__.append("HfFolder")
        return True
    except Exception:
        return False


def _patch_transformers_flux_compat():
    try:
        import transformers.utils as transformers_utils

        if hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
            return False
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        return True
    except Exception:
        return False


def _patch_transformers_no_init_weights_compat():
    try:
        import transformers.modeling_utils as modeling_utils
        if hasattr(modeling_utils, "no_init_weights"):
            return False

        from contextlib import contextmanager
        from transformers import initialization as transformers_init

        @contextmanager
        def _compat_no_init_weights(_enable=True):
            if not _enable:
                yield
                return

            with transformers_init.no_init_weights():
                yield

        modeling_utils.no_init_weights = _compat_no_init_weights
        return True
    except Exception:
        return False


def _patch_transformers_attention_mask_compat():
    try:
        import torch
        import transformers.modeling_attn_mask_utils as attn_utils

        if getattr(attn_utils, "_yoiko_causal_mask_patch_applied", False):
            return False

        def _compat_make_causal_mask(
            input_ids_shape,
            dtype,
            device,
            past_key_values_length=0,
            sliding_window=None,
        ):
            bsz, tgt_len = input_ids_shape
            working_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
            fill_value = float(torch.finfo(dtype).min)

            mask = torch.empty((tgt_len, tgt_len), dtype=working_dtype, device=device)
            mask.fill_(fill_value)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

            if past_key_values_length > 0:
                prefix = torch.zeros(
                    tgt_len,
                    past_key_values_length,
                    dtype=working_dtype,
                    device=device,
                )
                mask = torch.cat([prefix, mask], dim=-1)

            if sliding_window is not None:
                diagonal = past_key_values_length - sliding_window - 1
                context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
                mask.masked_fill_(context_mask, fill_value)

            if working_dtype != dtype:
                mask = mask.to(dtype)

            return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        attn_utils.AttentionMaskConverter._make_causal_mask = staticmethod(_compat_make_causal_mask)
        attn_utils._yoiko_causal_mask_patch_applied = True
        return True
    except Exception:
        return False


def apply_runtime_compat_shims(logger=None):
    applied = {
        "hf_hffolder": _patch_hf_hffolder_compat(),
        "flax_weights_name": _patch_transformers_flux_compat(),
        "no_init_weights": _patch_transformers_no_init_weights_compat(),
        "causal_mask": _patch_transformers_attention_mask_compat(),
    }
    _safe_log(
        logger,
        "runtime_compat "
        f"{format_dependency_versions()} "
        f"patched_hf_hffolder={applied['hf_hffolder']} "
        f"patched_flax_weights_name={applied['flax_weights_name']} "
        f"patched_no_init_weights={applied['no_init_weights']} "
        f"patched_causal_mask={applied['causal_mask']}",
    )
    return applied
