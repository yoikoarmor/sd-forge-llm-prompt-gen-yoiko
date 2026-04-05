import sys
import site
from pathlib import Path
from importlib import metadata
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion


FORGE_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != EXTENSION_ROOT]
if str(FORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FORGE_ROOT))

import launch


EXTENSION_ID = "sd-forge-llm-prompt-gen-yoiko"
LOG_PREFIX = f"[{EXTENSION_ID}]"
HF_COMPAT_MARKER_START = "# BEGIN sd-forge-llm-prompt-gen-yoiko hf_hub compat shim"
HF_COMPAT_MARKER_END = "# END sd-forge-llm-prompt-gen-yoiko hf_hub compat shim"
HF_INIT_MARKER_START = "# BEGIN sd-forge-llm-prompt-gen-yoiko hf_hub package compat"
HF_INIT_MARKER_END = "# END sd-forge-llm-prompt-gen-yoiko hf_hub package compat"
TF_UTILS_MARKER_START = "# BEGIN sd-forge-llm-prompt-gen-yoiko transformers utils compat"
TF_UTILS_MARKER_END = "# END sd-forge-llm-prompt-gen-yoiko transformers utils compat"
TF_MODELING_MARKER_START = "# BEGIN sd-forge-llm-prompt-gen-yoiko transformers modeling compat"
TF_MODELING_MARKER_END = "# END sd-forge-llm-prompt-gen-yoiko transformers modeling compat"
TF_MASK_MARKER_START = "# BEGIN sd-forge-llm-prompt-gen-yoiko transformers mask compat"
TF_MASK_MARKER_END = "# END sd-forge-llm-prompt-gen-yoiko transformers mask compat"


def iter_requirements(requirements_path: Path):
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        yield requirement


def parse_requirement(requirement: str):
    return Requirement(requirement)


def is_requirement_satisfied(requirement: Requirement):
    try:
        installed_version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        return False, None

    if requirement.specifier:
        try:
            parsed_version = Version(installed_version)
            return parsed_version in requirement.specifier, installed_version
        except InvalidVersion:
            return requirement.specifier.contains(installed_version, prereleases=True), installed_version

    return True, installed_version


def get_sitecustomize_path():
    for candidate in site.getsitepackages():
        path = Path(candidate) / "sitecustomize.py"
        if path.exists():
            return path

    fallback_root = Path(site.getsitepackages()[-1])
    fallback_root.mkdir(parents=True, exist_ok=True)
    return fallback_root / "sitecustomize.py"


def get_site_package_path(*parts: str):
    for candidate in site.getsitepackages():
        base = Path(candidate)
        if not base.exists():
            continue

        target = base.joinpath(*parts)
        if target.exists():
            return target

        if "site-packages" in str(base).lower():
            return target

    fallback_root = Path(site.getsitepackages()[-1])
    fallback_root.mkdir(parents=True, exist_ok=True)
    return fallback_root.joinpath(*parts)


def ensure_file_contains_patch(path: Path, marker_start: str, marker_end: str, patch_text: str):
    if path.exists():
        existing_text = path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(path)

    if marker_start in existing_text:
        print(f"{LOG_PREFIX} patch already present in {path}")
        return

    patched_text = existing_text.rstrip() + "\n\n" + patch_text.rstrip() + "\n"
    path.write_text(patched_text, encoding="utf-8")
    print(f"{LOG_PREFIX} installed package compatibility patch at {path}")


def ensure_package_compat_shims():
    hf_init_path = get_site_package_path("huggingface_hub", "__init__.py")
    tf_utils_path = get_site_package_path("transformers", "utils", "__init__.py")
    tf_modeling_path = get_site_package_path("transformers", "modeling_utils.py")
    tf_mask_path = get_site_package_path("transformers", "modeling_attn_mask_utils.py")

    hf_patch = f"""
{HF_INIT_MARKER_START}
if "HfFolder" not in globals():
    class HfFolder:
        @staticmethod
        def save_token(token):
            try:
                from ._login import _save_token
                _save_token(token, "compat-token")
            except Exception:
                pass

        @staticmethod
        def get_token():
            try:
                return get_token()
            except Exception:
                return None

        @staticmethod
        def delete_token():
            try:
                logout()
            except Exception:
                pass

    if "__all__" in globals() and "HfFolder" not in __all__:
        __all__.append("HfFolder")
{HF_INIT_MARKER_END}
"""

    tf_utils_patch = f"""
{TF_UTILS_MARKER_START}
if "FLAX_WEIGHTS_NAME" not in globals():
    FLAX_WEIGHTS_NAME = "flax_model.msgpack"
{TF_UTILS_MARKER_END}
"""

    tf_modeling_patch = f"""
{TF_MODELING_MARKER_START}
if "no_init_weights" not in globals():
    from contextlib import contextmanager as _yoiko_contextmanager
    from transformers import initialization as _yoiko_transformers_init

    @_yoiko_contextmanager
    def no_init_weights(_enable: bool = True):
        if not _enable:
            yield
            return
        with _yoiko_transformers_init.no_init_weights():
            yield
{TF_MODELING_MARKER_END}
"""

    tf_mask_patch = f"""
{TF_MASK_MARKER_START}
try:
    if not globals().get("_yoiko_causal_mask_patch_applied", False):
        def _yoiko_make_causal_mask(
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

        AttentionMaskConverter._make_causal_mask = staticmethod(_yoiko_make_causal_mask)
        _yoiko_causal_mask_patch_applied = True
except Exception as e:
    print("[transformers mask compat error]:", e)
{TF_MASK_MARKER_END}
"""

    ensure_file_contains_patch(hf_init_path, HF_INIT_MARKER_START, HF_INIT_MARKER_END, hf_patch)
    ensure_file_contains_patch(tf_utils_path, TF_UTILS_MARKER_START, TF_UTILS_MARKER_END, tf_utils_patch)
    ensure_file_contains_patch(tf_modeling_path, TF_MODELING_MARKER_START, TF_MODELING_MARKER_END, tf_modeling_patch)
    ensure_file_contains_patch(tf_mask_path, TF_MASK_MARKER_START, TF_MASK_MARKER_END, tf_mask_patch)


def ensure_startup_compat_shims():
    sitecustomize_path = get_sitecustomize_path()
    if sitecustomize_path.exists():
        existing_text = sitecustomize_path.read_text(encoding="utf-8")
    else:
        existing_text = ""

    if HF_COMPAT_MARKER_START in existing_text:
        print(f"{LOG_PREFIX} huggingface_hub compatibility shim already present")
        return

    shim = f"""

{HF_COMPAT_MARKER_START}
def _patch_hf_hffolder_compat():
    try:
        import huggingface_hub
        if hasattr(huggingface_hub, "HfFolder"):
            return

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
    except Exception as e:
        print("[sitecustomize error]:", e)

_patch_hf_hffolder_compat()

def _patch_transformers_flux_compat():
    try:
        import transformers.utils as transformers_utils
        if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
            transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    except Exception as e:
        print("[sitecustomize error]:", e)

_patch_transformers_flux_compat()

def _patch_transformers_no_init_weights_compat():
    try:
        import transformers.modeling_utils as modeling_utils
        if hasattr(modeling_utils, "no_init_weights"):
            return

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
    except Exception as e:
        print("[sitecustomize error]:", e)

_patch_transformers_no_init_weights_compat()

def _patch_transformers_attention_mask_compat():
    try:
        import torch
        import transformers.modeling_attn_mask_utils as attn_utils
        if getattr(attn_utils, "_yoiko_causal_mask_patch_applied", False):
            return

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
    except Exception as e:
        print("[sitecustomize error]:", e)

_patch_transformers_attention_mask_compat()
{HF_COMPAT_MARKER_END}
""".rstrip() + "\n"

    sitecustomize_path.write_text(existing_text.rstrip() + shim, encoding="utf-8")
    print(f"{LOG_PREFIX} installed startup compatibility shims at {sitecustomize_path}")


def main():
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        print(f"{LOG_PREFIX} requirements.txt not found; skipping install step")
        return

    for requirement in iter_requirements(requirements_path):
        try:
            parsed_requirement = parse_requirement(requirement)
            satisfied, installed_version = is_requirement_satisfied(parsed_requirement)
            if satisfied:
                if installed_version:
                    print(
                        f"{LOG_PREFIX} dependency already present: "
                        f"{parsed_requirement.name} ({installed_version})"
                    )
                else:
                    print(f"{LOG_PREFIX} dependency already present: {parsed_requirement.name}")
                continue

            if installed_version:
                print(
                    f"{LOG_PREFIX} upgrading dependency: {parsed_requirement.name} "
                    f"from {installed_version} to satisfy {requirement}"
                )
            else:
                print(f"{LOG_PREFIX} installing missing dependency: {requirement}")
            launch.run_pip(
                f"install {requirement}",
                f"{EXTENSION_ID} dependency: {requirement}",
            )
        except Exception as exc:
            print(f"{LOG_PREFIX} failed to install {requirement}: {exc}")

    try:
        ensure_startup_compat_shims()
    except Exception as exc:
        print(f"{LOG_PREFIX} failed to install startup compatibility shims: {exc}")

    try:
        ensure_package_compat_shims()
    except Exception as exc:
        print(f"{LOG_PREFIX} failed to install package compatibility shims: {exc}")


main()
