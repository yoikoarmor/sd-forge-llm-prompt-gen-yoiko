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


main()
