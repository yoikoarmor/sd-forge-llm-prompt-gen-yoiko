import importlib
import subprocess
import sys
from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version


def _safe_log(logger, message):
    if logger:
        logger(message)


def _is_requirement_satisfied(requirement_text):
    requirement = Requirement(requirement_text)
    try:
        installed_version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        return False, requirement, None

    if requirement.specifier:
        try:
            parsed_version = Version(installed_version)
            return parsed_version in requirement.specifier, requirement, installed_version
        except InvalidVersion:
            return requirement.specifier.contains(installed_version, prereleases=True), requirement, installed_version

    return True, requirement, installed_version


def _detect_llama_cpp_cuda_extra_index(logger=None):
    try:
        import torch
    except Exception as exc:
        _safe_log(logger, f"dependency_cuda_probe torch_unavailable={exc}")
        return None

    try:
        if not torch.cuda.is_available():
            _safe_log(logger, "dependency_cuda_probe cuda_available=False")
            return None

        cuda_version = str(getattr(torch.version, "cuda", "") or "").strip()
        major_minor = cuda_version.split(".")[:2]
        if len(major_minor) < 2 or not all(part.isdigit() for part in major_minor):
            _safe_log(logger, f"dependency_cuda_probe cuda_version_unusable={cuda_version}")
            return None

        cuda_tag = f"cu{major_minor[0]}{major_minor[1]}"
        extra_index = f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}"
        _safe_log(logger, f"dependency_cuda_probe cuda_version={cuda_version} extra_index={extra_index}")
        return extra_index
    except Exception as exc:
        _safe_log(logger, f"dependency_cuda_probe failed={exc}")
        return None


def ensure_runtime_requirement(requirement_text, *, logger=None, extra_index_url=None, import_name=None):
    satisfied, requirement, installed_version = _is_requirement_satisfied(requirement_text)
    if satisfied:
        _safe_log(
            logger,
            f"dependency already present: {requirement.name}"
            + (f" ({installed_version})" if installed_version else ""),
        )
        return False

    if installed_version:
        _safe_log(
            logger,
            f"updating dependency at model load: {requirement.name} "
            f"from {installed_version} to satisfy {requirement_text}",
        )
    else:
        _safe_log(logger, f"installing dependency at model load: {requirement_text}")

    command = [sys.executable, "-m", "pip", "install", "--upgrade", requirement_text]
    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        _safe_log(logger, f"dependency install failed returncode={completed.returncode} output={output[-1200:]}")
        raise RuntimeError(f"Failed to install {requirement_text}: {output[-1200:]}")

    _safe_log(logger, f"dependency install finished: {requirement_text}")
    importlib.invalidate_caches()
    if import_name:
        importlib.import_module(import_name)
    return True


def ensure_llama_cpp_runtime_dependencies(logger=None):
    extra_index = _detect_llama_cpp_cuda_extra_index(logger=logger)
    return ensure_runtime_requirement(
        "llama-cpp-python>=0.3.16",
        logger=logger,
        extra_index_url=extra_index,
        import_name="llama_cpp",
    )
