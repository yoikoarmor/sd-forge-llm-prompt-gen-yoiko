import sys
from pathlib import Path


FORGE_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != EXTENSION_ROOT]
if str(FORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FORGE_ROOT))

import launch


EXTENSION_ID = "sd-forge-llm-prompt-gen-yoiko"
LOG_PREFIX = f"[{EXTENSION_ID}]"


def iter_requirements(requirements_path: Path):
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        yield requirement


def main():
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        print(f"{LOG_PREFIX} requirements.txt not found; skipping install step")
        return

    for requirement in iter_requirements(requirements_path):
        package_name = requirement.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].strip()
        try:
            if launch.is_installed(package_name):
                print(f"{LOG_PREFIX} dependency already present: {package_name}")
                continue

            print(f"{LOG_PREFIX} installing missing dependency: {requirement}")
            launch.run_pip(
                f"install {requirement}",
                f"{EXTENSION_ID} dependency: {requirement}",
            )
        except Exception as exc:
            print(f"{LOG_PREFIX} failed to install {requirement}: {exc}")


main()
