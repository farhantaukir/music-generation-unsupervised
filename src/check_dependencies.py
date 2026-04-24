"""Runtime preflight checks for required third-party dependencies."""

from __future__ import annotations

import importlib.util
import sys
from typing import Dict


REQUIRED_MODULES: Dict[str, str] = {
    "numpy": "numpy>=1.26",
    "torch": "torch>=2.2",
    "tqdm": "tqdm>=4.66",
    "pretty_midi": "pretty_midi>=0.2.10",
    "mido": "mido>=1.3.2",
    "matplotlib": "matplotlib>=3.8",
}


def find_missing_modules() -> dict[str, str]:
    """Return missing import names mapped to their requirements spec."""
    missing: dict[str, str] = {}
    for module_name, requirement in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module_name) is None:
            missing[module_name] = requirement
    return missing


def main() -> int:
    """Run dependency checks and print actionable install guidance."""
    missing = find_missing_modules()
    if not missing:
        print("All required runtime dependencies are installed.")
        return 0

    print("Missing required dependencies:")
    for module_name, requirement in missing.items():
        print(f"- {module_name} ({requirement})")

    print("\nInstall them with:")
    print("pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
