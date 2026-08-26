#!/usr/bin/env python3
"""Managed Singularity orchestrator for ASIL benchmark evaluation.

The implementation lives in :mod:`asil.singularity.managed` and reuses the same
worker stack for isolated parallel evaluation. Compatibility names are
re-exported for the script-level tests.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asil.singularity import managed as _impl  # noqa: E402


for _name in dir(_impl):
    if _name in {"__builtins__", "__cached__", "__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}:
        continue
    globals()[_name] = getattr(_impl, _name)


def main(argv: list[str] | None = None) -> int:
    """Run the shared implementation while honoring script-level monkeypatches."""

    patchable = [
        name
        for name in globals()
        if (name.startswith("_") or name in {"WorkerStack", "WorkerRuntime", "PortPlan"})
        and hasattr(_impl, name)
    ]
    originals = {name: getattr(_impl, name) for name in patchable}
    try:
        for name in patchable:
            setattr(_impl, name, globals()[name])
        return _impl.main(argv)
    finally:
        for name, value in originals.items():
            setattr(_impl, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
