#!/usr/bin/env python3
"""Thin CLI for the public ASIL benchmark entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asil.benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
