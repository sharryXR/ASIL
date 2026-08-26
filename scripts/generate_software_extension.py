#!/usr/bin/env python
"""Compatibility entry point for `python -m asil.softwaregen`."""

from asil.softwaregen.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
