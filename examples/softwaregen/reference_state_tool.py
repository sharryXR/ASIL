#!/usr/bin/env python3
"""Reviewed structured-output bridge used by softwaregen reference examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("State must be a JSON object containing an items list.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("observe")
    update = subparsers.add_parser("set-value")
    update.add_argument("item_id")
    update.add_argument("value")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = _load(args.state)
    if args.command == "set-value":
        matches = [item for item in payload["items"] if isinstance(item, dict) and item.get("id") == args.item_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one item with id {args.item_id!r}.")
        matches[0]["value"] = args.value
        args.state.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
