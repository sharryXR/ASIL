#!/usr/bin/env python3
"""Audit full15 GUI evaluation source-of-truth grouping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from asil.benchmark import PROJECT_ROOT, _load_task_index_mapping
from asil.gui_eval import gui_eval_mode_by_software


GROUP_EXPLANATIONS = {
    "api_live": "observe() reads a live API/service source after GUI-side submission.",
    "persist_then_observe": "GUI-visible edits must be persisted to canonical files/projects before observe().",
    "live_shadow_required": "GUI-visible state may precede persistence; observe() needs GUI shadow state.",
    "custom_sync_existing": "Adapter already has custom sync_from_gui() logic and must be regression-verified.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GUI evaluation groups for a task set.")
    parser.add_argument("--task-set", default="test_full15.json")
    parser.add_argument(
        "--test-config-base-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_examples",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_audit_payload(task_set: str, test_config_base_dir: Path) -> dict[str, Any]:
    mapping = _load_task_index_mapping(
        task_index=task_set,
        test_config_base_dir=test_config_base_dir,
        software_filter=(),
        task_id_filter=(),
    )
    software_payload = {}
    group_counts: dict[str, int] = {}
    for software, task_ids in mapping.items():
        group = gui_eval_mode_by_software(software)
        group_counts[group] = group_counts.get(group, 0) + 1
        software_payload[software] = {
            "group": group,
            "task_count": len(task_ids),
            "explanation": GROUP_EXPLANATIONS[group],
        }
    return {
        "task_set": task_set,
        "software": software_payload,
        "group_counts": group_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_audit_payload(args.task_set, args.test_config_base_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GUI audit matrix saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
