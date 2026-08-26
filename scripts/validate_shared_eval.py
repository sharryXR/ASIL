#!/usr/bin/env python3
"""Validate that task end-states still satisfy the shared evaluator.

This script intentionally uses the same task JSON, the same adapter.observe()
output, and the same evaluator logic used by ASIL/GUI runs. It replays the
canonical `_asil.actions` to reach the intended end-state and records whether
the evaluator still considers that end-state correct.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from asil.benchmark import PROJECT_ROOT, _create_adapter
from asil.eval.evaluator import evaluate_task_result
from asil.eval.runner import TaskDefinition, resolve_placeholders
from asil.protocol import Action


FULL15_SOFTWARE = (
    "inkscape",
    "libreoffice",
    "blender",
    "obs",
    "gitea",
    "gimp",
    "libreoffice_writer",
    "libreoffice_impress",
    "code_server",
    "thunderbird",
    "nautilus",
    "kdenlive",
    "audacity",
    "drawio",
    "jupyterlab",
)


@dataclass
class TaskValidationRecord:
    task_id: str
    software: str
    status: str
    score: float
    success: bool
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate shared evaluator equivalence for a task set.")
    parser.add_argument(
        "--task-set",
        default="test_full15.json",
        help="Task-set index file inside evaluation_examples (default: test_full15.json).",
    )
    parser.add_argument(
        "--test-config-base-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_examples",
        help="Directory containing task-set indexes and examples/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report output path.",
    )
    return parser


def _software_tasks(index_path: Path) -> dict[str, list[TaskDefinition]]:
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    software_tasks: dict[str, list[TaskDefinition]] = {}
    for software in FULL15_SOFTWARE:
        task_ids = index.get(software, [])
        if not task_ids:
            continue
        software_tasks[software] = TaskDefinition.from_index(index_path, domain=software)
    return software_tasks


def _use_mock(software: str) -> bool:
    if software == "obs":
        return True
    if software == "blender" and shutil.which("blender") is None:
        return True
    return False


def main() -> int:
    args = build_parser().parse_args()
    index_path = args.test_config_base_dir / args.task_set
    report_path = args.output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[TaskValidationRecord] = []
    software_summary: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="asil_shared_eval_") as tmpdir:
        tmp_root = Path(tmpdir)
        for software, tasks in _software_tasks(index_path).items():
            software_root = tmp_root / software
            software_root.mkdir(parents=True, exist_ok=True)

            counts: Counter[str] = Counter()
            for task in tasks:
                try:
                    task_root = software_root / task.id
                    task_root.mkdir(parents=True, exist_ok=True)
                    adapter = _create_adapter(
                        software,
                        task_root,
                        mock=_use_mock(software),
                    )
                    if hasattr(adapter, "clear_gui_shadow_state"):
                        adapter.clear_gui_shadow_state()
                    if hasattr(adapter, "reset_state"):
                        adapter.reset_state()
                    if hasattr(adapter, "setup_state"):
                        adapter.setup_state(task.initial_state)
                    context = adapter.get_context()
                    resolved_actions = [resolve_placeholders(action, context) for action in task.actions]
                    obs = adapter.observe()
                    for action_data in resolved_actions:
                        obs = adapter.execute(Action(**action_data))
                    evaluation = evaluate_task_result(task, obs)
                    status = "pass" if evaluation.success else "unexpected_evaluator_drift"
                    record = TaskValidationRecord(
                        task_id=task.id,
                        software=software,
                        status=status,
                        score=evaluation.score,
                        success=evaluation.success,
                    )
                except Exception as exc:  # pragma: no cover - defensive report path
                    record = TaskValidationRecord(
                        task_id=task.id,
                        software=software,
                        status="blocked_by_env",
                        score=0.0,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                all_records.append(record)
                counts[record.status] += 1

            software_summary[software] = {
                "total": sum(counts.values()),
                "pass": counts["pass"],
                "unexpected_evaluator_drift": counts["unexpected_evaluator_drift"],
                "blocked_by_env": counts["blocked_by_env"],
            }

    overall = Counter(record.status for record in all_records)
    payload = {
        "task_set": args.task_set,
        "overall": {
            "total": len(all_records),
            "pass": overall["pass"],
            "unexpected_evaluator_drift": overall["unexpected_evaluator_drift"],
            "blocked_by_env": overall["blocked_by_env"],
        },
        "software_summary": software_summary,
        "tasks": [asdict(record) for record in all_records],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Shared evaluator validation report saved to {report_path}")
    print(
        f"overall: pass={overall['pass']}/{len(all_records)}, "
        f"drift={overall['unexpected_evaluator_drift']}, blocked={overall['blocked_by_env']}"
    )
    return 0 if overall["unexpected_evaluator_drift"] == 0 and overall["blocked_by_env"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
