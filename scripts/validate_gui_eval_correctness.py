#!/usr/bin/env python3
"""Validate GUI evaluation correctness against the shared evaluator."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from asil.benchmark import PROJECT_ROOT, _create_adapter
from asil.eval.evaluator import evaluate_task_result
from asil.eval.runner import TaskDefinition, resolve_placeholders
from asil.gui_eval import gui_eval_mode_by_software, sync_adapter_from_gui
from asil.gui_agent.session import resolve_gui_session_spec, start_gui_session
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
class GUICorrectnessRecord:
    task_id: str
    software: str
    mode: str
    status: str
    score: float
    success: bool
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate GUI evaluation correctness for benchmark tasks.")
    parser.add_argument(
        "--mode",
        choices=["saved_state_gui_parity"],
        default="saved_state_gui_parity",
    )
    parser.add_argument("--task-set", default="test_full15.json")
    parser.add_argument(
        "--test-config-base-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_examples",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software-filter", nargs="+", default=None)
    parser.add_argument("--task-id-filter", nargs="+", default=None)
    return parser


def _software_tasks(
    index_path: Path,
    software_filter: tuple[str, ...] | None,
    task_id_filter: tuple[str, ...] | None,
) -> dict[str, list[TaskDefinition]]:
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    software_tasks: dict[str, list[TaskDefinition]] = {}
    allowed_task_ids = set(task_id_filter or ())
    for software in FULL15_SOFTWARE:
        if software_filter and software not in software_filter:
            continue
        if not index.get(software):
            continue
        tasks = TaskDefinition.from_index(index_path, domain=software)
        if allowed_task_ids:
            tasks = [task for task in tasks if task.id in allowed_task_ids]
            if not tasks:
                continue
        software_tasks[software] = tasks
    return software_tasks


def _use_mock(software: str) -> bool:
    return software == "obs"


def _capture_step_zero(session, spec, task_dir: Path) -> None:
    image_path = task_dir / "step_0.png"
    capture_complete = bool(session.capture(image_path))
    render_meta = {
        "filename": image_path.name,
        "kind": "app_window" if spec.surface_type == "desktop" else "web_ui_screenshot",
        "backend": "x11-window-capture",
        "actual_page": True,
        "capture_complete": capture_complete,
    }
    (task_dir / "step_0.render.json").write_text(
        json.dumps(render_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_saved_state_gui_parity(
    *,
    task_set: str,
    test_config_base_dir: Path,
    output_dir: Path,
    software_filter: tuple[str, ...] | None,
    task_id_filter: tuple[str, ...] | None,
) -> dict[str, Any]:
    index_path = test_config_base_dir / task_set
    records: list[GUICorrectnessRecord] = []
    software_summary: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="asil_gui_correctness_") as tmpdir:
        tmp_root = Path(tmpdir)
        for software, tasks in _software_tasks(index_path, software_filter, task_id_filter).items():
            counts: Counter[str] = Counter()
            software_out = output_dir / software
            software_out.mkdir(parents=True, exist_ok=True)
            for task in tasks:
                task_dir = software_out / task.id
                task_dir.mkdir(parents=True, exist_ok=True)
                try:
                    adapter_root = tmp_root / software / task.id
                    adapter_root.mkdir(parents=True, exist_ok=True)
                    adapter = _create_adapter(software, adapter_root, mock=_use_mock(software))
                    if hasattr(adapter, "clear_gui_shadow_state"):
                        adapter.clear_gui_shadow_state()
                    if hasattr(adapter, "reset_state"):
                        adapter.reset_state()
                    if hasattr(adapter, "setup_state"):
                        adapter.setup_state(getattr(task, "initial_state", "default"))
                    context = adapter.get_context()
                    for action_data in [resolve_placeholders(action, context) for action in task.actions]:
                        adapter.execute(Action(**action_data))

                    spec = resolve_gui_session_spec(adapter)
                    with start_gui_session(spec) as session:
                        _capture_step_zero(session, spec, task_dir)
                        sync_adapter_from_gui(adapter, session)
                        obs = adapter.observe()
                        evaluation = evaluate_task_result(task, obs)
                    status = "pass" if evaluation.success else "unexpected_evaluator_drift"
                    error = ""
                    score = evaluation.score
                    success = evaluation.success
                except Exception as exc:  # pragma: no cover - runtime dependent
                    status = "startup_failed"
                    error = f"{type(exc).__name__}: {exc}"
                    score = 0.0
                    success = False

                record = GUICorrectnessRecord(
                    task_id=task.id,
                    software=software,
                    mode=gui_eval_mode_by_software(software),
                    status=status,
                    score=score,
                    success=success,
                    error=error,
                )
                counts[status] += 1
                records.append(record)
                (task_dir / "result.json").write_text(
                    json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            software_summary[software] = {
                "group": gui_eval_mode_by_software(software),
                "total": sum(counts.values()),
                "pass": counts["pass"],
                "startup_failed": counts["startup_failed"],
                "gui_visible_but_observe_missed": counts["gui_visible_but_observe_missed"],
                "unexpected_evaluator_drift": counts["unexpected_evaluator_drift"],
            }

    overall = Counter(record.status for record in records)
    return {
        "mode": "saved_state_gui_parity",
        "task_set": task_set,
        "overall": {
            "total": len(records),
            "pass": overall["pass"],
            "startup_failed": overall["startup_failed"],
            "gui_visible_but_observe_missed": overall["gui_visible_but_observe_missed"],
            "unexpected_evaluator_drift": overall["unexpected_evaluator_drift"],
        },
        "software_summary": software_summary,
        "tasks": [asdict(record) for record in records],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _run_saved_state_gui_parity(
        task_set=args.task_set,
        test_config_base_dir=args.test_config_base_dir,
        output_dir=output_dir,
        software_filter=tuple(args.software_filter) if args.software_filter else None,
        task_id_filter=tuple(args.task_id_filter) if args.task_id_filter else None,
    )

    report_path = output_dir / "gui_eval_correctness.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GUI correctness report saved to {report_path}")
    overall = payload["overall"]
    print(
        f"overall: pass={overall['pass']}/{overall['total']}, "
        f"startup_failed={overall['startup_failed']}, "
        f"drift={overall['unexpected_evaluator_drift']}"
    )
    return 0 if overall["startup_failed"] == 0 and overall["unexpected_evaluator_drift"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
