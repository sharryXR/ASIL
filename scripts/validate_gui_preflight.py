#!/usr/bin/env python3
"""Run a lightweight real-GUI preflight for the 15-software benchmark set."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any

from asil.benchmark import PROJECT_ROOT, _create_adapter
from asil.eval.runner import TaskDefinition
from asil.gui_eval import sync_adapter_from_gui
from asil.gui_agent.controller import X11GUIController
from asil.gui_agent.parser import GUIAction
from asil.gui_agent.session import (
    _cleanup_gui_processes,
    create_startup_diagnostics,
    resolve_gui_session_spec,
    start_gui_session,
)
from asil.rendering import stop_virtual_display


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
SOFTWARE_CHOICES = (*FULL15_SOFTWARE, "multi_apps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate real GUI preflight for benchmark software.")
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
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-software preflight artifacts are written.",
    )
    parser.add_argument(
        "--software",
        nargs="+",
        choices=SOFTWARE_CHOICES,
        default=None,
        help="Optional subset of software to preflight (default: all benchmark software).",
    )
    parser.add_argument(
        "--task-id-filter",
        nargs="+",
        default=None,
        help="Optional task id subset to preflight.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Preflight every task in the selected task set instead of only the first task per software.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose preflight report and step_0 screenshot already passed.",
    )
    parser.add_argument(
        "--task-timeout-s",
        type=float,
        default=600.0,
        help="Maximum seconds to spend on one preflight task before writing a timeout report.",
    )
    return parser


def _task_items(
    index_path: Path,
    software_filter: tuple[str, ...] | None = None,
    *,
    all_tasks: bool = False,
    task_id_filter: tuple[str, ...] | None = None,
) -> list[tuple[str, TaskDefinition]]:
    tasks: list[tuple[str, TaskDefinition]] = []
    selected = software_filter or FULL15_SOFTWARE
    allowed_task_ids = set(task_id_filter or ())
    for software in selected:
        loaded = TaskDefinition.from_index(index_path, domain=software)
        if allowed_task_ids:
            loaded = [task for task in loaded if task.id in allowed_task_ids]
        if all_tasks:
            tasks.extend((software, task) for task in loaded)
        elif loaded:
            tasks.append((software, loaded[0]))
    return tasks


def _artifact_dir(output_dir: Path, software: str, task: TaskDefinition, *, all_tasks: bool = False) -> Path:
    if all_tasks:
        return output_dir / software / task.id
    return output_dir / software


def _report_passed(report: dict[str, Any]) -> bool:
    return all(
        report.get(key)
        for key in (
            "session_ok",
            "capture_ok",
            "action_ok",
            "persist_ok",
            "observe_ok",
            "actual_page",
            "capture_complete",
        )
    )


def _load_resume_report(software_dir: Path) -> dict[str, Any] | None:
    report_path = software_dir / "preflight_report.json"
    image_path = software_dir / "step_0.png"
    if not report_path.exists() or not image_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(report, dict) and _report_passed(report):
        return report
    return None


def _new_report(software: str, task: TaskDefinition) -> dict[str, Any]:
    return {
        "software": software,
        "task_id": task.id,
        "session_ok": False,
        "capture_ok": False,
        "action_ok": False,
        "persist_ok": False,
        "observe_ok": False,
        "actual_page": False,
        "capture_complete": False,
        "error": "",
    }


def _write_report(software_dir: Path, report: dict[str, Any]) -> None:
    (software_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _cleanup_after_timeout(software: str, task: TaskDefinition, adapter_tmp: Path) -> None:
    try:
        adapter = _create_adapter(software, adapter_tmp, mock=_use_mock(software))
        if callable(getattr(adapter, "prepare_task", None)):
            adapter.prepare_task(task)
        spec = resolve_gui_session_spec(adapter)
        _cleanup_gui_processes(spec)
    except Exception:
        pass
    try:
        stop_virtual_display()
    except Exception:
        pass


def _run_preflight_task(
    *,
    software: str,
    task: TaskDefinition,
    software_dir: Path,
    adapter_tmp: Path,
) -> dict[str, Any]:
    controller = X11GUIController()
    adapter = _create_adapter(software, adapter_tmp, mock=_use_mock(software))
    if callable(getattr(adapter, "prepare_task", None)):
        adapter.prepare_task(task)
    elif hasattr(adapter, "clear_gui_shadow_state"):
        adapter.clear_gui_shadow_state()
    if not callable(getattr(adapter, "prepare_task", None)) and hasattr(adapter, "reset_state"):
        adapter.reset_state()
    if not callable(getattr(adapter, "prepare_task", None)) and hasattr(adapter, "setup_state"):
        adapter.setup_state(task.initial_state)

    report = _new_report(software, task)
    diagnostics: dict[str, Any] | None = None
    try:
        spec = resolve_gui_session_spec(adapter)
        diagnostics = create_startup_diagnostics(spec)
        with start_gui_session(spec, startup_diagnostics=diagnostics) as session:
            report["session_ok"] = True
            image_path = software_dir / "step_0.png"
            report["capture_complete"] = bool(session.capture(image_path))
            report["capture_ok"] = image_path.exists()
            capture_metadata = getattr(session, "last_capture_metadata", {}) or {}
            render_meta = {
                "filename": image_path.name,
                "kind": "app_window" if spec.surface_type in {"desktop", "multi_window"} else "web_ui_screenshot",
                "backend": "x11-window-capture",
                "actual_page": True,
                "capture_complete": report["capture_complete"],
                "capture_metadata": capture_metadata,
            }
            for key in (
                "display",
                "screen",
                "root_capture_size",
                "cropped_size",
                "window_id",
                "window_geometry",
                "fallback_reason",
                "fallback_used",
                "fallback_window_id",
                "fallback_app",
                "fallback_maximize_used",
                "active_window_too_small",
                "retry_capture_used",
                "window_visible",
                "window_visible_strict",
                "window_visible_tolerance_px",
                "crop_meets_expectation",
            ):
                if key in capture_metadata:
                    render_meta[key] = capture_metadata[key]
            (software_dir / "step_0.render.json").write_text(
                json.dumps(render_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            report["actual_page"] = True

            controller.set_capture_window_id(getattr(session, "last_capture_window_id", ""))
            controller.execute(GUIAction("MOVE_TO", {"x": 20, "y": 20}), spec=spec)
            report["action_ok"] = True

            if spec.persist_shortcuts:
                controller.persist(spec)
            report["persist_ok"] = True

            sync_adapter_from_gui(adapter, session)
            adapter.observe()
            report["observe_ok"] = True
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if diagnostics is not None:
            (software_dir / "startup_diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    _write_report(software_dir, report)
    return report


def _preflight_worker(
    queue: mp.Queue,
    software: str,
    task: TaskDefinition,
    software_dir: Path,
    adapter_tmp: Path,
) -> None:
    report = _run_preflight_task(
        software=software,
        task=task,
        software_dir=software_dir,
        adapter_tmp=adapter_tmp,
    )
    queue.put(report)


def _run_preflight_task_with_timeout(
    *,
    software: str,
    task: TaskDefinition,
    software_dir: Path,
    adapter_tmp: Path,
    timeout_s: float,
) -> dict[str, Any]:
    timeout_s = max(float(timeout_s), 0.0)
    if timeout_s <= 0:
        return _run_preflight_task(
            software=software,
            task=task,
            software_dir=software_dir,
            adapter_tmp=adapter_tmp,
        )

    queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_preflight_worker,
        args=(queue, software, task, software_dir, adapter_tmp),
        name=f"asil-gui-preflight-{software}-{task.id}",
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        _cleanup_after_timeout(software, task, adapter_tmp)
        report = _new_report(software, task)
        report["error"] = f"TimeoutError: preflight exceeded {timeout_s:.1f}s"
        _write_report(software_dir, report)
        return report

    try:
        return queue.get_nowait()
    except Exception:
        report_path = software_dir / "preflight_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(report, dict):
                    return report
            except Exception:
                pass
        report = _new_report(software, task)
        report["error"] = f"ProcessError: preflight worker exited with code {process.exitcode}"
        _write_report(software_dir, report)
        return report


def _use_mock(software: str) -> bool:
    return False


def main() -> int:
    args = build_parser().parse_args()
    index_path = args.test_config_base_dir / args.task_set
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    software_filter = tuple(args.software) if args.software else None
    task_id_filter = tuple(args.task_id_filter) if args.task_id_filter else None

    with tempfile.TemporaryDirectory(prefix="asil_gui_preflight_") as tmpdir:
        tmp_root = Path(tmpdir)
        for software, task in _task_items(
            index_path,
            software_filter,
            all_tasks=args.all_tasks,
            task_id_filter=task_id_filter,
        ):
            software_dir = _artifact_dir(output_dir, software, task, all_tasks=args.all_tasks)
            software_dir.mkdir(parents=True, exist_ok=True)
            summary_key = f"{software}/{task.id}" if args.all_tasks else software
            if args.resume:
                resume_report = _load_resume_report(software_dir)
                if resume_report is not None:
                    summary[summary_key] = resume_report
                    print(f"[preflight] skipping passed {summary_key}")
                    continue
            adapter_tmp = tmp_root / software / task.id
            adapter_tmp.mkdir(parents=True, exist_ok=True)
            report = _run_preflight_task_with_timeout(
                software=software,
                task=task,
                software_dir=software_dir,
                adapter_tmp=adapter_tmp,
                timeout_s=args.task_timeout_s,
            )
            summary[summary_key] = report

    summary_path = output_dir / "preflight_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok_count = sum(
        1
        for report in summary.values()
        if _report_passed(report)
    )
    print(f"GUI preflight summary saved to {summary_path}")
    print(f"full preflight pass: {ok_count}/{len(summary)}")
    return 0 if ok_count == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
