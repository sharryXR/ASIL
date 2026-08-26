"""Tests for the real-GUI preflight helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "validate_gui_preflight.py"
    spec = importlib.util.spec_from_file_location("asil_validate_gui_preflight", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_task_items_default_selects_first_task_per_software():
    module = _load_module()
    index_path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "test_multi_apps_80.json"

    selected = module._task_items(index_path, ("multi_apps",))

    assert [(software, task.id) for software, task in selected] == [("multi_apps", "multi_apps_001")]


def test_task_items_all_tasks_selects_full_multi_apps_index():
    module = _load_module()
    index_path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "test_multi_apps_80.json"

    selected = module._task_items(index_path, ("multi_apps",), all_tasks=True)

    assert len(selected) == 80
    assert selected[0][0] == "multi_apps"
    assert selected[0][1].id == "multi_apps_001"
    assert selected[-1][0] == "multi_apps"
    assert selected[-1][1].id == "multi_apps_080"


def test_task_items_filters_task_ids():
    module = _load_module()
    index_path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "test_multi_apps_80.json"

    selected = module._task_items(
        index_path,
        ("multi_apps",),
        all_tasks=True,
        task_id_filter=("multi_apps_025", "multi_apps_027"),
    )

    assert [(software, task.id) for software, task in selected] == [
        ("multi_apps", "multi_apps_025"),
        ("multi_apps", "multi_apps_027"),
    ]


def test_all_tasks_artifact_dirs_include_task_id_to_avoid_overwrite(tmp_path: Path):
    module = _load_module()
    index_path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "test_multi_apps_80.json"
    selected = module._task_items(index_path, ("multi_apps",), all_tasks=True)

    first_dir = module._artifact_dir(tmp_path, selected[0][0], selected[0][1], all_tasks=True)
    second_dir = module._artifact_dir(tmp_path, selected[1][0], selected[1][1], all_tasks=True)
    default_dir = module._artifact_dir(tmp_path, selected[0][0], selected[0][1])

    assert first_dir == tmp_path / "multi_apps" / "multi_apps_001"
    assert second_dir == tmp_path / "multi_apps" / "multi_apps_002"
    assert first_dir != second_dir
    assert default_dir == tmp_path / "multi_apps"


def test_load_resume_report_requires_passed_report_and_screenshot(tmp_path: Path):
    module = _load_module()
    software_dir = tmp_path / "multi_apps" / "multi_apps_001"
    software_dir.mkdir(parents=True)
    report = {
        "session_ok": True,
        "capture_ok": True,
        "action_ok": True,
        "persist_ok": True,
        "observe_ok": True,
        "actual_page": True,
        "capture_complete": True,
        "error": "",
    }
    (software_dir / "preflight_report.json").write_text(
        module.json.dumps(report) + "\n",
        encoding="utf-8",
    )

    assert module._load_resume_report(software_dir) is None

    (software_dir / "step_0.png").write_bytes(b"png")

    assert module._load_resume_report(software_dir) == report

    report["observe_ok"] = False
    (software_dir / "preflight_report.json").write_text(
        module.json.dumps(report) + "\n",
        encoding="utf-8",
    )

    assert module._load_resume_report(software_dir) is None


def test_report_passed_requires_actual_page_and_complete_capture():
    module = _load_module()
    report = {
        "session_ok": True,
        "capture_ok": True,
        "action_ok": True,
        "persist_ok": True,
        "observe_ok": True,
        "actual_page": True,
        "capture_complete": True,
    }

    assert module._report_passed(report) is True

    report["capture_complete"] = False
    assert module._report_passed(report) is False

    report["capture_complete"] = True
    report["actual_page"] = False
    assert module._report_passed(report) is False
