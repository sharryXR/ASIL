from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_audit_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "audit_gui_eval_matrix.py"
    spec = importlib.util.spec_from_file_location("audit_gui_eval_matrix", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_validate_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "validate_gui_eval_correctness.py"
    spec = importlib.util.spec_from_file_location("validate_gui_eval_correctness", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gui_eval_mode_groups_known_software():
    from asil.gui_eval import gui_eval_mode_by_software

    assert gui_eval_mode_by_software("gitea") == "api_live"
    assert gui_eval_mode_by_software("jupyterlab") == "live_shadow_required"
    assert gui_eval_mode_by_software("audacity") == "custom_sync_existing"
    assert gui_eval_mode_by_software("inkscape") == "persist_then_observe"


def test_gui_eval_mode_for_known_adapter_uses_software_policy_before_inherited_default():
    from asil.adapter import ASILAdapter
    from asil.gui_eval import gui_eval_mode_for_adapter

    class InkscapeLikeAdapter:
        app_name = "Inkscape"
        gui_eval_mode = ASILAdapter.gui_eval_mode

    assert gui_eval_mode_for_adapter(InkscapeLikeAdapter()) == "persist_then_observe"


def test_gui_eval_mode_for_unknown_adapter_honors_explicit_custom_policy():
    from asil.gui_eval import gui_eval_mode_for_adapter

    class CustomAdapter:
        app_name = "custom-app"

        def gui_eval_mode(self):
            return "live_shadow_required"

    assert gui_eval_mode_for_adapter(CustomAdapter()) == "live_shadow_required"


def test_gui_eval_mode_for_known_adapter_honors_explicit_override():
    from asil.gui_eval import gui_eval_mode_for_adapter

    class CustomInkscapeAdapter:
        app_name = "Inkscape"

        def gui_eval_mode(self):
            return "custom_sync_existing"

    assert gui_eval_mode_for_adapter(CustomInkscapeAdapter()) == "custom_sync_existing"


def test_sync_adapter_from_gui_supports_old_and_new_signatures():
    from asil.gui_eval import sync_adapter_from_gui

    calls: list[tuple[str, object | None]] = []

    class OldAdapter:
        def sync_from_gui(self):
            calls.append(("old", None))

    class NewAdapter:
        def sync_from_gui(self, session=None):
            calls.append(("new", session))

    sync_adapter_from_gui(OldAdapter(), session="session-a")
    sync_adapter_from_gui(NewAdapter(), session="session-b")

    assert calls == [("old", None), ("new", "session-b")]


def test_audit_gui_eval_matrix_writes_expected_grouping(tmp_path: Path):
    module = _load_audit_module()
    output_path = tmp_path / "audit.json"

    exit_code = module.main(
        [
            "--task-set",
            "test_full15.json",
            "--test-config-base-dir",
            str(Path(__file__).resolve().parent.parent / "evaluation_examples"),
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["software"]["jupyterlab"]["group"] == "live_shadow_required"
    assert payload["software"]["code_server"]["group"] == "live_shadow_required"
    assert payload["software"]["drawio"]["group"] == "live_shadow_required"
    assert payload["software"]["gitea"]["group"] == "api_live"
    assert payload["software"]["inkscape"]["group"] == "persist_then_observe"


def test_validate_gui_eval_correctness_main_writes_report(tmp_path: Path, monkeypatch):
    module = _load_validate_module()

    fake_payload = {
        "mode": "saved_state_gui_parity",
        "overall": {
            "total": 1,
            "pass": 1,
            "startup_failed": 0,
            "gui_visible_but_observe_missed": 0,
            "unexpected_evaluator_drift": 0,
        },
        "software_summary": {},
        "tasks": [],
    }

    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return fake_payload

    monkeypatch.setattr(module, "_run_saved_state_gui_parity", fake_run)
    output_dir = tmp_path / "correctness"

    exit_code = module.main(
        [
            "--mode",
            "saved_state_gui_parity",
            "--software-filter",
            "jupyterlab",
            "drawio",
            "--task-id-filter",
            "jupyterlab_03",
            "drawio_01",
            "--output",
            str(output_dir),
        ]
    )

    report = json.loads((output_dir / "gui_eval_correctness.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["overall"]["pass"] == 1
    assert calls == [
        {
            "task_set": "test_full15.json",
            "test_config_base_dir": Path(__file__).resolve().parent.parent / "evaluation_examples",
            "output_dir": output_dir,
            "software_filter": ("jupyterlab", "drawio"),
            "task_id_filter": ("jupyterlab_03", "drawio_01"),
        }
    ]
