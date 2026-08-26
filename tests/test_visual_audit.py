import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _load_visual_audit_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_visual_deltas.py"
    spec = importlib.util.spec_from_file_location("audit_visual_deltas", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_preflight_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "preflight_visual_gate.py"
    spec = importlib.util.spec_from_file_location("preflight_visual_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_edge_black_crop_detector_flags_black_right_and_bottom_bars(tmp_path: Path):
    module = _load_visual_audit_module()
    image_path = tmp_path / "cropped.png"

    image = Image.new("RGB", (400, 300), "white")
    for x in range(320, 400):
        for y in range(300):
            image.putpixel((x, y), (0, 0, 0))
    for x in range(400):
        for y in range(240, 300):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    metrics = module._edge_black_metrics(image_path)

    assert metrics["cropped_or_letterboxed"] is True
    assert metrics["edge_black_ratio"] > 0.8
    assert metrics["interior_black_ratio"] < 0.1


def test_edge_black_crop_detector_ignores_full_black_player_windows(tmp_path: Path):
    module = _load_visual_audit_module()
    image_path = tmp_path / "player.png"

    Image.new("RGB", (400, 300), "black").save(image_path)

    metrics = module._edge_black_metrics(image_path)

    assert metrics["cropped_or_letterboxed"] is False
    assert metrics["interior_black_ratio"] > 0.9


def test_visual_audit_tracks_capture_complete_from_render_metadata(tmp_path: Path):
    module = _load_visual_audit_module()
    task_dir = tmp_path / "audacity" / "audacity_01"
    task_dir.mkdir(parents=True)
    Image.new("RGB", (400, 300), "white").save(task_dir / "step_0.png")
    Image.new("RGB", (400, 300), "gray").save(task_dir / "step_1.png")
    (task_dir / "step_0.render.json").write_text(
        json.dumps({"actual_page": True, "capture_complete": True}),
        encoding="utf-8",
    )
    (task_dir / "step_1.render.json").write_text(
        json.dumps({"actual_page": True, "capture_complete": False}),
        encoding="utf-8",
    )

    audit = module._audit_task(task_dir, "audacity")

    assert audit.actual_page_all_true is True
    assert audit.capture_complete_all_true is False


def test_preflight_gate_fails_when_capture_complete_is_missing(tmp_path: Path, capsys):
    module = _load_preflight_module()
    audit_path = tmp_path / "visual_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "per_software": {
                    "audacity": {
                        "total_tasks": 2,
                        "all_identical": 0,
                        "actual_page_all_true": 2,
                        "capture_complete_all_true": 1,
                        "cropped_or_letterboxed": 0,
                        "max_render_to_action_gap_s": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with patch.object(sys, "argv", ["preflight_visual_gate.py", str(audit_path)]):
        exit_code = module.main()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "capture_complete" in out


def test_visual_audit_marks_missing_render_artifacts_incomplete(tmp_path: Path):
    module = _load_visual_audit_module()
    task_dir = tmp_path / "drawio" / "drawio_01"
    task_dir.mkdir(parents=True)

    audit = module._audit_task(task_dir, "drawio")

    assert audit.screenshot_count == 0
    assert audit.actual_page_all_true is False
    assert audit.capture_complete_all_true is False
