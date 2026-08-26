from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, PngImagePlugin

from asil.eval.raw_validation import (
    SUPPORTED_RULES,
    evaluate_task_raw,
    read_raw_state,
    validate_raw_final_state,
)


def _task(software: str, checkpoints: list[dict]):
    return SimpleNamespace(
        id=f"{software}_test",
        software=software,
        evaluator={
            "selection": "best_score",
            "paths": [
                {
                    "path_id": "main",
                    "checkpoints": checkpoints,
                }
            ],
        },
    )


def _checkpoint(rule: dict, *, weight: float = 1.0, checkpoint_id: str = "cp"):
    return {"id": checkpoint_id, "weight": weight, "rule": rule}


def test_supported_rules_are_explicit_and_nonempty():
    assert "element_exists" in SUPPORTED_RULES
    assert "element_value" in SUPPORTED_RULES
    assert "any_element_matches" in SUPPORTED_RULES
    assert all(isinstance(rule, str) and rule for rule in SUPPORTED_RULES)


def test_audacity_reader_scores_raw_json_without_observation_builder(tmp_path: Path):
    source = tmp_path / "audacity_project.json"
    source.write_text(
        json.dumps(
            {
                "tracks": [
                    {"id": "track_music", "name": "Narration Bed", "kind": "audio"},
                    {"id": "track_room_tone", "name": "Room Tone", "kind": "audio"},
                ]
            }
        )
    )
    adapter = SimpleNamespace(source_path=source)
    task = _task(
        "audacity",
        [
            _checkpoint(
                {
                    "element_value": {
                        "id": "track:track_music",
                        "key": "name",
                        "expected": "Narration Bed",
                    }
                },
                weight=0.5,
                checkpoint_id="renamed",
            ),
            _checkpoint(
                {
                    "any_element_matches": {
                        "type": "track",
                        "id": "track:track_room_tone",
                        "value": {"name": "Room Tone"},
                    }
                },
                weight=0.5,
                checkpoint_id="added",
            ),
        ],
    )

    report = validate_raw_final_state(adapter, task, evaluator_score=1.0)

    assert report["complete"] is True
    assert report["score"] == 1.0
    assert report["agreement"] is True
    assert report["raw_evidence"]["kind"] == "json_file"
    assert len(report["raw_sha256"]) == 64


def test_xml_json_png_and_workspace_readers_expose_selected_task_elements(tmp_path: Path):
    drawio = tmp_path / "diagram.drawio.json"
    drawio.write_text(
        json.dumps(
            {
                "canvas": {},
                "shapes": [{"id": "end", "label": "End", "shape_kind": "ellipse"}],
                "connectors": [],
            }
        )
    )
    drawio_state = read_raw_state(SimpleNamespace(source_path=drawio), _task("drawio", []))
    assert any(element["id"] == "shape:end" for element in drawio_state["elements"])

    svg = tmp_path / "image.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect id="rect1" x="10" y="20" '
        'width="100" height="50" rx="10" ry="10"/></svg>'
    )
    svg_state = read_raw_state(SimpleNamespace(source_path=svg), _task("inkscape", []))
    assert svg_state["elements"][0]["value"]["rx"] == "10"

    kdenlive = tmp_path / "project.kdenlive"
    kdenlive.write_text(
        '<kdenliveProject><timeline><track id="audio_main" name="Dialog Stem"/></timeline></kdenliveProject>'
    )
    kd_state = read_raw_state(SimpleNamespace(source_path=kdenlive), _task("kdenlive", []))
    assert kd_state["elements"][0]["id"] == "track:audio_main"

    png = tmp_path / "gimp_document.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text(
        "asil_gimp_state",
        json.dumps(
            {
                "canvas": {"width": 800, "height": 600},
                "layers": [{"id": "coral_accent", "kind": "ellipse", "fill": "#ff7f50"}],
            }
        ),
    )
    Image.new("RGB", (2, 2), "white").save(png, pnginfo=metadata)
    gimp_state = read_raw_state(SimpleNamespace(source_path=png), _task("gimp", []))
    assert gimp_state["elements"][0]["value"]["fill"] == "#ff7f50"

    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "config/settings.json").write_text('{"autosave": true}')
    code_state = read_raw_state(SimpleNamespace(source_path=workspace), _task("code_server", []))
    settings = next(e for e in code_state["elements"] if e["id"] == "file:config/settings.json")
    assert '"autosave": true' in settings["value"]["content"]


def test_jupyter_and_odf_readers_score_direct_file_content(tmp_path: Path):
    workspace = tmp_path / "jupyter"
    (workspace / "notebooks").mkdir(parents=True)
    (workspace / "README.md").write_text("Share final notebook\n")
    (workspace / "notebooks/analysis.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Analysis"]},
                    {
                        "cell_type": "code",
                        "source": ["print(42)"],
                        "outputs": [{"output_type": "stream", "text": ["42\n"]}],
                    },
                ]
            }
        )
    )
    notebook_state = read_raw_state(SimpleNamespace(source_path=workspace), _task("jupyterlab", []))
    cell = next(e for e in notebook_state["elements"] if e["id"] == "cell:notebooks/analysis.ipynb:1")
    assert cell["value"]["output"].strip() == "42"

    ods = tmp_path / "sheet.ods"
    content = """<?xml version="1.0" encoding="UTF-8"?>
    <office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
      xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
      xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <office:body><office:spreadsheet><table:table table:name="Sheet1">
      <table:table-row><table:table-cell office:value-type="string"><text:p>Sales Report</text:p></table:table-cell></table:table-row>
      </table:table></office:spreadsheet></office:body></office:document-content>"""
    with zipfile.ZipFile(ods, "w") as archive:
        archive.writestr("content.xml", content)
    ods_task = _task(
        "libreoffice",
        [_checkpoint({"element_value": {"id": "Sheet1!A1", "key": None, "expected": "Sales Report"}})],
    )
    report = validate_raw_final_state(SimpleNamespace(source_path=ods), ods_task, evaluator_score=1.0)
    assert report["complete"] is True
    assert report["score"] == 1.0


def test_nautilus_reader_uses_raw_state_and_filesystem(tmp_path: Path):
    workspace = tmp_path / "files"
    (workspace / "Projects/beta").mkdir(parents=True)
    (workspace / "Projects/beta/beta-milestones.txt").write_text("done")
    state_path = tmp_path / "nautilus_state.json"
    state_path.write_text(json.dumps({"current_dir": "Projects/beta"}))
    adapter = SimpleNamespace(
        source_path=workspace,
        workspace_path=workspace,
        state_path=state_path,
    )
    task = _task(
        "nautilus",
        [
            _checkpoint(
                {
                    "any_element_matches": {
                        "type": "directory_entry",
                        "value": {"path": "Projects/beta/beta-milestones.txt"},
                    }
                },
                weight=0.5,
            ),
            _checkpoint(
                {
                    "no_element_matches": {
                        "type": "directory_entry",
                        "value": {"path": "Projects/beta/milestones.txt"},
                    }
                },
                weight=0.5,
            ),
        ],
    )
    report = validate_raw_final_state(adapter, task, evaluator_score=1.0)
    assert report["complete"] is True
    assert report["score"] == 1.0


def test_raw_evaluator_fails_closed_on_unsupported_rules_and_reports_mismatch():
    raw_state = {
        "elements": [],
        "app_view": "",
        "raw_evidence": {"kind": "synthetic"},
        "raw_sha256": "0" * 64,
    }
    unsupported_task = _task("demo", [_checkpoint({"pixel_magic": {"expected": 1}})])
    unsupported = evaluate_task_raw(unsupported_task, raw_state, evaluator_score=1.0)
    assert unsupported["complete"] is False
    assert "unsupported rule" in unsupported["errors"][0]

    supported_task = _task(
        "demo",
        [_checkpoint({"element_exists": "missing"})],
    )
    mismatch = evaluate_task_raw(supported_task, raw_state, evaluator_score=1.0)
    assert mismatch["complete"] is True
    assert mismatch["score"] == 0.0
    assert mismatch["agreement"] is False
