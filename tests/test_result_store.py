from __future__ import annotations

import json
from pathlib import Path

from asil.result_store import (
    TaskKey,
    append_summary_entry,
    build_aggregate,
    flatten_task_mapping,
    group_task_keys,
    load_summary_entries,
    select_pending_tasks,
)


def test_flatten_and_group_task_mapping_round_trip():
    mapping = {
        "inkscape": ["inkscape_01", "inkscape_02"],
        "gitea": ["gitea_01"],
    }

    flattened = flatten_task_mapping(mapping)

    assert flattened == [
        TaskKey("inkscape", "inkscape_01"),
        TaskKey("inkscape", "inkscape_02"),
        TaskKey("gitea", "gitea_01"),
    ]
    assert group_task_keys(flattened) == mapping


def test_select_pending_tasks_skips_complete_and_cleans_incomplete(tmp_path: Path):
    root = tmp_path / "results"
    complete_dir = root / "inkscape" / "inkscape_01"
    complete_dir.mkdir(parents=True)
    (complete_dir / "result.txt").write_text("1.0", encoding="utf-8")

    incomplete_dir = root / "inkscape" / "inkscape_02"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "step_0.png").write_text("stale", encoding="utf-8")

    selection = select_pending_tasks(
        {"inkscape": ["inkscape_01", "inkscape_02"]},
        result_root=root,
        run_mode="single",
        participant="gui",
        comparison_participants=("asil", "cli", "gui"),
        resume=True,
        force_rerun=False,
        rerun_failed_only=False,
    )

    assert selection.total_tasks == 2
    assert selection.skipped_tasks == 1
    assert selection.cleaned_incomplete_tasks == 1
    assert selection.pending_tasks == (TaskKey("inkscape", "inkscape_02"),)
    assert not incomplete_dir.exists()


def test_select_pending_tasks_reruns_only_failed_tasks(tmp_path: Path):
    root = tmp_path / "results"
    passed_dir = root / "drawio" / "drawio_01"
    passed_dir.mkdir(parents=True)
    (passed_dir / "result.txt").write_text("1.0", encoding="utf-8")

    failed_dir = root / "drawio" / "drawio_02"
    failed_dir.mkdir(parents=True)
    (failed_dir / "result.txt").write_text("0.0", encoding="utf-8")

    selection = select_pending_tasks(
        {"drawio": ["drawio_01", "drawio_02"]},
        result_root=root,
        run_mode="single",
        participant="asil",
        comparison_participants=("asil", "cli", "gui"),
        resume=True,
        force_rerun=False,
        rerun_failed_only=True,
    )

    assert selection.skipped_tasks == 1
    assert selection.pending_tasks == (TaskKey("drawio", "drawio_02"),)


def test_select_pending_tasks_for_comparison_requires_all_participants(tmp_path: Path):
    root = tmp_path / "results"
    for participant in ("asil", "cli"):
        task_dir = root / participant / "gitea" / "gitea_01"
        task_dir.mkdir(parents=True)
        (task_dir / "result.txt").write_text("1.0", encoding="utf-8")

    partial_gui_dir = root / "gui" / "gitea" / "gitea_01"
    partial_gui_dir.mkdir(parents=True)
    (partial_gui_dir / "step_0.png").write_text("stale", encoding="utf-8")

    selection = select_pending_tasks(
        {"gitea": ["gitea_01"]},
        result_root=root,
        run_mode="comparison",
        participant="asil",
        comparison_participants=("asil", "cli", "gui"),
        resume=True,
        force_rerun=False,
        rerun_failed_only=False,
    )

    assert selection.pending_tasks == (TaskKey("gitea", "gitea_01"),)
    assert selection.cleaned_incomplete_tasks == 1
    assert not (root / "asil" / "gitea" / "gitea_01").exists()
    assert not (root / "cli" / "gitea" / "gitea_01").exists()
    assert not partial_gui_dir.exists()


def test_append_summary_entry_and_write_aggregate(tmp_path: Path):
    summary_dir = tmp_path / "summary"
    append_summary_entry(
        summary_dir,
        {
            "participant": "gui",
            "software": "inkscape",
            "task_id": "inkscape_01",
            "success": True,
            "score": 1.0,
            "steps": 8,
            "avg_latency_ms": 125.0,
            "e2e_time_s": 5.0,
            "deadlocked": False,
        },
    )
    append_summary_entry(
        summary_dir,
        {
            "participant": "gui",
            "software": "inkscape",
            "task_id": "inkscape_02",
            "success": False,
            "score": 0.0,
            "steps": 6,
            "avg_latency_ms": 140.0,
            "e2e_time_s": 7.0,
            "deadlocked": True,
        },
    )

    entries = load_summary_entries(summary_dir)

    assert [entry["task_id"] for entry in entries] == ["inkscape_01", "inkscape_02"]

    aggregate = build_aggregate(entries, is_comparison=False)
    assert aggregate["inkscape"]["total_tasks"] == 2
    assert aggregate["inkscape"]["passed"] == 1
    assert aggregate["inkscape"]["deadlock_rate"] == 0.5
    assert aggregate["inkscape"]["avg_steps"] == 7.0


def test_summary_results_file_is_json_array(tmp_path: Path):
    summary_dir = tmp_path / "summary"
    append_summary_entry(summary_dir, {"task_id": "one", "software": "obs", "participant": "gui"})

    payload = json.loads((summary_dir / "results.json").read_text(encoding="utf-8"))

    assert isinstance(payload, list)
    assert payload[0]["task_id"] == "one"
