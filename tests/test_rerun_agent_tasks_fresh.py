from pathlib import Path
import json

from scripts.rerun_agent_tasks_fresh import (
    _build_compose_run_command,
    _merge_completed_task_rerun,
)


def test_build_compose_run_command_uses_temp_index_file_for_selected_tasks(tmp_path):
    project_root = tmp_path / "repo"
    host_results_root = project_root / "results"
    host_results_root.mkdir(parents=True)
    host_output_dir = host_results_root / "target"
    host_output_json = host_output_dir / "results.json"
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"
    temp_index = project_root / "evaluation_examples" / ".tmp_drawio_subset.json"

    command = _build_compose_run_command(
        project_root=project_root,
        env_file=env_file,
        compose_file=compose_file,
        project_name="asil-expansion-gui-v2",
        software="drawio",
        provider="openai",
        model="gpt-5.4",
        max_steps=20,
        temp_index_name=temp_index.name,
        host_results_root=host_results_root,
        host_output_dir=host_output_dir,
        host_output_json=host_output_json,
    )

    assert command[-8:-2] == [
        "--test-config-base-dir",
        "evaluation_examples",
        "--task-set",
        temp_index.name,
        "--output-dir",
        "/results/target",
    ]
    assert command[-2:] == ["--output", "/results/target/results.json"]


def test_merge_completed_task_rerun_replaces_only_selected_tasks_and_recomputes_drawio_metrics(tmp_path):
    target_output_dir = tmp_path / "target"
    temp_output_dir = tmp_path / "temp"
    target_output_json = target_output_dir / "results.json"
    temp_output_json = temp_output_dir / "results.json"
    model = "gpt-5.4"
    software = "drawio"
    task_ids = ["drawio_05", "drawio_06"]

    target_root = target_output_dir / "asil_protocol" / "structured_json" / model / software
    temp_root = temp_output_dir / "asil_protocol" / "structured_json" / model / software
    target_root.mkdir(parents=True)
    temp_root.mkdir(parents=True)

    for task_id in ["drawio_01", "drawio_05", "drawio_06"]:
        task_dir = target_root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "result.txt").write_text("old", encoding="utf-8")

    for task_id, marker in [("drawio_05", "new-five"), ("drawio_06", "new-six")]:
        task_dir = temp_root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "result.txt").write_text(marker, encoding="utf-8")

    summary_dir = target_output_dir / "asil_protocol" / "structured_json" / model / "summary"
    summary_dir.mkdir(parents=True)
    (summary_dir / "results.json").write_text(json.dumps([
        {"software": software, "task_id": "drawio_01", "status": "success", "score": 1.0, "steps": 2, "e2e_time_s": 10.0, "difficulty": "simple", "timestamp": "t1"},
        {"software": software, "task_id": "drawio_05", "status": "success", "score": 1.0, "steps": 2, "e2e_time_s": 11.0, "difficulty": "simple", "timestamp": "t1"},
        {"software": software, "task_id": "drawio_06", "status": "failed", "score": 0.0, "steps": 1, "e2e_time_s": 12.0, "difficulty": "simple", "timestamp": "t1"},
        {"software": "gimp", "task_id": "gimp_01", "status": "success", "score": 1.0, "steps": 2, "e2e_time_s": 13.0, "difficulty": "simple", "timestamp": "t1"},
    ], indent=2), encoding="utf-8")
    (summary_dir / "aggregate.json").write_text(json.dumps({
        software: {"success_rate": 2/3, "passed": 2, "total_tasks": 3, "avg_steps": 5.0, "avg_latency_ms": 999.0, "avg_e2e_s": 999.0, "deadlock_rate": 0.0, "avg_coverage": 1.0, "per_software": {software: {"success_rate": 2/3, "total": 3, "passed": 2}}},
        "gimp": {"success_rate": 1.0, "passed": 1, "total_tasks": 1, "avg_steps": 2.0, "avg_latency_ms": 1.0, "avg_e2e_s": 13.0, "deadlock_rate": 0.0, "avg_coverage": 1.0, "per_software": {"gimp": {"success_rate": 1.0, "total": 1, "passed": 1}}},
    }, indent=2), encoding="utf-8")

    target_output_json.write_text(json.dumps({
        software: {
            "aggregate": {"success_rate": 2/3, "passed": 2, "total_tasks": 3, "avg_steps": 5.0, "avg_latency_ms": 999.0, "avg_e2e_s": 999.0, "deadlock_rate": 0.0, "avg_coverage": 1.0},
            "per_software": {software: {"success_rate": 2/3, "total": 3, "passed": 2}},
            "tasks": [
                {"task_id": "drawio_01", "software": software, "difficulty": "simple", "instruction": "keep", "success": True, "score": 1.0, "steps": 2, "e2e_time_s": 10.0, "deadlocked": False, "coverage": 1.0, "trajectory": [{"latency_ms": 100.0}]},
                {"task_id": "drawio_05", "software": software, "difficulty": "simple", "instruction": "old", "success": True, "score": 1.0, "steps": 2, "e2e_time_s": 11.0, "deadlocked": False, "coverage": 1.0, "trajectory": [{"latency_ms": 110.0}]},
                {"task_id": "drawio_06", "software": software, "difficulty": "simple", "instruction": "old", "success": False, "score": 0.0, "steps": 1, "e2e_time_s": 12.0, "deadlocked": False, "coverage": 1.0, "trajectory": [{"latency_ms": 120.0}]},
            ],
        },
        "gimp": {"aggregate": {"success_rate": 1.0, "passed": 1, "total_tasks": 1, "avg_steps": 2.0, "avg_latency_ms": 1.0, "avg_e2e_s": 13.0, "deadlock_rate": 0.0, "avg_coverage": 1.0}, "per_software": {"gimp": {"success_rate": 1.0, "total": 1, "passed": 1}}, "tasks": []},
    }, indent=2), encoding="utf-8")

    temp_summary_dir = temp_output_dir / "asil_protocol" / "structured_json" / model / "summary"
    temp_summary_dir.mkdir(parents=True)
    (temp_summary_dir / "results.json").write_text(json.dumps([
        {"software": software, "task_id": "drawio_05", "status": "success", "score": 1.0, "steps": 1, "e2e_time_s": 20.0, "difficulty": "simple", "timestamp": "t2"},
        {"software": software, "task_id": "drawio_06", "status": "success", "score": 1.0, "steps": 3, "e2e_time_s": 30.0, "difficulty": "simple", "timestamp": "t2"},
    ], indent=2), encoding="utf-8")
    temp_output_json.write_text(json.dumps({
        software: {
            "aggregate": {"success_rate": 1.0},
            "per_software": {software: {"success_rate": 1.0, "total": 2, "passed": 2}},
            "tasks": [
                {"task_id": "drawio_05", "software": software, "difficulty": "simple", "instruction": "new", "success": True, "score": 1.0, "steps": 1, "e2e_time_s": 20.0, "deadlocked": False, "coverage": 1.0, "trajectory": [{"latency_ms": 210.0}, {"latency_ms": 220.0}]},
                {"task_id": "drawio_06", "software": software, "difficulty": "simple", "instruction": "new", "success": True, "score": 1.0, "steps": 3, "e2e_time_s": 30.0, "deadlocked": False, "coverage": 1.0, "trajectory": [{"latency_ms": 310.0}]},
            ],
        }
    }, indent=2), encoding="utf-8")

    _merge_completed_task_rerun(
        target_output_dir=target_output_dir,
        target_output_json=target_output_json,
        temp_output_dir=temp_output_dir,
        temp_output_json=temp_output_json,
        software=software,
        task_ids=task_ids,
        model=model,
    )

    assert (target_root / "drawio_01" / "result.txt").read_text(encoding="utf-8") == "old"
    assert (target_root / "drawio_05" / "result.txt").read_text(encoding="utf-8") == "new-five"
    assert (target_root / "drawio_06" / "result.txt").read_text(encoding="utf-8") == "new-six"

    summary_rows = json.loads((summary_dir / "results.json").read_text(encoding="utf-8"))
    drawio_rows = sorted([r for r in summary_rows if r["software"] == software], key=lambda r: r["task_id"])
    assert [row["task_id"] for row in drawio_rows] == ["drawio_01", "drawio_05", "drawio_06"]
    assert [row["score"] for row in drawio_rows] == [1.0, 1.0, 1.0]

    merged_output = json.loads(target_output_json.read_text(encoding="utf-8"))
    tasks = sorted(merged_output[software]["tasks"], key=lambda row: row["task_id"])
    assert [task["task_id"] for task in tasks] == ["drawio_01", "drawio_05", "drawio_06"]
    assert merged_output[software]["aggregate"]["passed"] == 3
    assert merged_output[software]["aggregate"]["total_tasks"] == 3
    assert merged_output[software]["aggregate"]["success_rate"] == 1.0
    assert round(merged_output[software]["aggregate"]["avg_steps"], 4) == 2.0
    assert round(merged_output[software]["aggregate"]["avg_e2e_s"], 4) == 20.0
    assert round(merged_output[software]["aggregate"]["avg_latency_ms"], 4) == 210.0

    aggregate = json.loads((summary_dir / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate[software]["passed"] == 3
    assert aggregate[software]["success_rate"] == 1.0
    assert aggregate["gimp"]["passed"] == 1
