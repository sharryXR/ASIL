"""Tests for fresh-container software rerun orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_rerun_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "rerun_agent_software_fresh.py"
    spec = importlib.util.spec_from_file_location("rerun_agent_software_fresh", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_delete_software_result_tree_removes_only_selected_software(tmp_path: Path):
    module = _load_rerun_module()
    result_root = tmp_path / "asil_protocol" / "structured_json" / "gpt-5.4"
    (result_root / "code_server" / "code_server_01").mkdir(parents=True)
    (result_root / "drawio" / "drawio_01").mkdir(parents=True)
    (result_root / "nautilus" / "nautilus_01").mkdir(parents=True)

    module._delete_software_result_tree(result_root, "code_server")

    assert not (result_root / "code_server").exists()
    assert (result_root / "drawio").exists()
    assert (result_root / "nautilus").exists()


def test_build_compose_run_command_targets_temp_results_in_container(tmp_path: Path):
    module = _load_rerun_module()
    project_root = tmp_path / "asil"
    results_root = project_root / "results"
    results_root.mkdir(parents=True)
    temp_output_dir = results_root / ".rerun_tmp" / "code_server"
    temp_output_json = temp_output_dir / "results_code_server.json"

    command = module._build_compose_run_command(
        project_root=project_root,
        env_file=project_root / ".env",
        compose_file=project_root / "docker" / "docker-compose.yml",
        project_name="asil-expansion-gui-v2",
        software="code_server",
        provider="openai",
        model="gpt-5.4",
        max_steps=20,
        test_config_base_dir="evaluation_examples",
        test_all="test_expansion_all.json",
        host_results_root=results_root,
        host_output_dir=temp_output_dir,
        host_output_json=temp_output_json,
    )

    assert command[:6] == [
        "docker",
        "compose",
        "--env-file",
        str(project_root / ".env"),
        "-f",
        str(project_root / "docker" / "docker-compose.yml"),
    ]
    assert "--profile" in command
    assert "eval" in command
    assert "--software" in command and "code_server" in command
    assert "--task-set" in command and "test_expansion_all.json" in command
    assert str(Path("/results/.rerun_tmp/code_server")) in command
    assert str(Path("/results/.rerun_tmp/code_server/results_code_server.json")) in command


def test_merge_software_run_updates_only_selected_software_entries(tmp_path: Path):
    module = _load_rerun_module()
    target_output_dir = tmp_path / "final"
    model = "gpt-5.4"
    target_result_root = target_output_dir / "asil_protocol" / "structured_json" / model
    (target_result_root / "code_server" / "code_server_01").mkdir(parents=True)
    (target_result_root / "nautilus" / "nautilus_01").mkdir(parents=True)
    (target_result_root / "nautilus" / "nautilus_01" / "result.txt").write_text("1.0", encoding="utf-8")
    (target_result_root / "code_server" / "code_server_01" / "result.txt").write_text("0.0", encoding="utf-8")

    target_summary_dir = target_result_root / "summary"
    target_summary_dir.mkdir(parents=True)
    (target_summary_dir / "results.json").write_text(
        json.dumps(
            [
                {"software": "code_server", "task_id": "code_server_01", "status": "failed"},
                {"software": "nautilus", "task_id": "nautilus_01", "status": "success"},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (target_summary_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "code_server": {"passed": 0, "total_tasks": 1},
                "nautilus": {"passed": 1, "total_tasks": 1},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    target_output_json = target_output_dir / "results_gpt54_full10.json"
    target_output_json.write_text(
        json.dumps(
            {
                "code_server": {"aggregate": {"passed": 0}},
                "nautilus": {"aggregate": {"passed": 1}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_output_dir = tmp_path / "temp_code_server"
    temp_result_root = temp_output_dir / "asil_protocol" / "structured_json" / model
    (temp_result_root / "code_server" / "code_server_01").mkdir(parents=True)
    (temp_result_root / "code_server" / "code_server_01" / "result.txt").write_text("1.0", encoding="utf-8")
    temp_summary_dir = temp_result_root / "summary"
    temp_summary_dir.mkdir(parents=True)
    (temp_summary_dir / "results.json").write_text(
        json.dumps(
            [{"software": "code_server", "task_id": "code_server_01", "status": "success"}],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (temp_summary_dir / "aggregate.json").write_text(
        json.dumps({"code_server": {"passed": 1, "total_tasks": 1}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_output_json = temp_output_dir / "results_code_server.json"
    temp_output_json.write_text(
        json.dumps({"code_server": {"aggregate": {"passed": 1}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    module._merge_completed_software_run(
        target_output_dir=target_output_dir,
        target_output_json=target_output_json,
        temp_output_dir=temp_output_dir,
        temp_output_json=temp_output_json,
        software="code_server",
        model=model,
    )

    assert (target_result_root / "code_server" / "code_server_01" / "result.txt").read_text(encoding="utf-8") == "1.0"

    merged_results = json.loads((target_summary_dir / "results.json").read_text(encoding="utf-8"))
    assert merged_results == [
        {"software": "nautilus", "task_id": "nautilus_01", "status": "success"},
        {"software": "code_server", "task_id": "code_server_01", "status": "success"},
    ]

    merged_aggregate = json.loads((target_summary_dir / "aggregate.json").read_text(encoding="utf-8"))
    assert merged_aggregate["code_server"]["passed"] == 1
    assert merged_aggregate["nautilus"]["passed"] == 1

    merged_output = json.loads(target_output_json.read_text(encoding="utf-8"))
    assert merged_output["code_server"]["aggregate"]["passed"] == 1
    assert merged_output["nautilus"]["aggregate"]["passed"] == 1
