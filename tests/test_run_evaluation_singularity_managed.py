"""Tests for the managed Singularity benchmark wrapper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_evaluation_singularity_managed.py"
    spec = importlib.util.spec_from_file_location("asil_run_evaluation_singularity_managed", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_port_plan_uses_stable_worker_offsets():
    module = _load_module()

    plan = module._port_plan(3, base_port=31000, port_stride=20)

    assert plan.as_dict() == {
        "gitea": 31040,
        "obs_mock": 31041,
        "code_server": 31042,
        "jupyterlab": 31043,
        "drawio": 31044,
    }


def test_port_plan_rejects_too_small_stride():
    module = _load_module()

    try:
        module._port_plan(1, base_port=31000, port_stride=4)
    except ValueError as exc:
        assert "port_stride=4" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_singularity_exec_command_uses_cleanenv_binds_and_writable_tmpfs():
    module = _load_module()

    command = module._build_singularity_exec_command(
        singularity_bin="singularity",
        image_path=Path("/images/asil_eval.sif"),
        binds=[(Path("/host/results"), "/results", None), (Path("/host/src"), "/app/src", "ro")],
        command=["python", "/app/scripts/run_benchmark.py", "--task-set", "test_full15.json"],
        writable_tmpfs=True,
    )

    assert command == [
        "singularity",
        "exec",
        "--cleanenv",
        "--writable-tmpfs",
        "--bind",
        "/host/results:/results",
        "--bind",
        "/host/src:/app/src:ro",
        "/images/asil_eval.sif",
        "python",
        "/app/scripts/run_benchmark.py",
        "--task-set",
        "test_full15.json",
    ]


def test_prepare_worker_args_reuses_managed_docker_result_shape():
    module = _load_module()

    rewritten = module._prepare_forwarded_args_for_singularity_worker(
        forwarded_args=[
            "--task-set",
            "test_full15.json",
            "--participant",
            "asil",
            "--asil-execution",
            "deterministic",
            "--num-envs",
            "8",
            "--output",
            "results/out.json",
        ],
        shard_task_set=".generated-shard.w01.json",
        worker_output="results/.worker-results/w01.json",
    )

    assert "--managed-docker" in rewritten
    assert rewritten[rewritten.index("--num-envs") + 1] == "1"
    assert rewritten[rewritten.index("--task-set") + 1] == ".generated-shard.w01.json"
    assert rewritten[rewritten.index("--output") + 1] == "results/.worker-results/w01.json"


def test_load_base_env_maps_openai_base_url_to_agent_env(tmp_path: Path, monkeypatch):
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_BASE_URL=http://127.0.0.1:18082/v1\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    env = module._load_base_env(env_file)

    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:18082/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:18082/v1"


def test_singularity_process_env_uses_prefixes_instead_of_command_line():
    module = _load_module()

    process_env = module._singularity_process_env(
        {"PATH": "/usr/bin"},
        {"OPENAI_API_KEY": "secret-value", "GITEA_URL": "http://127.0.0.1:31000"},
    )

    assert process_env["SINGULARITYENV_OPENAI_API_KEY"] == "secret-value"
    assert process_env["APPTAINERENV_OPENAI_API_KEY"] == "secret-value"
    assert process_env["SINGULARITYENV_GITEA_URL"] == "http://127.0.0.1:31000"


def test_eval_env_defaults_obs_to_mock_service(tmp_path: Path):
    module = _load_module()
    runtime = module.WorkerRuntime.create(runtime_root=tmp_path, run_slug="run", worker_index=1)
    stack = module.WorkerStack(
        singularity_bin="singularity",
        sif_dir=tmp_path / "images",
        runtime=runtime,
        ports=module._port_plan(1, base_port=31000, port_stride=20),
        display=module._display_plan(1, base_display=90),
        base_env={},
        writable_tmpfs=True,
        health_timeout=1,
    )

    env = stack._eval_env()

    assert env["OBS_REAL_GUI"] == "false"
    assert env["OBS_WS_HOST"] == "127.0.0.1"
    assert env["OBS_WS_PORT"] == "31001"
    assert env["DISPLAY"] == ":90"
    assert env["ASIL_XVFB_DISPLAY"] == ":90"


def test_eval_env_assigns_distinct_worker_display(tmp_path: Path):
    module = _load_module()
    runtime = module.WorkerRuntime.create(runtime_root=tmp_path, run_slug="run", worker_index=3)
    stack = module.WorkerStack(
        singularity_bin="singularity",
        sif_dir=tmp_path / "images",
        runtime=runtime,
        ports=module._port_plan(3, base_port=31000, port_stride=20),
        display=module._display_plan(3, base_display=90),
        base_env={},
        writable_tmpfs=True,
        health_timeout=1,
    )

    env = stack._eval_env()

    assert env["DISPLAY"] == ":92"
    assert env["ASIL_XVFB_DISPLAY"] == ":92"


def test_main_dry_run_reports_parallel_plan_and_ports_without_sifs(tmp_path: Path, capsys):
    module = _load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps({"gitea": ["gitea_01"], "drawio": ["drawio_01"]}),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--num-envs",
            "2",
            "--task-set",
            "task_set.json",
            "--test-config-base-dir",
            str(config_dir),
            "--dry-run",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Total tasks: 2" in stdout
    assert "worker w01" in stdout
    assert "display=:90" in stdout
    assert "worker w02" in stdout
    assert "display=:91" in stdout


def test_main_single_worker_invokes_singularity_worker(tmp_path: Path):
    module = _load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(json.dumps({"gitea": ["gitea_01"]}), encoding="utf-8")

    with (
        patch.object(module, "_resolve_singularity_bin", return_value="singularity"),
        patch.object(module, "_validate_sifs"),
        patch.object(module, "_check_ports_available"),
        patch.object(module, "_install_signal_handlers"),
        patch.object(module, "_run_one_worker", return_value=0) as run_one,
        patch.object(module._managed, "_rebuild_shared_outputs"),
        patch.object(module, "_write_singularity_metadata"),
    ):
        exit_code = module.main(
            [
                "--num-envs",
                "1",
                "--sif-dir",
                str(tmp_path / "images"),
                "--runtime-root",
                str(tmp_path / "runtime"),
                "--task-set",
                "task_set.json",
                "--test-config-base-dir",
                str(config_dir),
                "--participant",
                "asil",
                "--asil-execution",
                "deterministic",
            ]
        )

    assert exit_code == 0
    assert run_one.call_args.kwargs["worker_index"] == 1
    assert run_one.call_args.kwargs["ports"].as_dict()["gitea"] == module.DEFAULT_BASE_PORT
    assert run_one.call_args.kwargs["display"].display == ":90"
