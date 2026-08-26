"""Tests for the managed docker wrapper around the public benchmark CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import call, patch


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_evaluation_managed.py"
    spec = importlib.util.spec_from_file_location("asil_run_evaluation_managed", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_eval_run_command_wraps_run_benchmark_args():
    module = _load_module()
    project_root = Path("/tmp/project")
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"

    command = module._build_eval_run_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name="asil-expansion-gui-v2",
        forwarded_args=["--task-set", "test_full15.json", "--agent"],
    )

    assert command == [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "-p",
        "asil-expansion-gui-v2",
        "--profile",
        "eval",
        "run",
        "--rm",
        "eval",
        "python",
        "scripts/run_benchmark.py",
        "--task-set",
        "test_full15.json",
        "--agent",
    ]


def test_build_compose_build_command_targets_shared_services():
    module = _load_module()
    project_root = Path("/tmp/project")
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"

    command = module._build_compose_build_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name="asil-gui-prebuild",
        services=["obs-mock", "eval"],
    )

    assert command == [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "-p",
        "asil-gui-prebuild",
        "build",
        "obs-mock",
        "eval",
    ]


def test_rewrite_forwarded_result_paths_maps_relative_results_into_container_mount():
    module = _load_module()
    project_root = Path("/tmp/project")

    rewritten = module._rewrite_forwarded_result_paths(
        [
            "--task-set",
            "test_full15.json",
            "--output-dir",
            "results/candidate",
            "--output=results/candidate/out.json",
        ],
        project_root=project_root,
    )

    assert rewritten == [
        "--task-set",
        "test_full15.json",
        "--output-dir",
        "/results/candidate",
        "--output=/results/candidate/out.json",
    ]


def test_rewrite_forwarded_result_paths_maps_absolute_project_results_path():
    module = _load_module()
    project_root = Path("/tmp/project")

    rewritten = module._rewrite_forwarded_result_paths(
        [
            "--output-dir",
            str(project_root / "results" / "candidate"),
            "--output",
            str(project_root / "results" / "candidate" / "out.json"),
        ],
        project_root=project_root,
    )

    assert rewritten == [
        "--output-dir",
        "/results/candidate",
        "--output",
        "/results/candidate/out.json",
    ]


def test_build_worker_project_name_uses_stable_suffixes():
    module = _load_module()
    assert module._build_worker_project_name("asil-gui", 1) == "asil-gui-w01"
    assert module._build_worker_project_name("asil-gui", 12) == "asil-gui-w12"


def test_stable_round_robin_shards_distributes_tasks_evenly():
    module = _load_module()
    tasks = [
        module.TaskKey("inkscape", "inkscape_01"),
        module.TaskKey("inkscape", "inkscape_02"),
        module.TaskKey("gitea", "gitea_01"),
        module.TaskKey("drawio", "drawio_01"),
        module.TaskKey("obs", "obs_01"),
    ]

    shards = module._stable_round_robin_shards(tasks, 2)

    assert shards == [
        [
            module.TaskKey("inkscape", "inkscape_01"),
            module.TaskKey("gitea", "gitea_01"),
            module.TaskKey("obs", "obs_01"),
        ],
        [
            module.TaskKey("inkscape", "inkscape_02"),
            module.TaskKey("drawio", "drawio_01"),
        ],
    ]


def test_prepare_forwarded_args_for_worker_replaces_task_set_and_output():
    module = _load_module()
    rewritten = module._prepare_forwarded_args_for_worker(
        forwarded_args=[
            "--task-set",
            "test_full15.json",
            "--participant",
            "gui",
            "--num-envs",
            "4",
            "--output",
            "results/out.json",
        ],
        shard_task_set=".generated/shard.w01.json",
        worker_output="results/.worker-results/w01.json",
    )

    assert "--managed-docker" in rewritten
    assert rewritten.count("--task-set") == 1
    assert rewritten[rewritten.index("--task-set") + 1] == ".generated/shard.w01.json"
    assert rewritten[rewritten.index("--output") + 1] == "results/.worker-results/w01.json"
    assert rewritten[rewritten.index("--num-envs") + 1] == "1"


def test_prepare_forwarded_args_for_worker_strips_multi_value_filters():
    module = _load_module()
    rewritten = module._prepare_forwarded_args_for_worker(
        forwarded_args=[
            "--task-set",
            "test_full15.json",
            "--software-filter",
            "gitea",
            "code_server",
            "--task-id-filter",
            "gitea_01",
            "code_server_01",
            "--participant",
            "gui",
            "--output",
            "results/out.json",
        ],
        shard_task_set=".generated/shard.w01.json",
        worker_output="results/.worker-results/w01.json",
    )

    assert "--software-filter" not in rewritten
    assert "--task-id-filter" not in rewritten
    assert "gitea" not in rewritten
    assert "code_server" not in rewritten
    assert "gitea_01" not in rewritten
    assert "code_server_01" not in rewritten


def test_main_runs_eval_and_always_shuts_down_on_success():
    module = _load_module()
    completed = subprocess.CompletedProcess(args=["ok"], returncode=0)

    with (
        patch.object(module.subprocess, "run", side_effect=[completed, completed]) as mock_run,
        patch.object(module, "_cleanup_project_prefix") as mock_cleanup,
    ):
        exit_code = module.main(["--task-set", "test_full15.json", "--agent"])

    project_root = Path(module.__file__).resolve().parent.parent
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"
    run_command = module._build_eval_run_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name="asil-managed",
        forwarded_args=module._rewrite_forwarded_result_paths(
            module._ensure_flag(
                module._replace_or_append_flag(
                    ["--task-set", "test_full15.json", "--agent"],
                    "--num-envs",
                    "1",
                ),
                "--managed-docker",
            ),
            project_root=project_root,
        ),
    )
    down_command = module._build_down_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name="asil-managed",
    )

    assert exit_code == 0
    assert mock_run.call_args_list == [
        call(run_command, cwd=str(project_root), check=True),
        call(down_command, cwd=str(project_root), check=False),
    ]
    mock_cleanup.assert_not_called()


def test_main_auto_cleans_jupyterlab_prefix_before_and_after_single_worker():
    module = _load_module()
    completed = subprocess.CompletedProcess(args=["ok"], returncode=0)

    with (
        patch.object(module.subprocess, "run", side_effect=[completed, completed]),
        patch.object(module, "_cleanup_project_prefix") as mock_cleanup,
    ):
        exit_code = module.main(
            [
                "--compose-project-name",
                "asil-jupyterlab-fixgate-test",
                "--task-set",
                "test_full15.json",
                "--participant",
                "gui",
                "--software-filter",
                "jupyterlab",
            ]
        )

    assert exit_code == 0
    assert mock_cleanup.call_args_list == [
        call("asil-jupyterlab-fixgate-test"),
        call("asil-jupyterlab-fixgate-test"),
    ]


def test_main_dry_run_reports_parallel_plan(tmp_path: Path, capsys):
    module = _load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps(
            {
                "inkscape": ["inkscape_01", "inkscape_02"],
                "gitea": ["gitea_01"],
            }
        ),
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
    assert "Total tasks: 3" in stdout
    assert "Unfinished tasks: 3" in stdout
    assert "shard 1" in stdout
    assert "shard 2" in stdout


def test_main_parallel_preserves_asil_success_hint_policy(tmp_path: Path):
    module = _load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps({"inkscape": ["inkscape_01", "inkscape_02"]}),
        encoding="utf-8",
    )

    with patch.object(module, "_run_parallel_workers", return_value=0) as run_workers:
        exit_code = module.main(
            [
                "--num-envs",
                "2",
                "--task-set",
                "task_set.json",
                "--test-config-base-dir",
                str(config_dir),
                "--asil-success-hint",
                "none",
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )

    assert exit_code == 0
    assert run_workers.call_args.kwargs["config"].asil_success_hint == "none"
    assert "--asil-success-hint" in run_workers.call_args.kwargs["forwarded_args"]


def test_main_parallel_preserves_independent_raw_validation(tmp_path: Path):
    module = _load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps({"inkscape": ["inkscape_01", "inkscape_02"]}),
        encoding="utf-8",
    )

    with patch.object(module, "_run_parallel_workers", return_value=0) as run_workers:
        exit_code = module.main(
            [
                "--num-envs",
                "2",
                "--task-set",
                "task_set.json",
                "--test-config-base-dir",
                str(config_dir),
                "--independent-raw-validation",
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )

    assert exit_code == 0
    assert run_workers.call_args.kwargs["config"].independent_raw_validation is True
    assert "--independent-raw-validation" in run_workers.call_args.kwargs["forwarded_args"]


def test_run_parallel_workers_prebuilds_shared_images_once(tmp_path: Path):
    module = _load_module()
    project_root = Path(module.__file__).resolve().parent.parent
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"

    config = module._benchmark.BenchmarkConfig(
        task_index="test_full15.json",
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        participant="gui",
        run_mode="single",
        provider="openai",
        model="gpt-5.4",
        max_steps=15,
        managed_docker=True,
        num_envs=2,
    )

    task_mapping = {"code_server": ["code_server_01", "code_server_02"]}
    pending_tasks = (
        module.TaskKey("code_server", "code_server_01"),
        module.TaskKey("code_server", "code_server_02"),
    )

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.args = _args
            self.kwargs = _kwargs

        def wait(self):
            return 0

    completed = subprocess.CompletedProcess(args=["ok"], returncode=0)

    with (
        patch.object(module.subprocess, "run", return_value=completed) as mock_run,
        patch.object(module.subprocess, "Popen", side_effect=lambda *a, **k: _FakePopen(*a, **k)),
        patch.object(module, "_rebuild_shared_outputs"),
    ):
        exit_code = module._run_parallel_workers(
            env_file=env_file,
            compose_file=compose_file,
            base_project_name="asil-gui-prebuild",
            forwarded_args=[
                "--task-set",
                "test_full15.json",
                "--participant",
                "gui",
                "--provider",
                "openai",
                "--model",
                "gpt-5.4",
                "--max-steps",
                "15",
                "--output-dir",
                "results/candidate",
                "--output",
                "results/candidate/results.json",
            ],
            config=config,
            task_mapping=task_mapping,
            pending_tasks=pending_tasks,
        )

    assert exit_code == 0
    prebuild_command = module._build_compose_build_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name="asil-gui-prebuild",
        services=["obs-mock", "eval"],
    )
    assert mock_run.call_args_list[0] == call(prebuild_command, cwd=str(project_root), check=True)


def test_run_parallel_workers_cleans_generated_task_sets_on_success(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = module._benchmark.BenchmarkConfig(
        task_index="test_full15.json",
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        participant="gui",
        run_mode="single",
        provider="openai",
        model="gpt-5.4",
        max_steps=15,
        managed_docker=True,
        num_envs=2,
    )
    pending_tasks = (
        module.TaskKey("code_server", "code_server_01"),
        module.TaskKey("code_server", "code_server_02"),
    )

    class _FakePopen:
        def wait(self):
            return 0

    completed = subprocess.CompletedProcess(args=["ok"], returncode=0)

    with (
        patch.object(module, "PROJECT_ROOT", project_root),
        patch.object(module.subprocess, "run", return_value=completed),
        patch.object(module.subprocess, "Popen", return_value=_FakePopen()),
        patch.object(module, "_restore_host_ownership"),
        patch.object(module, "_rebuild_shared_outputs"),
    ):
        exit_code = module._run_parallel_workers(
            env_file=project_root / ".env",
            compose_file=project_root / "docker" / "docker-compose.yml",
            base_project_name="asil-gui-cleanup",
            forwarded_args=["--task-set", "test_full15.json", "--participant", "gui"],
            config=config,
            task_mapping={"code_server": ["code_server_01", "code_server_02"]},
            pending_tasks=pending_tasks,
        )

    assert exit_code == 0
    assert not list((project_root / "evaluation_examples").glob(".generated-*"))


def test_run_parallel_workers_keeps_generated_task_sets_on_worker_failure(tmp_path: Path):
    module = _load_module()
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = module._benchmark.BenchmarkConfig(
        task_index="test_full15.json",
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        participant="gui",
        run_mode="single",
        provider="openai",
        model="gpt-5.4",
        max_steps=15,
        managed_docker=True,
        num_envs=2,
    )
    pending_tasks = (
        module.TaskKey("code_server", "code_server_01"),
        module.TaskKey("code_server", "code_server_02"),
    )

    class _FakePopen:
        def wait(self):
            return 7

    completed = subprocess.CompletedProcess(args=["ok"], returncode=0)

    with (
        patch.object(module, "PROJECT_ROOT", project_root),
        patch.object(module.subprocess, "run", return_value=completed),
        patch.object(module.subprocess, "Popen", return_value=_FakePopen()),
        patch.object(module, "_restore_host_ownership"),
        patch.object(module, "_rebuild_shared_outputs"),
    ):
        exit_code = module._run_parallel_workers(
            env_file=project_root / ".env",
            compose_file=project_root / "docker" / "docker-compose.yml",
            base_project_name="asil-gui-keep-failed",
            forwarded_args=["--task-set", "test_full15.json", "--participant", "gui"],
            config=config,
            task_mapping={"code_server": ["code_server_01", "code_server_02"]},
            pending_tasks=pending_tasks,
        )

    assert exit_code == 7
    remaining_shards = sorted(
        path.name for path in (project_root / "evaluation_examples").glob(".generated-*")
    )
    assert remaining_shards == [
        ".generated-asil-gui-keep-failed-test_full15.w01.json",
        ".generated-asil-gui-keep-failed-test_full15.w02.json",
    ]
