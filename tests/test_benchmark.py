from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from asil.benchmark import (
    BenchmarkConfig,
    BenchmarkRunResult,
    SOFTWARE_CHOICES,
    _load_tasks_for_software,
    _write_args_json,
    _task_result_from_gui_exception,
    _result_root_for_config,
    build_arg_parser,
    config_from_namespace,
    run_benchmark,
)


def test_config_from_namespace_maps_agent_mode_and_task_set(tmp_path: Path):
    parser = build_arg_parser()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps(
            {
                "drawio": ["drawio_01"],
                "jupyterlab": ["jupyterlab_01"],
            }
        ),
        encoding="utf-8",
    )

    args = parser.parse_args(
        [
            "--agent",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4",
            "--task-set",
            "task_set.json",
            "--test-config-base-dir",
            str(config_dir),
            "--output-dir",
            str(tmp_path / "results"),
            "--output",
            str(tmp_path / "results.json"),
        ]
    )

    config = config_from_namespace(args)

    assert config.participant == "asil"
    assert config.run_mode == "single"
    assert config.asil_execution == "agentic"
    assert config.software == ("drawio", "jupyterlab")
    assert config.task_index == "task_set.json"
    assert config.provider == "openai"
    assert config.model == "gpt-5.4"


def test_config_from_namespace_maps_new_participant_flags(tmp_path: Path):
    parser = build_arg_parser()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps({"nautilus": ["nautilus_01"]}),
        encoding="utf-8",
    )

    args = parser.parse_args(
        [
            "--participant",
            "gui",
            "--run-mode",
            "single",
            "--task-set",
            "task_set.json",
            "--test-config-base-dir",
            str(config_dir),
            "--output-dir",
            str(tmp_path / "results"),
            "--output",
            str(tmp_path / "results.json"),
        ]
    )

    config = config_from_namespace(args)

    assert config.participant == "gui"
    assert config.run_mode == "single"
    assert config.asil_execution == "deterministic"
    assert config.software == ("nautilus",)


def test_config_from_namespace_maps_comparison_mode():
    parser = build_arg_parser()

    args = parser.parse_args(["--run-mode", "comparison"])
    config = config_from_namespace(args)

    assert config.run_mode == "comparison"
    assert config.comparison_participants == ("asil", "cli", "gui")


def test_config_from_namespace_keeps_agentic_asil_execution_in_comparison():
    parser = build_arg_parser()

    args = parser.parse_args(["--run-mode", "comparison", "--asil-execution", "agentic"])
    config = config_from_namespace(args)

    assert config.run_mode == "comparison"
    assert config.asil_execution == "agentic"


def test_config_from_namespace_captures_resume_and_parallel_flags():
    parser = build_arg_parser()
    args = parser.parse_args(["--resume", "--num-envs", "3", "--rerun-failed-only"])

    config = config_from_namespace(args)

    assert config.resume is True
    assert config.rerun_failed_only is True
    assert config.num_envs == 3


def test_config_from_namespace_controls_asil_success_hint_policy():
    parser = build_arg_parser()

    default_config = config_from_namespace(parser.parse_args([]))
    no_hint_config = config_from_namespace(
        parser.parse_args(["--asil-success-hint", "none"])
    )

    assert default_config.asil_success_hint == "evaluator"
    assert no_hint_config.asil_success_hint == "none"


def test_config_from_namespace_controls_independent_raw_validation():
    parser = build_arg_parser()

    default_config = config_from_namespace(parser.parse_args([]))
    enabled_config = config_from_namespace(
        parser.parse_args(["--independent-raw-validation"])
    )

    assert default_config.independent_raw_validation is False
    assert enabled_config.independent_raw_validation is True


def test_config_from_namespace_preserves_task_id_filter(tmp_path: Path):
    parser = build_arg_parser()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps({"drawio": ["drawio_01", "drawio_02"]}),
        encoding="utf-8",
    )

    args = parser.parse_args(
        [
            "--task-set",
            "task_set.json",
            "--test-config-base-dir",
            str(config_dir),
            "--task-id-filter",
            "drawio_02",
        ]
    )

    config = config_from_namespace(args)

    assert config.software == ("drawio",)
    assert config.task_id_filter == ("drawio_02",)


def test_result_root_for_config_uses_osworld_style_layouts(tmp_path: Path):
    base = tmp_path / "results"

    assert _result_root_for_config(
        BenchmarkConfig(
            task_index="test_full15.json",
            output_dir=base,
            output_json=base / "out.json",
            participant="gui",
            run_mode="single",
            provider="openai",
            model="gpt-5.4",
        )
    ) == base / "pyautogui" / "screenshot" / "gpt-5.4"

    assert _result_root_for_config(
        BenchmarkConfig(
            task_index="test_full15.json",
            output_dir=base,
            output_json=base / "out.json",
            participant="cli",
            run_mode="single",
        )
    ) == base / "shell" / "terminal" / "cli-baseline"

    assert _result_root_for_config(
        BenchmarkConfig(
            task_index="test_full15.json",
            output_dir=base,
            output_json=base / "out.json",
            participant="asil",
            run_mode="single",
            asil_execution="deterministic",
        )
    ) == base / "semantic" / "structured" / "asil-deterministic"

    assert _result_root_for_config(
        BenchmarkConfig(
            task_index="test_full15.json",
            output_dir=base,
            output_json=base / "out.json",
            participant="asil",
            run_mode="single",
            asil_execution="agentic",
            model="gpt-5.4",
        )
    ) == base / "semantic" / "structured" / "asil-gpt-5.4"

    assert _result_root_for_config(
        BenchmarkConfig(
            task_index="test_full15.json",
            output_dir=base,
            output_json=base / "out.json",
            participant="asil",
            run_mode="comparison",
            asil_execution="agentic",
            model="gpt-5.4",
        )
    ) == base / "comparison" / "agentic__gpt-5.4"


def test_run_benchmark_delegates_to_namespace_executor(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_small.json").write_text(
        json.dumps({"inkscape": ["inkscape_01"]}),
        encoding="utf-8",
    )
    config = BenchmarkConfig(
        task_index="test_small.json",
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        participant="asil",
        run_mode="single",
        asil_execution="deterministic",
        test_config_base_dir=config_dir,
        dry_run=True,
    )

    with patch("asil.benchmark._execute_from_namespace", return_value=0) as mock_execute:
        result = run_benchmark(config)

    assert isinstance(result, BenchmarkRunResult)
    assert result.exit_code == 0
    assert result.config.software == ("inkscape",)
    assert result.config.task_index == config.task_index
    mock_execute.assert_called_once()
    args = mock_execute.call_args.args[0]
    assert args.task_set == "test_small.json"
    assert args.software == ["inkscape"]
    assert args.software_filter == ["inkscape"]
    assert args.agent is False
    assert args.comparison is False
    assert args.participant == "asil"
    assert args.run_mode == "single"
    assert args.asil_execution == "deterministic"
    assert args.asil_success_hint == "evaluator"
    assert args.independent_raw_validation is False


def test_run_benchmark_preserves_task_id_filter_in_namespace(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_small.json").write_text(
        json.dumps({"inkscape": ["inkscape_01", "inkscape_02"]}),
        encoding="utf-8",
    )
    config = BenchmarkConfig(
        task_index="test_small.json",
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        participant="asil",
        run_mode="single",
        asil_execution="deterministic",
        software=("inkscape",),
        task_id_filter=("inkscape_02",),
        test_config_base_dir=config_dir,
        dry_run=True,
    )

    with patch("asil.benchmark._execute_from_namespace", return_value=0) as mock_execute:
        run_benchmark(config)

    args = mock_execute.call_args.args[0]
    assert args.task_id_filter == ["inkscape_02"]


def test_run_benchmark_preserves_rebuttal_audit_controls_in_materialized_config(
    tmp_path: Path,
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_small.json").write_text(
        json.dumps({"inkscape": ["inkscape_01"]}),
        encoding="utf-8",
    )
    config = BenchmarkConfig(
        task_index="test_small.json",
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        participant="asil",
        run_mode="single",
        asil_execution="agentic",
        asil_success_hint="none",
        independent_raw_validation=True,
        test_config_base_dir=config_dir,
    )

    with patch("asil.benchmark._execute_from_namespace", return_value=0) as execute:
        result = run_benchmark(config)

    args = execute.call_args.args[0]
    assert result.config.asil_success_hint == "none"
    assert result.config.independent_raw_validation is True
    assert args.asil_success_hint == "none"
    assert args.independent_raw_validation is True


def test_args_json_records_rebuttal_audit_controls(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--participant",
            "asil",
            "--asil-execution",
            "agentic",
            "--asil-success-hint",
            "none",
            "--independent-raw-validation",
        ]
    )
    args.software = ["inkscape"]

    _write_args_json(args, tmp_path)

    payload = json.loads((tmp_path / "args.json").read_text())
    assert payload["asil_success_hint"] == "none"
    assert payload["independent_raw_validation"] is True


def test_load_tasks_for_software_accepts_cross_software_task_id_filters(tmp_path: Path):
    import argparse

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "examples").symlink_to(
        Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples",
        target_is_directory=True,
    )
    (config_dir / "test_small.json").write_text(
        json.dumps(
            {
                "drawio": ["drawio_01"],
                "jupyterlab": ["jupyterlab_01"],
            }
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        task_set="test_small.json",
        test_config_base_dir=config_dir,
        task_id_filter=["drawio_01", "jupyterlab_01"],
        software_filter=["drawio", "jupyterlab"],
        software=["drawio", "jupyterlab"],
    )

    drawio_tasks = _load_tasks_for_software(args, "drawio")
    jupyter_tasks = _load_tasks_for_software(args, "jupyterlab")

    assert [task.id for task in drawio_tasks] == ["drawio_01"]
    assert [task.id for task in jupyter_tasks] == ["jupyterlab_01"]


def test_execution_mode_conflict_is_rejected():
    parser = build_arg_parser()
    args = parser.parse_args(["--agent", "--execution-mode", "comparison"])

    try:
        config_from_namespace(args)
        assert False, "Expected conflicting execution mode flags to raise"
    except ValueError as exc:
        assert "Conflicting execution mode flags" in str(exc)


def test_execution_mode_compatibility_maps_to_new_semantics():
    parser = build_arg_parser()
    args = parser.parse_args(["--execution-mode", "agent"])

    config = config_from_namespace(args)

    assert config.participant == "asil"
    assert config.run_mode == "single"
    assert config.asil_execution == "agentic"


def test_test_full15_contains_all_current_software_and_300_tasks():
    path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "test_full15.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert sorted(payload) == sorted(
        [
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
        ]
    )
    assert set(payload).issubset(set(SOFTWARE_CHOICES))
    assert all(len(tasks) == 20 for tasks in payload.values())
    assert sum(len(tasks) for tasks in payload.values()) == 300


def test_test_gui_smoke_contains_expected_desktop_and_web_representatives():
    path = Path(__file__).resolve().parent.parent / "evaluation_examples" / "test_gui_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "inkscape": ["inkscape_01"],
        "libreoffice": ["libreoffice_01"],
        "obs": ["obs_01"],
        "gitea": ["gitea_01"],
        "drawio": ["drawio_01"],
        "jupyterlab": ["jupyterlab_01"],
    }


def test_config_from_namespace_resolves_all_software_from_task_set(tmp_path: Path):
    parser = build_arg_parser()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "task_set.json").write_text(
        json.dumps(
            {
                "inkscape": ["inkscape_01"],
                "gitea": ["gitea_01"],
                "drawio": ["drawio_01"],
            }
        ),
        encoding="utf-8",
    )

    args = parser.parse_args(
        [
            "--task-set",
            "task_set.json",
            "--test-config-base-dir",
            str(config_dir),
        ]
    )

    config = config_from_namespace(args)

    assert config.software == ("inkscape", "gitea", "drawio")


def test_task_result_from_gui_exception_preserves_startup_failure_category(tmp_path: Path):
    from asil.gui_agent.session import GUISessionStartupError

    class FakeAdapter:
        app_name = "Fake GUI App"

    class FakeTask:
        id = "fake_01"
        software = "fake"
        difficulty = "simple"
        instruction = "Launch the app."

    task_dir = tmp_path / "fake" / "fake_01"
    result = _task_result_from_gui_exception(
        FakeAdapter(),
        FakeTask(),
        GUISessionStartupError("browser_crashed", "Browser crashed during startup."),
        task_dir,
    )

    payload = json.loads((task_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["error_category"] == "browser_crashed"
    assert "browser_crashed" in (task_dir / "runtime_error.txt").read_text(encoding="utf-8")
    assert not (task_dir / "result.txt").exists()
    assert result.steps == 0
    assert result.step_results[0].action_type == "STARTUP_FAIL"
    assert getattr(result, "_skip_summary", False) is True
