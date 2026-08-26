from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.verify_rebuttal_docker as verifier

from scripts.verify_rebuttal_docker import (
    BootstrapReport,
    GateResult,
    HostFacts,
    finalize_report,
    validate_cleanup,
    validate_compose_config,
    validate_image_inspection,
    validate_public_readiness,
    validate_runtime_probe,
    validate_service_state,
    build_runtime_probe_command,
    build_smoke_command,
    compute_build_provenance,
    parse_compose_ps,
    validate_smoke_results,
    CommandResult,
    run_docker_verification,
    count_task_fixtures,
    evaluate_host_facts,
    validate_required_repository_files,
    write_report,
    REQUIRED_REPOSITORY_FILES,
)


ROOT = Path(__file__).resolve().parents[1]


def _host(**overrides: object) -> HostFacts:
    values: dict[str, object] = {
        "os_id": "ubuntu",
        "os_version": "22.04",
        "architecture": "x86_64",
        "docker_server_version": "28.5.1",
        "compose_version": "2.40.3",
        "docker_daemon_ready": True,
        "disk_free_bytes": 40 * 1024**3,
        "memory_total_bytes": 20 * 1024**3,
        "network_endpoints": {
            "registry-1.docker.io": True,
            "archive.ubuntu.com": True,
            "pypi.org": True,
            "playwright.azureedge.net": True,
        },
    }
    values.update(overrides)
    return HostFacts(**values)  # type: ignore[arg-type]


def _statuses(results: list[GateResult]) -> dict[str, str]:
    return {result.gate_id: result.status for result in results}


@pytest.mark.parametrize("version", ["22.04", "24.04"])
def test_supported_host_passes_version_and_platform_gates(version: str) -> None:
    statuses = _statuses(evaluate_host_facts(_host(os_version=version)))

    assert statuses["host.platform"] == "pass"
    assert statuses["host.docker"] == "pass"
    assert statuses["host.compose"] == "pass"
    assert statuses["host.disk"] == "pass"
    assert statuses["host.memory"] == "pass"
    assert statuses["host.network"] == "pass"


def test_unsupported_architecture_and_old_docker_fail_closed() -> None:
    statuses = _statuses(
        evaluate_host_facts(
            _host(architecture="aarch64", docker_server_version="23.0.6")
        )
    )

    assert statuses["host.platform"] == "fail"
    assert statuses["host.docker"] == "fail"


def test_low_disk_fails_and_low_memory_warns() -> None:
    results = evaluate_host_facts(
        _host(disk_free_bytes=20 * 1024**3, memory_total_bytes=8 * 1024**3)
    )
    by_id = {result.gate_id: result for result in results}

    assert by_id["host.disk"].status == "fail"
    assert by_id["host.disk"].mandatory is True
    assert by_id["host.memory"].status == "warn"
    assert by_id["host.memory"].mandatory is False


def test_ready_requires_every_mandatory_gate() -> None:
    report = BootstrapReport(
        schema_version="1.0",
        started_at="2026-07-12T00:00:00Z",
        finished_at="",
        commit="abc123",
        dirty=False,
        gates=[
            GateResult("required.ok", "pass", {}),
            GateResult("optional.warn", "warn", {}, mandatory=False),
        ],
        ready=False,
    )
    assert finalize_report(report).ready is True

    report.gates.append(GateResult("required.fail", "fail", {}, "fix it"))
    assert finalize_report(report).ready is False


def test_report_refuses_secret_shapes(tmp_path: Path) -> None:
    report = BootstrapReport(
        schema_version="1.0",
        started_at="2026-07-12T00:00:00Z",
        finished_at="2026-07-12T00:01:00Z",
        commit="abc123",
        dirty=False,
        gates=[
            GateResult(
                "bad",
                "pass",
                {"accidental": "ghp_" + "a" * 32},
            )
        ],
        ready=True,
    )

    with pytest.raises(ValueError, match="credential-shaped"):
        write_report(report, tmp_path / "report.json")


def test_report_writes_stable_non_secret_json(tmp_path: Path) -> None:
    report = BootstrapReport(
        schema_version="1.0",
        started_at="2026-07-12T00:00:00Z",
        finished_at="2026-07-12T00:01:00Z",
        commit="abc123",
        dirty=False,
        gates=[GateResult("host.platform", "pass", {"os": "ubuntu-22.04"})],
        ready=True,
    )
    path = tmp_path / "report.json"

    write_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["gates"][0]["gate_id"] == "host.platform"
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_repository_gate_requires_tracked_bootstrap_inputs(tmp_path: Path) -> None:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/Dockerfile.eval").write_text("FROM ubuntu\n")

    results = validate_required_repository_files(tmp_path)

    assert any(result.status == "fail" for result in results)
    missing = next(result for result in results if result.gate_id == "repository.files")
    assert "docker/docker-compose.yml" in missing.evidence["missing"]


def test_repository_gate_includes_the_bootstrap_and_frozen_smoke_inputs() -> None:
    assert {
        ".env.example",
        "scripts/bootstrap_rebuttal_docker.sh",
        "scripts/verify_rebuttal_docker.py",
        "evaluation_examples/test_rebuttal_docker_smoke.json",
        "evaluation_examples/test_full15_multi_apps_380.json",
        "evaluation_examples/test_full15_realwork_hard.json",
        "evaluation_examples/test_easy60.json",
    } <= set(REQUIRED_REPOSITORY_FILES)


def test_compose_gate_rejects_latest_and_unpinned_third_party_images() -> None:
    config = {
        "services": {
            "eval": {"image": "asil-eval:local", "build": {"context": ".."}},
            "obs-mock": {
                "image": "asil-obs-mock:local",
                "build": {"context": "."},
            },
            "gitea": {"image": "gitea/gitea:1.21"},
            "init": {"image": "curlimages/curl:latest"},
        }
    }

    result = validate_compose_config(config)

    assert result.status == "fail"
    assert set(result.evidence["mutable_services"]) == {"gitea", "init"}


def test_image_gate_rejects_stale_labels_and_wrong_architecture() -> None:
    images = {
        "asil-eval:local": {
            "Id": "sha256:1",
            "Architecture": "arm64",
            "Config": {"Labels": {"org.asil.git-commit": "old"}},
        }
    }
    result = validate_image_inspection(
        images, {"org.asil.git-commit": "new"}, required_images=("asil-eval:local",)
    )

    assert result.status == "fail"
    assert result.evidence["wrong_architecture"] == ["asil-eval:local"]
    assert result.evidence["label_mismatches"]


def test_service_runtime_and_cleanup_gates_fail_closed() -> None:
    services = validate_service_state(
        {"gitea": "healthy", "obs-mock": "unhealthy"},
        required=("gitea", "obs-mock"),
    )
    runtime = validate_runtime_probe(
        {
            "adapter_count": 15,
            "schema_count": 14,
            "chromium_launch": True,
            "xvfb": True,
            "openbox": True,
        }
    )
    cleanup = validate_cleanup(
        {"containers": ["asil-rebuttal-bootstrap-eval-1"], "networks": []}
    )

    assert services.status == "fail"
    assert runtime.status == "fail"
    assert cleanup.status == "fail"


def test_readiness_gate_cross_checks_selected_task_ids(tmp_path: Path) -> None:
    (tmp_path / "evaluation_examples/examples/demo").mkdir(parents=True)
    (tmp_path / "evaluation_examples/examples/demo/demo_01.json").write_text("{}")
    for filename in (
        "test_full15_multi_apps_380.json",
        "test_full15_realwork_hard.json",
        "test_easy60.json",
        "test_rebuttal_docker_smoke.json",
    ):
        task_ids = ["demo_01", "demo_missing"] if filename == "test_easy60.json" else ["demo_01"]
        (tmp_path / "evaluation_examples" / filename).write_text(
            json.dumps({"demo": task_ids})
        )

    result = validate_public_readiness(tmp_path)

    assert result.status == "fail"
    assert "demo_missing" in result.evidence["missing_task_ids"]


def test_build_provenance_is_independent_of_checkout_path(tmp_path: Path) -> None:
    roots = [tmp_path / "one", tmp_path / "other-location"]
    for root in roots:
        (root / "src/pkg").mkdir(parents=True)
        (root / "src/pkg/module.py").write_text("VALUE = 1\n")
        (root / "constraints-host.txt").write_text("playwright==1.58.0\n")

    first = compute_build_provenance(roots[0], commit="abc")
    second = compute_build_provenance(roots[1], commit="abc")

    assert first == second
    assert first["org.asil.git-commit"] == "abc"
    assert len(first["org.asil.source-sha256"]) == 64

    (roots[0] / "src/pkg/__pycache__").mkdir()
    (roots[0] / "src/pkg/__pycache__/module.cpython-311.pyc").write_bytes(b"cache")
    assert compute_build_provenance(roots[0], commit="abc") == first


def test_runtime_probe_command_uses_existing_image_without_dependencies(tmp_path: Path) -> None:
    command = build_runtime_probe_command(
        root=tmp_path,
        env_file=tmp_path / ".env",
        project_name="asil-bootstrap",
    )

    assert command[:2] == ["docker", "compose"]
    assert command[-8:-4] == ["run", "--rm", "--no-deps", "eval"]
    assert command[-4:] == [
        "python",
        "scripts/verify_rebuttal_docker.py",
        "--container-runtime-probe",
        "--root=/app",
    ]


def test_smoke_command_is_deterministic_and_never_requests_a_model(tmp_path: Path) -> None:
    command = build_smoke_command(root=tmp_path, project_name="asil-bootstrap-smoke")
    joined = " ".join(command)

    assert "run_evaluation_managed.py" in joined
    assert "--task-set test_rebuttal_docker_smoke.json" in joined
    assert "--participant asil" in joined
    assert "--asil-execution deterministic" in joined
    assert "--provider mock" in joined
    assert "--model" not in command
    assert "--force-rerun" in command


def test_compose_ps_parser_accepts_healthy_services_and_completed_initializer() -> None:
    payload = "\n".join(
        [
            json.dumps({"Service": "gitea", "State": "running", "Health": "healthy"}),
            json.dumps({"Service": "gitea-init", "State": "exited", "ExitCode": 0}),
            json.dumps({"Service": "obs-mock", "State": "running", "Health": "healthy"}),
        ]
    )

    states = parse_compose_ps(payload)

    assert states == {
        "gitea": "healthy",
        "gitea-init": "completed",
        "obs-mock": "healthy",
    }


def test_smoke_result_gate_requires_all_frozen_tasks_and_artifacts(tmp_path: Path) -> None:
    result_root = tmp_path / "run"
    for software, task_id in (("drawio", "drawio_04"), ("blender", "blender_05")):
        task_dir = result_root / "semantic/structured/asil-deterministic" / software / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "result.txt").write_text("1.0\n")
        (task_dir / "evaluation.json").write_text('{"score": 1.0}\n')
        (task_dir / "traj.jsonl").write_text("{}\n")

    passed = validate_smoke_results(result_root, ("drawio_04", "blender_05"))
    failed = validate_smoke_results(result_root, ("drawio_04", "gitea_11"))

    assert passed.status == "pass"
    assert failed.status == "fail"
    assert failed.evidence["missing_task_ids"] == ["gitea_11"]


def test_frozen_docker_smoke_covers_all_access_families_and_multi_app() -> None:
    path = ROOT / "evaluation_examples/test_rebuttal_docker_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "drawio": ["drawio_04"],
        "blender": ["blender_05"],
        "gitea": ["gitea_11"],
        "multi_apps": ["multi_apps_001"],
    }


def test_runtime_probe_accepts_complete_capability_payload() -> None:
    result = validate_runtime_probe(
        {
            "adapter_count": 15,
            "schema_count": 15,
            "chromium_launch": True,
            "xvfb": True,
            "openbox": True,
            "gitea_token": True,
            "gitea_api": True,
            "task_fixture_count": 2028,
            "desktop_tool_count": 11,
            "mounted_workspace_count": 3,
        }
    )

    assert result.status == "pass"


def test_runtime_probe_uses_the_real_libreoffice_adapter_class_name() -> None:
    assert verifier.runtime_adapter_class_name("libreoffice") == "LibreOfficeAdapter"
    assert verifier.runtime_adapter_class_name("libreoffice_writer") == "LibreOfficeWriterAdapter"
    assert verifier.runtime_adapter_class_name("libreoffice_impress") == "LibreOfficeImpressAdapter"
    assert verifier.runtime_adapter_class_name("jupyterlab") == "JupyterLabAdapter"
    assert verifier.runtime_adapter_class_name("obs") == "OBSAdapter"


def test_runtime_probe_rejects_missing_desktop_tools_or_workspace_mounts() -> None:
    result = validate_runtime_probe(
        {
            "adapter_count": 15,
            "schema_count": 15,
            "chromium_launch": True,
            "xvfb": True,
            "openbox": True,
            "gitea_token": True,
            "gitea_api": True,
            "task_fixture_count": 2028,
            "desktop_tool_count": 10,
            "mounted_workspace_count": 2,
        }
    )

    assert result.status == "fail"
    assert set(result.evidence["mismatches"]) >= {
        "desktop_tool_count",
        "mounted_workspace_count",
    }


def test_docker_verification_sequences_services_runtime_smoke_and_cleanup(tmp_path: Path) -> None:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/docker-compose.yml").write_text("services: {}\n")
    (tmp_path / ".env").write_text("")
    smoke_root = tmp_path / "results/setup/docker_smoke"
    commands: list[list[str]] = []
    compose_config = {
        "services": {
            "eval": {"image": "asil-eval:local", "build": {"context": ".."}},
            "obs-mock": {"image": "asil-obs-mock:local", "build": {"context": "."}},
            "gitea": {"image": "gitea/gitea@sha256:" + "a" * 64},
        }
    }
    image_payload = [
        {
            "Id": "sha256:eval",
            "RepoTags": ["asil-eval:local"],
            "Architecture": "amd64",
            "Config": {"Labels": {"org.asil.git-commit": "abc"}},
        },
        {
            "Id": "sha256:obs",
            "RepoTags": ["asil-obs-mock:local"],
            "Architecture": "amd64",
            "Config": {"Labels": {"org.asil.git-commit": "abc"}},
        },
    ]
    ps_payload = "\n".join(
        json.dumps({"Service": service, "State": "running", "Health": "healthy"})
        for service in ("gitea", "obs-mock", "code-server", "jupyterlab", "drawio")
    )
    runtime_payload = {
        "adapter_count": 15,
        "schema_count": 15,
        "chromium_launch": True,
        "xvfb": True,
        "openbox": True,
        "gitea_token": True,
        "gitea_api": True,
        "task_fixture_count": 2028,
        "desktop_tool_count": 11,
        "mounted_workspace_count": 3,
    }

    def runner(command: list[str], **_: object) -> CommandResult:
        commands.append(command)
        joined = " ".join(command)
        if "config --format json" in joined:
            return CommandResult(0, json.dumps(compose_config), "")
        if command[:3] == ["docker", "image", "inspect"]:
            return CommandResult(0, json.dumps(image_payload), "")
        if " ps --all --format json" in joined:
            down_seen = any(" down --remove-orphans" in " ".join(item) for item in commands)
            return CommandResult(0, "" if down_seen else ps_payload, "")
        if "--container-runtime-probe" in command:
            return CommandResult(0, json.dumps(runtime_payload), "")
        if "run_evaluation_managed.py" in joined:
            for software, task_id in (
                ("drawio", "drawio_04"),
                ("blender", "blender_05"),
                ("gitea", "gitea_11"),
                ("multi_apps", "multi_apps_001"),
            ):
                task_dir = smoke_root / "semantic/structured/asil-deterministic" / software / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "result.txt").write_text("1.0\n")
                (task_dir / "evaluation.json").write_text('{"score": 1.0}\n')
                (task_dir / "traj.jsonl").write_text("{}\n")
            return CommandResult(0, "", "")
        return CommandResult(0, "", "")

    results = run_docker_verification(
        root=tmp_path,
        project_name="asil-bootstrap",
        expected_labels={"org.asil.git-commit": "abc"},
        runner=runner,
        include_readiness=False,
    )

    statuses = {result.gate_id: result.status for result in results}
    assert statuses["repository.compose"] == "pass"
    assert statuses["images.provenance"] == "pass"
    assert statuses["services.health"] == "pass"
    assert statuses["runtime.capabilities"] == "pass"
    assert statuses["smoke.deterministic"] == "pass"
    assert statuses["cleanup.resources"] == "pass"
    joined_commands = "\n".join(" ".join(command) for command in commands)
    assert " up -d --wait" in joined_commands
    assert " down --remove-orphans" in joined_commands


def test_task_fixture_counter_counts_json_files(tmp_path: Path) -> None:
    fixtures = tmp_path / "evaluation_examples/examples/demo"
    fixtures.mkdir(parents=True)
    (fixtures / "one.json").write_text("{}")
    (fixtures / "two.json").write_text("{}")
    (fixtures / "ignore.txt").write_text("x")

    assert count_task_fixtures(tmp_path) == 2


def test_main_all_combines_host_repository_and_docker_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(verifier, "_git_metadata", lambda root: ("abc", False))
    monkeypatch.setattr(verifier, "collect_host_facts", lambda root: _host())
    monkeypatch.setattr(
        verifier,
        "validate_required_repository_files",
        lambda root: [GateResult("repository.files", "pass", {})],
    )
    monkeypatch.setattr(
        verifier,
        "run_docker_verification",
        lambda **kwargs: [GateResult("smoke.deterministic", "pass", {})],
    )
    monkeypatch.setattr(
        verifier,
        "compute_build_provenance",
        lambda root, commit: {"org.asil.git-commit": commit},
    )

    exit_code = verifier.main(
        ["--phase", "all", "--root", str(tmp_path), "--report", str(report_path)]
    )
    payload = json.loads(report_path.read_text())

    assert exit_code == 0
    assert payload["ready"] is True
    assert {gate["gate_id"] for gate in payload["gates"]} >= {
        "host.platform",
        "repository.files",
        "smoke.deterministic",
    }


def test_main_preflight_combines_host_and_repository_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(verifier, "_git_metadata", lambda root: ("abc", False))
    monkeypatch.setattr(verifier, "collect_host_facts", lambda root: _host())
    monkeypatch.setattr(
        verifier,
        "validate_required_repository_files",
        lambda root: [GateResult("repository.files", "pass", {})],
    )

    exit_code = verifier.main(
        ["--phase", "preflight", "--root", str(tmp_path), "--report", str(report_path)]
    )
    payload = json.loads(report_path.read_text())

    assert exit_code == 0
    assert payload["ready"] is True
    assert {gate["gate_id"] for gate in payload["gates"]} >= {
        "host.platform",
        "repository.files",
    }


def test_wait_for_file_returns_immediately_when_present(tmp_path: Path) -> None:
    token = tmp_path / "gitea_token.txt"
    token.write_text("deadbeef", encoding="utf-8")
    assert verifier._wait_for_file(token, 5.0) is True


def test_wait_for_file_times_out_when_absent(tmp_path: Path) -> None:
    token = tmp_path / "never_written.txt"
    assert verifier._wait_for_file(token, 0.0) is False


def test_wait_for_file_tolerates_gitea_init_write_race(tmp_path: Path) -> None:
    # Reproduces the gitea-init race: ``docker compose up --wait`` returns while the
    # one-shot init container is still running, so the token file appears slightly
    # after the runtime probe starts polling. The probe must observe it, not sample
    # a single instant and fail closed.
    import threading

    token = tmp_path / "gitea_token.txt"
    writer = threading.Timer(0.4, lambda: token.write_text("late-token", encoding="utf-8"))
    writer.start()
    try:
        assert verifier._wait_for_file(token, 5.0) is True
    finally:
        writer.cancel()
