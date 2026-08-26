#!/usr/bin/env python3
"""Fail-closed readiness verifier for the public ASIL Docker environment."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
REQUIRED_NETWORK_ENDPOINTS = (
    "registry-1.docker.io",
    "archive.ubuntu.com",
    "pypi.org",
    "playwright.azureedge.net",
)
REQUIRED_REPOSITORY_FILES = (
    ".env.example",
    "constraints-host.txt",
    "docker/Dockerfile.eval",
    "docker/Dockerfile.obs-mock",
    "docker/docker-compose.yml",
    "evaluation_examples/test_full15.json",
    "evaluation_examples/test_full15_multi_apps_380.json",
    "evaluation_examples/test_full15_realwork_hard.json",
    "evaluation_examples/test_easy60.json",
    "evaluation_examples/test_rebuttal_docker_smoke.json",
    "pyproject.toml",
    "scripts/run_evaluation_managed.py",
    "scripts/bootstrap_rebuttal_docker.sh",
    "scripts/verify_rebuttal_docker.py",
)
SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: Literal["pass", "fail", "warn", "skip"]
    evidence: dict[str, object]
    remediation: str = ""
    mandatory: bool = True


@dataclass(frozen=True)
class HostFacts:
    os_id: str
    os_version: str
    architecture: str
    docker_server_version: str
    compose_version: str
    docker_daemon_ready: bool
    disk_free_bytes: int
    memory_total_bytes: int
    network_endpoints: Mapping[str, bool]


@dataclass
class BootstrapReport:
    schema_version: str
    started_at: str
    finished_at: str
    commit: str
    dirty: bool
    gates: list[GateResult] = field(default_factory=list)
    ready: bool = False


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", value)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def _wait_for_file(path: Path, timeout_s: float) -> bool:
    """Poll for a file to exist for up to ``timeout_s`` seconds.

    ``docker compose up --wait`` reports the one-shot ``gitea-init`` container as
    ready as soon as it starts, because it has no healthcheck. The token that
    container writes at the very end of its REST setup can therefore appear a few
    seconds after the ``--no-deps`` runtime capability probe launches. Sampling the
    token path a single instant is a race; poll briefly so the probe observes the
    token the init container is guaranteed to write, without weakening any
    benchmark task, evaluator, or scoring behavior.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if path.is_file():
            return True
        if time.monotonic() >= deadline:
            return path.is_file()
        time.sleep(1.0)


def evaluate_host_facts(facts: HostFacts) -> list[GateResult]:
    supported_platform = (
        facts.os_id.lower() == "ubuntu"
        and facts.os_version in {"22.04", "24.04"}
        and facts.architecture in {"x86_64", "amd64"}
    )
    docker_ok = facts.docker_daemon_ready and _version_tuple(facts.docker_server_version) >= (24,)
    compose_ok = _version_tuple(facts.compose_version) >= (2,)
    disk_ok = facts.disk_free_bytes >= 30 * GIB
    memory_ok = facts.memory_total_bytes >= 16 * GIB
    missing_network = sorted(
        endpoint for endpoint, reachable in facts.network_endpoints.items() if not reachable
    )
    return [
        GateResult(
            "host.platform",
            "pass" if supported_platform else "fail",
            {
                "os_id": facts.os_id,
                "os_version": facts.os_version,
                "architecture": facts.architecture,
            },
            "Use x86_64 Ubuntu 22.04 or 24.04.",
        ),
        GateResult(
            "host.docker",
            "pass" if docker_ok else "fail",
            {
                "server_version": facts.docker_server_version,
                "daemon_ready": facts.docker_daemon_ready,
            },
            "Install/start Docker Engine 24 or newer and grant the current user daemon access.",
        ),
        GateResult(
            "host.compose",
            "pass" if compose_ok else "fail",
            {"version": facts.compose_version},
            "Install Docker Compose v2.",
        ),
        GateResult(
            "host.disk",
            "pass" if disk_ok else "fail",
            {"free_bytes": facts.disk_free_bytes, "required_bytes": 30 * GIB},
            "Free at least 30 GiB on the filesystem containing this checkout.",
        ),
        GateResult(
            "host.memory",
            "pass" if memory_ok else "warn",
            {"total_bytes": facts.memory_total_bytes, "recommended_bytes": 16 * GIB},
            "Use a host with at least 16 GiB RAM for parallel GUI experiments.",
            mandatory=False,
        ),
        GateResult(
            "host.network",
            "pass" if not missing_network else "fail",
            {
                "checked": sorted(facts.network_endpoints),
                "unreachable": missing_network,
            },
            "Restore DNS/HTTPS access to every listed public distribution endpoint.",
        ),
    ]


def validate_required_repository_files(root: Path) -> list[GateResult]:
    missing = [relative for relative in REQUIRED_REPOSITORY_FILES if not (root / relative).is_file()]
    broken_links = [
        str(path.relative_to(root))
        for path in (root / "evaluation_examples").rglob("*")
        if path.is_symlink() and not path.exists()
    ] if (root / "evaluation_examples").exists() else []
    return [
        GateResult(
            "repository.files",
            "pass" if not missing else "fail",
            {"required_count": len(REQUIRED_REPOSITORY_FILES), "missing": missing},
            "Restore every required tracked bootstrap and public benchmark input from Git.",
        ),
        GateResult(
            "repository.symlinks",
            "pass" if not broken_links else "fail",
            {"broken": broken_links},
            "Replace broken task fixture symlinks with tracked files.",
        ),
    ]


def validate_compose_config(config: dict[str, object]) -> GateResult:
    services = config.get("services", {})
    if not isinstance(services, dict):
        services = {}
    mutable: list[str] = []
    missing_builds: list[str] = []
    for name, raw in services.items():
        service = raw if isinstance(raw, dict) else {}
        image = str(service.get("image", ""))
        if name in {"eval", "obs-mock"}:
            if not service.get("build"):
                missing_builds.append(str(name))
            continue
        if not image or "@sha256:" not in image or image.endswith(":latest"):
            mutable.append(str(name))
    ok = bool(services) and not mutable and not missing_builds
    return GateResult(
        "repository.compose",
        "pass" if ok else "fail",
        {
            "service_count": len(services),
            "mutable_services": sorted(mutable),
            "local_services_missing_build": sorted(missing_builds),
        },
        "Pin third-party images by digest and retain local eval/OBS build definitions.",
    )


def validate_image_inspection(
    images: dict[str, dict[str, object]],
    expected_labels: dict[str, str],
    *,
    required_images: Sequence[str] = ("asil-eval:local", "asil-obs-mock:local"),
) -> GateResult:
    missing: list[str] = []
    wrong_architecture: list[str] = []
    mismatches: list[dict[str, str]] = []
    image_ids: dict[str, str] = {}
    for name in required_images:
        image = images.get(name)
        if not image:
            missing.append(name)
            continue
        image_ids[name] = str(image.get("Id", ""))
        if str(image.get("Architecture", "")) not in {"amd64", "x86_64"}:
            wrong_architecture.append(name)
        config = image.get("Config", {})
        labels = config.get("Labels", {}) if isinstance(config, dict) else {}
        labels = labels if isinstance(labels, dict) else {}
        for key, expected in expected_labels.items():
            actual = str(labels.get(key, ""))
            if actual != expected:
                mismatches.append(
                    {"image": name, "label": key, "expected": expected, "actual": actual}
                )
    ok = not missing and not wrong_architecture and not mismatches
    return GateResult(
        "images.provenance",
        "pass" if ok else "fail",
        {
            "image_ids": image_ids,
            "missing": missing,
            "wrong_architecture": wrong_architecture,
            "label_mismatches": mismatches,
        },
        "Rebuild local images from this checkout using the bootstrap command.",
    )


def validate_service_state(
    state: dict[str, str],
    *,
    required: Sequence[str] = ("gitea", "obs-mock", "code-server", "jupyterlab", "drawio"),
) -> GateResult:
    unhealthy = {name: state.get(name, "missing") for name in required if state.get(name) != "healthy"}
    return GateResult(
        "services.health",
        "pass" if not unhealthy else "fail",
        {"required": list(required), "states": state, "unhealthy": unhealthy},
        "Inspect dedicated Compose service logs and restore all health checks.",
    )


def validate_runtime_probe(payload: dict[str, object]) -> GateResult:
    expected = {
        "adapter_count": 15,
        "schema_count": 15,
        "chromium_launch": True,
        "xvfb": True,
        "openbox": True,
        "gitea_token": True,
        "gitea_api": True,
        "desktop_tool_count": 11,
        "mounted_workspace_count": 3,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    try:
        fixture_count = int(payload.get("task_fixture_count", 0))
    except (TypeError, ValueError):
        fixture_count = 0
    if fixture_count <= 0:
        mismatches["task_fixture_count"] = {"expected": ">0", "actual": fixture_count}
    return GateResult(
        "runtime.capabilities",
        "pass" if not mismatches else "fail",
        {"mismatches": mismatches, "probe": payload},
        "Rebuild the eval image and inspect the failed runtime capability.",
    )


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path, relative: str) -> str:
    base = root / relative
    digest = hashlib.sha256()
    if not base.is_dir():
        return "missing"
    for path in sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ):
        relative_path = path.relative_to(root).as_posix()
        digest.update(f"{_sha256_file(path)}  {relative_path}\n".encode("utf-8"))
    return digest.hexdigest()


def compute_build_provenance(root: Path, *, commit: str) -> dict[str, str]:
    constraints = root / "constraints-host.txt"
    return {
        "org.asil.git-commit": commit,
        "org.asil.constraints-sha256": _sha256_file(constraints) if constraints.is_file() else "missing",
        "org.asil.source-sha256": _sha256_tree(root, "src"),
    }


def _compose_prefix(*, root: Path, env_file: Path, project_name: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(root / "docker/docker-compose.yml"),
        "-p",
        project_name,
    ]


def build_runtime_probe_command(
    *, root: Path, env_file: Path, project_name: str
) -> list[str]:
    return [
        *_compose_prefix(root=root, env_file=env_file, project_name=project_name),
        "--profile",
        "eval",
        "run",
        "--rm",
        "--no-deps",
        "eval",
        "python",
        "scripts/verify_rebuttal_docker.py",
        "--container-runtime-probe",
        "--root=/app",
    ]


def build_smoke_command(*, root: Path, project_name: str) -> list[str]:
    output_dir = root / "results/setup/docker_smoke"
    return [
        sys.executable,
        str(root / "scripts/run_evaluation_managed.py"),
        "--compose-project-name",
        project_name,
        "--num-envs",
        "1",
        "--task-set",
        "test_rebuttal_docker_smoke.json",
        "--participant",
        "asil",
        "--asil-execution",
        "deterministic",
        "--provider",
        "mock",
        "--max-steps",
        "15",
        "--force-rerun",
        "--output-dir",
        str(output_dir),
        "--output",
        str(output_dir / "results.json"),
    ]


def parse_compose_ps(payload: str) -> dict[str, str]:
    stripped = payload.strip()
    if not stripped:
        return {}
    parsed: list[dict[str, object]] = []
    try:
        value = json.loads(stripped)
        if isinstance(value, list):
            parsed = [row for row in value if isinstance(row, dict)]
        elif isinstance(value, dict):
            parsed = [value]
    except json.JSONDecodeError:
        for line in stripped.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                parsed.append(row)
    states: dict[str, str] = {}
    for row in parsed:
        service = str(row.get("Service", ""))
        if not service:
            continue
        state = str(row.get("State", "")).lower()
        health = str(row.get("Health", "")).lower()
        try:
            exit_code = int(row.get("ExitCode", -1))
        except (TypeError, ValueError):
            exit_code = -1
        if service == "gitea-init" and state == "exited" and exit_code == 0:
            states[service] = "completed"
        elif state == "running" and health == "healthy":
            states[service] = "healthy"
        elif state == "running" and not health:
            states[service] = "running"
        else:
            states[service] = health or state or "unknown"
    return states


def validate_smoke_results(result_root: Path, expected_task_ids: Sequence[str]) -> GateResult:
    task_dirs: dict[str, Path] = {
        path.parent.name: path.parent for path in result_root.rglob("result.txt")
    }
    missing = sorted(task_id for task_id in expected_task_ids if task_id not in task_dirs)
    invalid: list[dict[str, object]] = []
    for task_id in expected_task_ids:
        task_dir = task_dirs.get(task_id)
        if task_dir is None:
            continue
        required = ("result.txt", "evaluation.json", "traj.jsonl")
        missing_artifacts = [name for name in required if not (task_dir / name).is_file()]
        try:
            score = float((task_dir / "result.txt").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            score = -1.0
        if missing_artifacts or score != 1.0:
            invalid.append(
                {"task_id": task_id, "score": score, "missing_artifacts": missing_artifacts}
            )
    ok = not missing and not invalid
    return GateResult(
        "smoke.deterministic",
        "pass" if ok else "fail",
        {
            "expected_task_ids": list(expected_task_ids),
            "completed_task_ids": sorted(task_dirs),
            "missing_task_ids": missing,
            "invalid": invalid,
        },
        "Inspect deterministic smoke output and the dedicated managed-Docker logs.",
    )


def _execute_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 600.0,
) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(124, "", type(exc).__name__)
    return CommandResult(
        completed.returncode,
        completed.stdout[-1_000_000:],
        completed.stderr[-1_000_000:],
    )


def _command_failure(gate_id: str, command: Sequence[str], result: CommandResult) -> GateResult:
    return GateResult(
        gate_id,
        "fail",
        {
            "command": list(command),
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-2000:],
        },
        "Inspect results/setup/logs and rerun the bootstrap after fixing this command.",
    )


def _images_by_tag(payload: object) -> dict[str, dict[str, object]]:
    rows = payload if isinstance(payload, list) else []
    images: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for tag in row.get("RepoTags", []) or []:
            images[str(tag)] = row
    return images


def run_docker_verification(
    *,
    root: Path,
    project_name: str,
    expected_labels: dict[str, str],
    runner: Callable[..., CommandResult] = _execute_command,
    include_readiness: bool = True,
) -> list[GateResult]:
    env_file = root / ".env"
    prefix = _compose_prefix(root=root, env_file=env_file, project_name=project_name)
    gates: list[GateResult] = []

    config_command = [*prefix, "--profile", "eval", "config", "--format", "json"]
    config_result = runner(config_command, cwd=root, timeout=60.0)
    if config_result.returncode == 0:
        try:
            config_payload = json.loads(config_result.stdout)
            gates.append(validate_compose_config(config_payload))
        except json.JSONDecodeError:
            gates.append(_command_failure("repository.compose", config_command, config_result))
    else:
        gates.append(_command_failure("repository.compose", config_command, config_result))

    image_command = ["docker", "image", "inspect", "asil-eval:local", "asil-obs-mock:local"]
    image_result = runner(image_command, cwd=root, timeout=60.0)
    if image_result.returncode == 0:
        try:
            gates.append(
                validate_image_inspection(
                    _images_by_tag(json.loads(image_result.stdout)), expected_labels
                )
            )
        except json.JSONDecodeError:
            gates.append(_command_failure("images.provenance", image_command, image_result))
    else:
        gates.append(_command_failure("images.provenance", image_command, image_result))

    up_command = [
        *prefix,
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "240",
        "gitea",
        "gitea-init",
        "obs-mock",
        "code-server",
        "jupyterlab",
        "drawio",
    ]
    down_command = [*prefix, "down", "--remove-orphans"]
    try:
        up_result = runner(up_command, cwd=root, timeout=300.0)
        if up_result.returncode != 0:
            gates.append(_command_failure("services.start", up_command, up_result))

        ps_command = [*prefix, "ps", "--all", "--format", "json"]
        ps_result = runner(ps_command, cwd=root, timeout=60.0)
        states = parse_compose_ps(ps_result.stdout) if ps_result.returncode == 0 else {}
        gates.append(validate_service_state(states))

        runtime_command = build_runtime_probe_command(
            root=root, env_file=env_file, project_name=project_name
        )
        runtime_result = runner(runtime_command, cwd=root, timeout=180.0)
        if runtime_result.returncode == 0:
            try:
                gates.append(validate_runtime_probe(json.loads(runtime_result.stdout)))
            except json.JSONDecodeError:
                gates.append(_command_failure("runtime.capabilities", runtime_command, runtime_result))
        else:
            gates.append(_command_failure("runtime.capabilities", runtime_command, runtime_result))

        smoke_command = build_smoke_command(
            root=root, project_name=f"{project_name}-smoke"
        )
        smoke_env = dict(os.environ)
        smoke_env["ASIL_MANAGED_SKIP_BUILD"] = "1"
        smoke_result = runner(smoke_command, cwd=root, env=smoke_env, timeout=1200.0)
        if smoke_result.returncode != 0:
            gates.append(_command_failure("smoke.command", smoke_command, smoke_result))
        gates.append(
            validate_smoke_results(
                root / "results/setup/docker_smoke",
                ("drawio_04", "blender_05", "gitea_11", "multi_apps_001"),
            )
        )
        if include_readiness:
            gates.append(validate_public_readiness(root))
    finally:
        down_result = runner(down_command, cwd=root, timeout=180.0)
        if down_result.returncode != 0:
            gates.append(_command_failure("cleanup.down", down_command, down_result))
        ps_after_command = [*prefix, "ps", "--all", "--format", "json"]
        ps_after = runner(ps_after_command, cwd=root, timeout=60.0)
        remaining_services = sorted(parse_compose_ps(ps_after.stdout)) if ps_after.returncode == 0 else ["unknown"]
        network_command = [
            "docker",
            "network",
            "ls",
            "--filter",
            f"name=^{project_name}_",
            "--format",
            "{{.Name}}",
        ]
        networks = runner(network_command, cwd=root, timeout=60.0)
        remaining_networks = [
            line.strip() for line in networks.stdout.splitlines() if line.strip()
        ] if networks.returncode == 0 else ["unknown"]
        gates.append(
            validate_cleanup(
                {"containers": remaining_services, "networks": remaining_networks}
            )
        )
    return gates


def runtime_adapter_class_name(software: str) -> str:
    explicit_names = {
        "jupyterlab": "JupyterLabAdapter",
        "libreoffice": "LibreOfficeAdapter",
        "obs": "OBSAdapter",
    }
    if software in explicit_names:
        return explicit_names[software]
    if software.startswith("libreoffice_"):
        suffix = software.removeprefix("libreoffice_")
        return "LibreOffice" + "".join(
            part.capitalize() for part in suffix.split("_")
        ) + "Adapter"
    return "".join(part.capitalize() for part in software.split("_")) + "Adapter"


def container_runtime_probe(root: Path) -> dict[str, object]:
    sys.path.insert(0, str(root / "src"))

    task_index = _load_json(root / "evaluation_examples/test_full15.json")
    software = list(task_index) if isinstance(task_index, dict) else []
    imported = 0
    for name in software:
        class_name = runtime_adapter_class_name(name)
        module = __import__(f"asil.adapters.{name}", fromlist=[class_name])
        getattr(module, class_name)
        imported += 1
    schema_count = sum(
        (root / f"src/asil/action_schemas/{name}.json").is_file() for name in software
    )

    chromium_launch = False
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        browser.close()
        chromium_launch = True

    display = ":97"
    env = dict(os.environ)
    env["DISPLAY"] = display
    xvfb_process = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x720x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    openbox_process: subprocess.Popen[bytes] | None = None
    try:
        time.sleep(0.5)
        xvfb_ok = xvfb_process.poll() is None
        openbox_process = subprocess.Popen(
            ["openbox"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        openbox_ok = openbox_process.poll() is None
    finally:
        if openbox_process is not None:
            openbox_process.terminate()
            try:
                openbox_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                openbox_process.kill()
        xvfb_process.terminate()
        try:
            xvfb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb_process.kill()

    gitea_token = _wait_for_file(
        Path(os.environ.get("GITEA_TOKEN_FILE", "/shared/gitea_token.txt")),
        float(os.environ.get("ASIL_RUNTIME_PROBE_TOKEN_TIMEOUT_S", "60")),
    )
    gitea_api = False
    try:
        with urllib.request.urlopen(
            os.environ.get("GITEA_URL", "http://gitea:3000") + "/api/v1/version",
            timeout=10,
        ) as response:
            gitea_api = response.status == 200
    except OSError:
        pass
    fixture_count = count_task_fixtures(root)
    desktop_commands = (
        "audacity",
        "blender",
        "celluloid",
        "gimp",
        "inkscape",
        "kdenlive",
        "libreoffice",
        "nautilus",
        "obs",
        "thunderbird",
        "vlc",
    )
    desktop_tools = {
        command: shutil.which(command) or "" for command in desktop_commands
    }
    workspace_mounts = {
        path: Path(path).is_dir()
        for path in ("/results", "/shared-workspaces", "/app/evaluation_examples")
    }
    return {
        "adapter_count": imported,
        "schema_count": schema_count,
        "chromium_launch": chromium_launch,
        "xvfb": xvfb_ok,
        "openbox": openbox_ok,
        "gitea_token": gitea_token,
        "gitea_api": gitea_api,
        "task_fixture_count": fixture_count,
        "desktop_tool_count": sum(bool(path) for path in desktop_tools.values()),
        "desktop_tools": desktop_tools,
        "mounted_workspace_count": sum(workspace_mounts.values()),
        "workspace_mounts": workspace_mounts,
    }


def count_task_fixtures(root: Path) -> int:
    return sum(1 for _ in (root / "evaluation_examples/examples").rglob("*.json"))


def validate_public_readiness(root: Path) -> GateResult:
    required_json = (
        "evaluation_examples/test_full15.json",
        "evaluation_examples/test_full15_multi_apps_380.json",
        "evaluation_examples/test_full15_realwork_hard.json",
        "evaluation_examples/test_easy60.json",
        "evaluation_examples/test_rebuttal_docker_smoke.json",
    )
    missing_files: list[str] = []
    invalid_json: list[str] = []
    for relative in required_json:
        path = root / relative
        if not path.is_file():
            missing_files.append(relative)
            continue
        try:
            _load_json(path)
        except (OSError, json.JSONDecodeError):
            invalid_json.append(relative)

    task_ids = {
        path.stem
        for path in (root / "evaluation_examples/examples").rglob("*.json")
    } if (root / "evaluation_examples/examples").exists() else set()
    missing_task_ids: list[str] = []
    for relative in required_json:
        selection_path = root / relative
        if not selection_path.is_file():
            continue
        try:
            selection = _load_json(selection_path)
            if isinstance(selection, dict):
                selected = [task_id for values in selection.values() if isinstance(values, list) for task_id in values]
                missing_task_ids.extend(
                    str(task_id) for task_id in selected if str(task_id) not in task_ids
                )
        except (OSError, json.JSONDecodeError):
            pass
    missing_task_ids = sorted(set(missing_task_ids))
    ok = not missing_files and not invalid_json and not missing_task_ids
    return GateResult(
        "release.inventory",
        "pass" if ok else "fail",
        {
            "required_json_count": len(required_json),
            "missing_files": missing_files,
            "invalid_json": invalid_json,
            "task_fixture_count": len(task_ids),
            "missing_task_ids": missing_task_ids,
        },
        "Restore the checked-in public benchmark indexes and referenced task fixtures.",
    )


def validate_cleanup(resources: dict[str, list[str]]) -> GateResult:
    remaining = {
        kind: names for kind, names in resources.items() if names
    }
    return GateResult(
        "cleanup.resources",
        "pass" if not remaining else "fail",
        {"remaining": remaining},
        "Run the dedicated bootstrap Compose down command and recheck the project prefix.",
    )


def finalize_report(report: BootstrapReport) -> BootstrapReport:
    report.finished_at = _utc_now()
    report.ready = all(
        result.status == "pass" for result in report.gates if result.mandatory
    )
    return report


def write_report(report: BootstrapReport, path: Path) -> None:
    payload = dataclasses.asdict(report)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ValueError("report contains a credential-shaped value")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_text(command: Sequence[str], *, timeout: float = 15.0) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return result.returncode == 0, result.stdout.strip()


def _os_release() -> tuple[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values.get("ID", ""), values.get("VERSION_ID", "")


def collect_host_facts(root: Path = PROJECT_ROOT) -> HostFacts:
    os_id, os_version = _os_release()
    docker_ok, docker_version = _run_text(
        ["docker", "version", "--format", "{{.Server.Version}}"]
    )
    _, compose_version = _run_text(["docker", "compose", "version", "--short"])
    memory_bytes = 0
    try:
        mem_kib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            )
        )
        memory_bytes = mem_kib * 1024
    except (OSError, StopIteration, ValueError):
        pass
    endpoints: dict[str, bool] = {}
    for endpoint in REQUIRED_NETWORK_ENDPOINTS:
        try:
            socket.getaddrinfo(endpoint, 443)
            endpoints[endpoint] = True
        except OSError:
            endpoints[endpoint] = False
    return HostFacts(
        os_id=os_id,
        os_version=os_version,
        architecture=platform.machine(),
        docker_server_version=docker_version,
        compose_version=compose_version,
        docker_daemon_ready=docker_ok,
        disk_free_bytes=shutil.disk_usage(root).free,
        memory_total_bytes=memory_bytes,
        network_endpoints=endpoints,
    )


def _git_metadata(root: Path) -> tuple[str, bool]:
    ok, commit = _run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    status_ok, status = _run_text(["git", "-C", str(root), "status", "--porcelain"])
    return (commit if ok else "unknown", bool(status) if status_ok else True)


def _read_optional_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("host", "repository", "preflight", "images", "services", "runtime", "readiness", "cleanup", "all"),
        default="all",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results/setup/docker_bootstrap_report.json",
    )
    parser.add_argument("--compose-config-json", type=Path)
    parser.add_argument("--images-json", type=Path)
    parser.add_argument("--services-json", type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--cleanup-json", type=Path)
    parser.add_argument("--project-name", default=os.environ.get("ASIL_BOOTSTRAP_PROJECT_NAME", "asil-bootstrap"))
    parser.add_argument("--container-runtime-probe", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.container_runtime_probe:
        print(json.dumps(container_runtime_probe(root), sort_keys=True))
        return 0

    commit, dirty = _git_metadata(root)
    report = BootstrapReport(
        schema_version="1.0",
        started_at=_utc_now(),
        finished_at="",
        commit=commit,
        dirty=dirty,
    )
    if args.phase == "all":
        try:
            report.gates.extend(evaluate_host_facts(collect_host_facts(root)))
            report.gates.extend(validate_required_repository_files(root))
            report.gates.extend(
                run_docker_verification(
                    root=root,
                    project_name=args.project_name,
                    expected_labels=compute_build_provenance(root, commit=commit),
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.gates.append(
                GateResult("verifier.exception", "fail", {"type": type(exc).__name__}, str(exc))
            )
        finalize_report(report)
        write_report(report, args.report)
        print(json.dumps({"ready": report.ready, "report": str(args.report), "gates": len(report.gates)}, sort_keys=True))
        return 0 if report.ready else 1

    phases = {"host", "repository"} if args.phase == "preflight" else {args.phase}
    try:
        if "host" in phases:
            report.gates.extend(evaluate_host_facts(collect_host_facts(root)))
        if "repository" in phases:
            report.gates.extend(validate_required_repository_files(root))
            if args.compose_config_json:
                report.gates.append(validate_compose_config(_read_optional_json(args.compose_config_json)))
        if "images" in phases:
            report.gates.append(validate_image_inspection(_read_optional_json(args.images_json), {}))
        if "services" in phases:
            services = _read_optional_json(args.services_json)
            report.gates.append(validate_service_state({str(k): str(v) for k, v in services.items()}))
        if "runtime" in phases:
            report.gates.append(validate_runtime_probe(_read_optional_json(args.runtime_json)))
        if "readiness" in phases:
            report.gates.append(validate_public_readiness(root))
        if "cleanup" in phases:
            cleanup = _read_optional_json(args.cleanup_json)
            normalized = {
                str(key): [str(value) for value in values]
                for key, values in cleanup.items()
                if isinstance(values, list)
            }
            report.gates.append(validate_cleanup(normalized))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.gates.append(
            GateResult("verifier.exception", "fail", {"type": type(exc).__name__}, str(exc))
        )
    finalize_report(report)
    write_report(report, args.report)
    print(json.dumps({"ready": report.ready, "report": str(args.report), "gates": len(report.gates)}, sort_keys=True))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
