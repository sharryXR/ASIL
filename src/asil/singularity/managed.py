#!/usr/bin/env python3
"""Managed Singularity orchestrator for scripts/run_benchmark.py.

This mirrors scripts/run_evaluation_managed.py while replacing Docker Compose
with one Singularity service stack per worker. Each worker gets its own ports
and bind-mounted runtime directories because Singularity uses host networking.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_ROOT))

import asil.benchmark as _benchmark  # noqa: E402
import run_evaluation_managed as _managed  # noqa: E402
from asil.result_store import TaskKey, select_pending_tasks  # noqa: E402


DEFAULT_SIF_DIR = PROJECT_ROOT / "singularity" / "images"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "results" / ".singularity-runtime"
DEFAULT_BASE_PORT = 31000
DEFAULT_PORT_STRIDE = 20
DEFAULT_BASE_DISPLAY = 90
SERVICE_OFFSETS = {
    "gitea": 0,
    "obs_mock": 1,
    "code_server": 2,
    "jupyterlab": 3,
    "drawio": 4,
}
SIF_NAMES = {
    "eval": "asil_eval.sif",
    "gitea": "gitea.sif",
    "obs_mock": "obs_mock.sif",
    "code_server": "code_server.sif",
    "jupyterlab": "jupyterlab.sif",
    "drawio": "drawio.sif",
}
PASSTHROUGH_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GENAI_API_KEY",
    "GEMINI_BASE_URL",
    "ASIL_GUI_LLM_TIMEOUT_S",
    "ASIL_GUI_LLM_CALL_TIMEOUT_S",
    "ASIL_GUI_INIT_WATCHDOG_S",
    "EVAL_OBS_REAL_GUI",
    "EVAL_OBS_WS_PROTOCOL",
}

_ACTIVE_STACKS: list["WorkerStack"] = []


@dataclass(frozen=True)
class PortPlan:
    worker_index: int
    base_port: int
    port_stride: int

    def port(self, service: str) -> int:
        return self.base_port + (self.worker_index - 1) * self.port_stride + SERVICE_OFFSETS[service]

    def as_dict(self) -> dict[str, int]:
        return {service: self.port(service) for service in SERVICE_OFFSETS}

    def reserved_ports(self) -> dict[str, int]:
        ports = self.as_dict()
        drawio_port = ports["drawio"]
        if drawio_port > 20_000:
            ports["drawio_shutdown"] = drawio_port - 20_000
            ports["drawio_ajp"] = drawio_port - 19_000
        else:
            ports["drawio_shutdown"] = drawio_port + 10_000
            ports["drawio_ajp"] = drawio_port + 11_000
        return ports


@dataclass(frozen=True)
class DisplayPlan:
    worker_index: int
    base_display: int

    @property
    def display_number(self) -> int:
        return self.base_display + self.worker_index - 1

    @property
    def display(self) -> str:
        return f":{self.display_number}"

    def as_dict(self) -> dict[str, str | int]:
        return {"worker_index": self.worker_index, "display": self.display}


@dataclass(frozen=True)
class WorkerRuntime:
    worker_index: int
    name: str
    root: Path
    logs: Path
    gitea_data: Path
    shared: Path
    shared_workspaces: Path
    tmp: Path

    @classmethod
    def create(cls, *, runtime_root: Path, run_slug: str, worker_index: int) -> "WorkerRuntime":
        name = f"{run_slug}-w{worker_index:02d}"
        root = runtime_root / name
        return cls(
            worker_index=worker_index,
            name=name,
            root=root,
            logs=root / "logs",
            gitea_data=root / "gitea-data",
            shared=root / "shared",
            shared_workspaces=root / "shared-workspaces",
            tmp=root / "tmp",
        )

    def prepare(self) -> None:
        for path in (self.logs, self.gitea_data, self.shared, self.shared_workspaces, self.tmp):
            path.mkdir(parents=True, exist_ok=True)


class WorkerStack:
    def __init__(
        self,
        *,
        singularity_bin: str,
        sif_dir: Path,
        runtime: WorkerRuntime,
        ports: PortPlan,
        display: DisplayPlan,
        base_env: dict[str, str],
        writable_tmpfs: bool,
        health_timeout: float,
    ) -> None:
        self.singularity_bin = singularity_bin
        self.sif_dir = sif_dir
        self.runtime = runtime
        self.ports = ports
        self.display = display
        self.base_env = base_env
        self.writable_tmpfs = writable_tmpfs
        self.health_timeout = health_timeout
        self.processes: list[subprocess.Popen[Any]] = []
        self._log_handles: list[Any] = []

    def start(self) -> None:
        self.runtime.prepare()
        _ACTIVE_STACKS.append(self)

        self._start_service(
            "gitea",
            image="gitea",
            command=["/opt/asil/start_gitea.sh"],
            binds=[(self.runtime.gitea_data, "/data", None), (self.runtime.tmp, "/tmp", None)],
            env={
                "HOST": "127.0.0.1",
                "PORT": str(self.ports.port("gitea")),
            },
        )
        _wait_http(
            f"http://127.0.0.1:{self.ports.port('gitea')}/api/v1/version",
            timeout_s=self.health_timeout,
            label=f"{self.runtime.name}:gitea",
        )
        _initialize_gitea(
            base_url=f"http://127.0.0.1:{self.ports.port('gitea')}",
            shared_dir=self.runtime.shared,
            timeout_s=self.health_timeout,
        )

        self._start_service(
            "obs_mock",
            image="obs_mock",
            command=["python", "/opt/asil/start_obs_mock.py"],
            binds=[(self.runtime.tmp, "/tmp", None)],
            env={"HOST": "127.0.0.1", "PORT": str(self.ports.port("obs_mock"))},
        )
        self._start_service(
            "code_server",
            image="code_server",
            command=["/opt/asil/start_code_server.sh"],
            binds=[(self.runtime.shared_workspaces, "/shared-workspaces", None), (self.runtime.tmp, "/tmp", None)],
            env={
                "HOST": "127.0.0.1",
                "PORT": str(self.ports.port("code_server")),
                "SHARED_WORKSPACES": "/shared-workspaces",
                "HOME": "/tmp/code-server-home",
            },
        )
        self._start_service(
            "jupyterlab",
            image="jupyterlab",
            command=["/opt/asil/start_jupyterlab.sh"],
            binds=[(self.runtime.shared_workspaces, "/shared-workspaces", None), (self.runtime.tmp, "/tmp", None)],
            env={
                "HOST": "127.0.0.1",
                "PORT": str(self.ports.port("jupyterlab")),
                "SHARED_WORKSPACES": "/shared-workspaces",
                "HOME": "/tmp/jupyter-home",
                "JUPYTER_RUNTIME_DIR": "/tmp/jupyter-runtime",
            },
        )
        self._start_service(
            "drawio",
            image="drawio",
            command=["/opt/asil/start_drawio.sh"],
            binds=[(self.runtime.tmp, "/tmp", None)],
            env={
                "PORT": str(self.ports.port("drawio")),
                "SHUTDOWN_PORT": str(self.ports.reserved_ports()["drawio_shutdown"]),
                "AJP_PORT": str(self.ports.reserved_ports()["drawio_ajp"]),
                "HOME": "/tmp/drawio-home",
            },
        )

        _wait_tcp("127.0.0.1", self.ports.port("obs_mock"), timeout_s=self.health_timeout, label=f"{self.runtime.name}:obs_mock")
        _wait_http(
            f"http://127.0.0.1:{self.ports.port('code_server')}/",
            timeout_s=self.health_timeout,
            label=f"{self.runtime.name}:code_server",
        )
        _wait_http(
            f"http://127.0.0.1:{self.ports.port('jupyterlab')}/lab",
            timeout_s=self.health_timeout,
            label=f"{self.runtime.name}:jupyterlab",
        )
        _wait_http(
            f"http://127.0.0.1:{self.ports.port('drawio')}/",
            timeout_s=self.health_timeout,
            label=f"{self.runtime.name}:drawio",
        )

    def start_eval(self, forwarded_args: list[str]) -> subprocess.Popen[Any]:
        container_env = self._eval_env()
        binds = [
            (PROJECT_ROOT / "results", "/results", None),
            (PROJECT_ROOT / "src", "/app/src", "ro"),
            (PROJECT_ROOT / "scripts", "/app/scripts", "ro"),
            (PROJECT_ROOT / "evaluation_examples", "/app/evaluation_examples", "ro"),
            (self.runtime.shared, "/shared", "ro"),
            (self.runtime.shared_workspaces, "/shared-workspaces", None),
            (self.runtime.tmp, "/tmp", None),
        ]
        command = _build_singularity_exec_command(
            singularity_bin=self.singularity_bin,
            image_path=self.sif_dir / SIF_NAMES["eval"],
            binds=binds,
            command=["python", "/app/scripts/run_benchmark.py", *forwarded_args],
            writable_tmpfs=self.writable_tmpfs,
        )
        log_handle = (self.runtime.logs / "eval.log").open("ab")
        self._log_handles.append(log_handle)
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=_singularity_process_env(self.base_env, container_env),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        return process

    def stop(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 15
        for process in reversed(self.processes):
            remaining = max(0.1, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in self._log_handles:
            try:
                handle.close()
            except Exception:
                pass
        try:
            _ACTIVE_STACKS.remove(self)
        except ValueError:
            pass

    def _start_service(
        self,
        service: str,
        *,
        image: str,
        command: list[str],
        binds: list[tuple[Path, str, str | None]],
        env: dict[str, str],
    ) -> None:
        log_handle = (self.runtime.logs / f"{service}.log").open("ab")
        self._log_handles.append(log_handle)
        full_command = _build_singularity_exec_command(
            singularity_bin=self.singularity_bin,
            image_path=self.sif_dir / SIF_NAMES[image],
            binds=binds,
            command=command,
            writable_tmpfs=self.writable_tmpfs,
        )
        process = subprocess.Popen(
            full_command,
            cwd=str(PROJECT_ROOT),
            env=_singularity_process_env(self.base_env, env),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self.processes.append(process)

    def _eval_env(self) -> dict[str, str]:
        env = _selected_passthrough_env(self.base_env)
        if "OPENAI_API_BASE" not in env and env.get("OPENAI_BASE_URL"):
            env["OPENAI_API_BASE"] = env["OPENAI_BASE_URL"]
        if "OPENAI_BASE_URL" not in env and env.get("OPENAI_API_BASE"):
            env["OPENAI_BASE_URL"] = env["OPENAI_API_BASE"]

        env.update(
            {
                "PYTHONPATH": "/app/src",
                "ASIL_SANDBOX": "true",
                "GITEA_URL": f"http://127.0.0.1:{self.ports.port('gitea')}",
                "GITEA_ADMIN": "asil_admin",
                "GITEA_PASSWORD": "asil_password",
                "GITEA_TOKEN_FILE": "/shared/gitea_token.txt",
                "GITEA_OWNER": "asil_admin",
                "GITEA_REPO": "test-repo",
                "OBS_WS_HOST": "127.0.0.1",
                "OBS_WS_PORT": str(self.ports.port("obs_mock")),
                "OBS_WS_PASSWORD": "",
                "OBS_WS_PROTOCOL": self.base_env.get("EVAL_OBS_WS_PROTOCOL", "auto"),
                "OBS_REAL_GUI": self.base_env.get("EVAL_OBS_REAL_GUI", "false"),
                "CODE_SERVER_URL": f"http://127.0.0.1:{self.ports.port('code_server')}",
                "CODE_SERVER_WORKSPACE_ROOT": "/shared-workspaces",
                "JUPYTERLAB_URL": f"http://127.0.0.1:{self.ports.port('jupyterlab')}",
                "JUPYTERLAB_WORKSPACE_ROOT": "/shared-workspaces",
                "DRAWIO_URL": f"http://127.0.0.1:{self.ports.port('drawio')}",
                "DISPLAY": self.display.display,
                "ASIL_XVFB_DISPLAY": self.display.display,
            }
        )
        return env


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _load_base_env(env_file: Path) -> dict[str, str]:
    env = _load_env_file(env_file)
    env.update({key: value for key, value in os.environ.items() if value is not None})
    if "OPENAI_API_BASE" not in env and env.get("OPENAI_BASE_URL"):
        env["OPENAI_API_BASE"] = env["OPENAI_BASE_URL"]
    if "OPENAI_BASE_URL" not in env and env.get("OPENAI_API_BASE"):
        env["OPENAI_BASE_URL"] = env["OPENAI_API_BASE"]
    return env


def _selected_passthrough_env(base_env: dict[str, str]) -> dict[str, str]:
    return {key: base_env[key] for key in PASSTHROUGH_ENV_KEYS if base_env.get(key)}


def _singularity_process_env(base_env: dict[str, str], container_env: dict[str, str]) -> dict[str, str]:
    process_env = dict(os.environ)
    for key, value in container_env.items():
        if value is None:
            continue
        process_env[f"SINGULARITYENV_{key}"] = str(value)
        process_env[f"APPTAINERENV_{key}"] = str(value)
    for key in ("PATH", "HOME", "TMPDIR"):
        if key in base_env and key not in process_env:
            process_env[key] = base_env[key]
    return process_env


def _build_singularity_exec_command(
    *,
    singularity_bin: str,
    image_path: Path,
    binds: list[tuple[Path, str, str | None]],
    command: list[str],
    writable_tmpfs: bool,
) -> list[str]:
    built = [singularity_bin, "exec", "--cleanenv"]
    if writable_tmpfs:
        built.append("--writable-tmpfs")
    for host_path, container_path, mode in binds:
        host_path = host_path.resolve()
        bind_spec = f"{host_path}:{container_path}"
        if mode:
            bind_spec = f"{bind_spec}:{mode}"
        built.extend(["--bind", bind_spec])
    built.append(str(image_path))
    built.extend(command)
    return built


def _validate_sifs(sif_dir: Path) -> None:
    missing = [name for name in SIF_NAMES.values() if not (sif_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing Singularity images in {sif_dir}: {missing_text}. "
            "Build them with `bash scripts/build_singularity_images.sh --sif-dir singularity/images`."
        )


def _resolve_singularity_bin(value: str | None) -> str:
    if value:
        return value
    for candidate in ("singularity", "apptainer"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("singularity/apptainer executable not found")


def _singularity_version(singularity_bin: str) -> str:
    try:
        result = subprocess.run(
            [singularity_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return (result.stdout or result.stderr).strip()


def _port_plan(worker_index: int, *, base_port: int, port_stride: int) -> PortPlan:
    if port_stride <= max(SERVICE_OFFSETS.values()):
        raise ValueError(
            f"port_stride={port_stride} is too small for service offsets {SERVICE_OFFSETS}; "
            f"use at least {max(SERVICE_OFFSETS.values()) + 1}."
        )
    return PortPlan(worker_index=worker_index, base_port=base_port, port_stride=port_stride)


def _display_plan(worker_index: int, *, base_display: int) -> DisplayPlan:
    if base_display < 1:
        raise ValueError(f"base_display={base_display} must be positive.")
    return DisplayPlan(worker_index=worker_index, base_display=base_display)


def _check_ports_available(plans: list[PortPlan]) -> None:
    seen: dict[int, str] = {}
    for plan in plans:
        for service, port in plan.reserved_ports().items():
            label = f"w{plan.worker_index:02d}:{service}"
            if port in seen:
                raise RuntimeError(f"Port allocation collision: {port} used by {seen[port]} and {label}")
            seen[port] = label
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise RuntimeError(f"Port {port} for {label} is not available: {exc}") from exc


def _urlopen_no_proxy(url: str, *, timeout: float):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url, timeout=timeout)


def _wait_http(url: str, *, timeout_s: float, label: str) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with _urlopen_no_proxy(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {label} at {url}: {last_error}")


def _wait_tcp(host: str, port: int, *, timeout_s: float, label: str) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {label} at {host}:{port}: {last_error}")


def _basic_auth_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _http_json(
    method: str,
    url: str,
    *,
    auth: tuple[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if auth is not None:
        headers["Authorization"] = _basic_auth_header(*auth)
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"body": body}
        return exc.code, parsed


def _http_form(method: str, url: str, *, payload: dict[str, str], timeout: float = 10) -> int:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _initialize_gitea(*, base_url: str, shared_dir: Path, timeout_s: float) -> None:
    admin = "asil_admin"
    password = "asil_password"
    repo = "test-repo"
    auth = (admin, password)
    api = f"{base_url.rstrip('/')}/api/v1"

    _wait_http(f"{api}/version", timeout_s=timeout_s, label="gitea api")

    _http_json(
        "POST",
        f"{api}/admin/users",
        payload={
            "username": admin,
            "password": password,
            "email": "admin@asil.local",
            "must_change_password": False,
            "send_notify": False,
        },
    )
    _http_form(
        "POST",
        f"{base_url.rstrip('/')}/user/sign_up",
        payload={
            "user_name": admin,
            "password": password,
            "retype": password,
            "email": "admin@asil.local",
        },
    )

    _http_json("DELETE", f"{api}/users/{admin}/tokens/eval-token", auth=auth)
    token_status, token_payload = _http_json(
        "POST",
        f"{api}/users/{admin}/tokens",
        auth=auth,
        payload={"name": "eval-token", "scopes": ["all"]},
    )
    token = str(token_payload.get("sha1", "")).strip()
    if token_status not in {200, 201} or not token:
        raise RuntimeError(f"Gitea token initialization failed at {base_url} (HTTP {token_status})")

    _http_json(
        "POST",
        f"{api}/user/repos",
        auth=auth,
        payload={
            "name": repo,
            "description": "ASIL evaluation test repository",
            "private": False,
            "auto_init": True,
            "default_branch": "main",
        },
    )
    time.sleep(2)
    _http_json(
        "POST",
        f"{api}/repos/{admin}/{repo}/branches",
        auth=auth,
        payload={"new_branch_name": "feature/login-fix", "old_branch_name": "main"},
    )
    _, readme_payload = _http_json(
        "GET",
        f"{api}/repos/{admin}/{repo}/contents/README.md?ref=feature/login-fix",
        auth=auth,
    )
    readme_sha = str(readme_payload.get("sha", "")).strip()
    if readme_sha:
        content = base64.b64encode(b"Fix mobile login bug\n").decode("ascii")
        _http_json(
            "PUT",
            f"{api}/repos/{admin}/{repo}/contents/README.md",
            auth=auth,
            payload={
                "message": "fix: resolve mobile login issue",
                "content": content,
                "branch": "feature/login-fix",
                "sha": readme_sha,
            },
        )
    _http_json(
        "POST",
        f"{api}/user/repos",
        auth=auth,
        payload={
            "name": "old-project",
            "description": "To be deleted",
            "private": False,
            "auto_init": True,
            "default_branch": "main",
        },
    )
    _http_json(
        "POST",
        f"{api}/admin/users",
        auth=auth,
        payload={
            "email": "dev@asil.local",
            "full_name": "Dev User",
            "login_name": "dev_user",
            "password": "dev_password",
            "send_notify": False,
            "source_id": 0,
            "username": "dev_user",
            "must_change_password": False,
        },
    )

    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "gitea_token.txt").write_text(token + "\n", encoding="utf-8")


def _prepare_forwarded_args_for_singularity_worker(
    *,
    forwarded_args: list[str],
    shard_task_set: str,
    worker_output: str,
) -> list[str]:
    return _managed._prepare_forwarded_args_for_worker(
        forwarded_args=forwarded_args,
        shard_task_set=shard_task_set,
        worker_output=worker_output,
    )


def _rewrite_forwarded_result_paths(forwarded_args: list[str]) -> list[str]:
    return _managed._rewrite_forwarded_result_paths(forwarded_args, project_root=PROJECT_ROOT)


def _safe_remove_runtime(runtime: WorkerRuntime, *, runtime_root: Path) -> None:
    try:
        runtime.root.resolve().relative_to(runtime_root.resolve())
    except ValueError:
        return
    shutil.rmtree(runtime.root, ignore_errors=True)


def _write_singularity_metadata(
    *,
    config,
    sif_dir: Path,
    singularity_bin: str,
    worker_ports: list[PortPlan],
    worker_displays: list[DisplayPlan],
    runtime_root: Path,
) -> None:
    result_root = _benchmark._result_root_for_config(config)
    summary_dir = result_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sif_dir / "manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = None
    payload = {
        "backend": "singularity",
        "num_envs": config.num_envs,
        "singularity_bin": singularity_bin,
        "singularity_version": _singularity_version(singularity_bin),
        "sif_dir": str(sif_dir),
        "sif_manifest": manifest,
        "runtime_root": str(runtime_root),
        "worker_ports": [
            {"worker_index": plan.worker_index, "ports": plan.as_dict()} for plan in worker_ports
        ],
        "worker_displays": [plan.as_dict() for plan in worker_displays],
    }
    (summary_dir / "singularity_backend.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        for stack in list(_ACTIVE_STACKS):
            stack.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _build_config_with_num_envs(config, num_envs: int):
    return _benchmark.BenchmarkConfig(
        software=config.software,
        task_index=config.task_index,
        output_dir=config.output_dir,
        output_json=config.output_json,
        participant=config.participant,
        run_mode=config.run_mode,
        comparison_participants=config.comparison_participants,
        asil_execution=config.asil_execution,
        execution_mode=config.execution_mode,
        provider=config.provider,
        model=config.model,
        max_steps=config.max_steps,
        test_config_base_dir=config.test_config_base_dir,
        osworld_format=config.osworld_format,
        docker=config.docker,
        managed_docker=True,
        mock=config.mock,
        dry_run=config.dry_run,
        resume=config.resume,
        force_rerun=config.force_rerun,
        rerun_failed_only=config.rerun_failed_only,
        num_envs=num_envs,
    )


def _run_one_worker(
    *,
    singularity_bin: str,
    sif_dir: Path,
    runtime_root: Path,
    run_slug: str,
    worker_index: int,
    ports: PortPlan,
    display: DisplayPlan,
    base_env: dict[str, str],
    forwarded_args: list[str],
    writable_tmpfs: bool,
    health_timeout: float,
    keep_runtime: bool,
) -> int:
    runtime = WorkerRuntime.create(runtime_root=runtime_root, run_slug=run_slug, worker_index=worker_index)
    stack = WorkerStack(
        singularity_bin=singularity_bin,
        sif_dir=sif_dir,
        runtime=runtime,
        ports=ports,
        display=display,
        base_env=base_env,
        writable_tmpfs=writable_tmpfs,
        health_timeout=health_timeout,
    )
    exit_code = 1
    completed = False
    try:
        print(f"[singularity-managed] starting {runtime.name} ports={ports.as_dict()} display={display.display}")
        stack.start()
        process = stack.start_eval(forwarded_args)
        exit_code = process.wait()
        completed = True
        print(f"[singularity-managed] worker {runtime.name} finished with code {exit_code}")
    finally:
        stack.stop()
        if completed and exit_code == 0 and not keep_runtime:
            _safe_remove_runtime(runtime, runtime_root=runtime_root)
    return exit_code


def _run_parallel_workers(
    *,
    singularity_bin: str,
    sif_dir: Path,
    runtime_root: Path,
    run_slug: str,
    forwarded_args: list[str],
    config,
    pending_tasks: tuple[TaskKey, ...],
    base_env: dict[str, str],
    base_port: int,
    port_stride: int,
    base_display: int,
    writable_tmpfs: bool,
    health_timeout: float,
    keep_runtime: bool,
    keep_generated_task_sets: bool,
) -> int:
    evaluation_root = PROJECT_ROOT / "evaluation_examples"
    shards = _managed._stable_round_robin_shards(list(pending_tasks), config.num_envs)
    worker_specs: list[tuple[WorkerStack, subprocess.Popen[Any], Path]] = []
    started_stacks: list[WorkerStack] = []
    shard_paths: list[Path] = []
    exit_code = 0
    workers_completed = False

    try:
        for worker_index, shard_tasks in enumerate(shards, start=1):
            ports = _port_plan(worker_index, base_port=base_port, port_stride=port_stride)
            display = _display_plan(worker_index, base_display=base_display)
            runtime = WorkerRuntime.create(runtime_root=runtime_root, run_slug=run_slug, worker_index=worker_index)
            stack = WorkerStack(
                singularity_bin=singularity_bin,
                sif_dir=sif_dir,
                runtime=runtime,
                ports=ports,
                display=display,
                base_env=base_env,
                writable_tmpfs=writable_tmpfs,
                health_timeout=health_timeout,
            )
            shard_path = _managed._write_shard_task_set(
                evaluation_root=evaluation_root,
                base_name=run_slug,
                worker_index=worker_index,
                shard_tasks=shard_tasks,
            )
            shard_paths.append(shard_path)
            worker_output = f"results/.worker-results/{run_slug}.w{worker_index:02d}.json"
            worker_args = _prepare_forwarded_args_for_singularity_worker(
                forwarded_args=forwarded_args,
                shard_task_set=shard_path.name,
                worker_output=worker_output,
            )
            rewritten_args = _rewrite_forwarded_result_paths(worker_args)
            print(
                f"[singularity-managed] starting {runtime.name} tasks={len(shard_tasks)} "
                f"ports={ports.as_dict()} display={display.display}"
            )
            stack.start()
            started_stacks.append(stack)
            process = stack.start_eval(rewritten_args)
            worker_specs.append((stack, process, shard_path))

        for stack, process, _ in worker_specs:
            return_code = process.wait()
            if return_code != 0 and exit_code == 0:
                exit_code = return_code
            print(f"[singularity-managed] worker {stack.runtime.name} finished with code {return_code}")
        workers_completed = True
    finally:
        for stack in reversed(started_stacks):
            stack.stop()
            if workers_completed and exit_code == 0 and not keep_runtime:
                _safe_remove_runtime(stack.runtime, runtime_root=runtime_root)
        _managed._rebuild_shared_outputs(config)
        if workers_completed and exit_code == 0 and not keep_generated_task_sets:
            _managed._cleanup_generated_task_sets(shard_paths)
        elif shard_paths and keep_generated_task_sets:
            print("[singularity-managed] keeping generated task-set shards by request")
        elif shard_paths and exit_code != 0:
            print("[singularity-managed] keeping generated task-set shards because at least one worker failed")

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Managed Singularity wrapper/orchestrator for scripts/run_benchmark.py",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--sif-dir", type=Path, default=DEFAULT_SIF_DIR)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--port-stride", type=int, default=DEFAULT_PORT_STRIDE)
    parser.add_argument("--base-display", type=int, default=DEFAULT_BASE_DISPLAY)
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument("--singularity-bin", default=None)
    parser.add_argument("--run-name", default="asil-singularity")
    parser.add_argument("--keep-runtime", action="store_true")
    parser.add_argument("--no-writable-tmpfs", action="store_true")
    parser.add_argument(
        "--keep-generated-task-sets",
        action="store_true",
        default=_managed._keep_generated_task_sets(),
        help="Keep managed worker .generated-* task-set shards after a successful parallel run.",
    )
    args, forwarded_args = parser.parse_known_args(argv)

    if not forwarded_args:
        parser.error(
            "Pass benchmark arguments, for example "
            "--task-set test_full15.json --participant asil --asil-execution agentic --provider openai"
        )

    sif_dir = args.sif_dir if args.sif_dir.is_absolute() else PROJECT_ROOT / args.sif_dir
    runtime_root = args.runtime_root if args.runtime_root.is_absolute() else PROJECT_ROOT / args.runtime_root

    benchmark_args, config = _managed._materialize_benchmark_config(forwarded_args)
    num_envs = max(1, int(args.num_envs if args.num_envs is not None else config.num_envs))
    config = _build_config_with_num_envs(config, num_envs)

    task_mapping = _benchmark._load_task_index_mapping(
        task_index=benchmark_args.task_set,
        test_config_base_dir=benchmark_args.test_config_base_dir,
        software_filter=tuple(getattr(benchmark_args, "software_filter", ()) or ()),
        task_id_filter=tuple(getattr(benchmark_args, "task_id_filter", ()) or ()),
    )
    result_root = _benchmark._result_root_for_config(config)
    result_root.mkdir(parents=True, exist_ok=True)
    selection = select_pending_tasks(
        task_mapping,
        result_root=result_root,
        run_mode=config.run_mode,
        participant=config.participant,
        comparison_participants=config.comparison_participants,
        resume=config.resume,
        force_rerun=config.force_rerun,
        rerun_failed_only=config.rerun_failed_only,
    )

    if config.dry_run:
        _managed._dry_run_parallel_plan(
            task_mapping=task_mapping,
            pending_tasks=selection.pending_tasks,
            skipped_tasks=selection.skipped_tasks,
            num_envs=num_envs,
        )
        for worker_index in range(1, num_envs + 1):
            ports = _port_plan(worker_index, base_port=args.base_port, port_stride=args.port_stride)
            display = _display_plan(worker_index, base_display=args.base_display)
            print(f"  worker w{worker_index:02d}: ports={ports.as_dict()} display={display.display}")
        return 0

    singularity_bin = _resolve_singularity_bin(args.singularity_bin)
    _validate_sifs(sif_dir)

    actual_workers = 1 if num_envs == 1 else min(num_envs, max(1, len(selection.pending_tasks)))
    port_plans = [
        _port_plan(worker_index, base_port=args.base_port, port_stride=args.port_stride)
        for worker_index in range(1, actual_workers + 1)
    ]
    display_plans = [
        _display_plan(worker_index, base_display=args.base_display)
        for worker_index in range(1, actual_workers + 1)
    ]
    _check_ports_available(port_plans)
    _install_signal_handlers()

    base_env = _load_base_env(PROJECT_ROOT / ".env")
    run_slug = f"{args.run_name}-{Path(config.task_index).stem}"
    writable_tmpfs = not args.no_writable_tmpfs

    if num_envs == 1:
        rewritten_args = _rewrite_forwarded_result_paths(
            _managed._ensure_flag(
                _managed._replace_or_append_flag(forwarded_args, "--num-envs", "1"),
                "--managed-docker",
            )
        )
        exit_code = _run_one_worker(
            singularity_bin=singularity_bin,
            sif_dir=sif_dir,
            runtime_root=runtime_root,
            run_slug=run_slug,
            worker_index=1,
            ports=port_plans[0],
            display=display_plans[0],
            base_env=base_env,
            forwarded_args=rewritten_args,
            writable_tmpfs=writable_tmpfs,
            health_timeout=args.health_timeout,
            keep_runtime=args.keep_runtime,
        )
        _managed._rebuild_shared_outputs(config)
        _write_singularity_metadata(
            config=config,
            sif_dir=sif_dir,
            singularity_bin=singularity_bin,
            worker_ports=port_plans,
            worker_displays=display_plans,
            runtime_root=runtime_root,
        )
        return exit_code

    if not selection.pending_tasks:
        _managed._rebuild_shared_outputs(config)
        _write_singularity_metadata(
            config=config,
            sif_dir=sif_dir,
            singularity_bin=singularity_bin,
            worker_ports=port_plans,
            worker_displays=display_plans,
            runtime_root=runtime_root,
        )
        print(
            f"All selected tasks already complete under {result_root} "
            f"(skipped={selection.skipped_tasks}/{selection.total_tasks})."
        )
        return 0

    exit_code = _run_parallel_workers(
        singularity_bin=singularity_bin,
        sif_dir=sif_dir,
        runtime_root=runtime_root,
        run_slug=run_slug,
        forwarded_args=forwarded_args,
        config=config,
        pending_tasks=selection.pending_tasks,
        base_env=base_env,
        base_port=args.base_port,
        port_stride=args.port_stride,
        base_display=args.base_display,
        writable_tmpfs=writable_tmpfs,
        health_timeout=args.health_timeout,
        keep_runtime=args.keep_runtime,
        keep_generated_task_sets=args.keep_generated_task_sets,
    )
    _write_singularity_metadata(
        config=config,
        sif_dir=sif_dir,
        singularity_bin=singularity_bin,
        worker_ports=port_plans,
        worker_displays=display_plans,
        runtime_root=runtime_root,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
