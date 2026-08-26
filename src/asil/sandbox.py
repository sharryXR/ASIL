"""ASIL Sandbox -- Docker-based isolated evaluation environment.

Provides Docker infrastructure services (Gitea, OBS mock) for reproducible
experiments. The evaluation itself runs on the host (or inside the eval
container when using ``docker compose --profile eval run eval ...``).

Usage (host-side)::

    with ASILSandbox() as sb:
        # Gitea at sb.gitea_url, OBS mock at sb.obs_ws_host:sb.obs_ws_port
        token = sb.gitea_token
        ...

Usage (eval container)::

    # Services are already running; read env vars directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class ASILSandbox:
    """Manages Docker infrastructure services for ASIL evaluation."""

    COMPOSE_FILE = Path(__file__).parent.parent.parent / "docker" / "docker-compose.yml"

    def __init__(
        self,
        compose_path: Path | str | None = None,
        project_name: str = "asil-eval",
    ):
        self.compose_path = Path(compose_path) if compose_path else self.COMPOSE_FILE
        self.project_name = project_name
        self._started = False
        self._gitea_token: str | None = None

    # ── Docker availability ──────────────────────────────────────────────

    @property
    def is_docker_available(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def is_sandbox_mode(self) -> bool:
        """True when running inside a Docker container (eval service)."""
        return os.environ.get("ASIL_SANDBOX", "").lower() == "true"

    # ── Service endpoints ────────────────────────────────────────────────

    @property
    def gitea_url(self) -> str:
        return os.environ.get("GITEA_URL", "http://localhost:3000")

    @property
    def obs_ws_host(self) -> str:
        return os.environ.get("OBS_WS_HOST", "localhost")

    @property
    def obs_ws_port(self) -> int:
        return int(os.environ.get("OBS_WS_PORT", "4455"))

    @property
    def gitea_token(self) -> str:
        """Return the Gitea API token, creating one if needed.

        When the sandbox started services itself, always creates a fresh token
        (env vars may contain stale tokens from previous runs).
        """
        if self._gitea_token:
            return self._gitea_token
        if not self._started:
            # Not managing services — trust env var
            token = os.environ.get("GITEA_TOKEN", "")
            if token:
                self._gitea_token = token
                return token
        # Create a fresh token via REST API
        token = self._create_gitea_token()
        if token:
            self._gitea_token = token
        return token

    def _create_gitea_token(self) -> str:
        """Create a Gitea API token using basic auth."""
        import requests as req

        api = f"{self.gitea_url}/api/v1"
        auth = ("asil_admin", "asil_password")
        no_proxy = {"http": None, "https": None}

        # Delete existing token (idempotent)
        req.delete(
            f"{api}/users/asil_admin/tokens/asil-eval-token",
            auth=auth, timeout=5, proxies=no_proxy,
        )
        # Create new token
        try:
            resp = req.post(
                f"{api}/users/asil_admin/tokens",
                auth=auth,
                json={"name": "asil-eval-token", "scopes": ["all"]},
                timeout=5, proxies=no_proxy,
            )
            if resp.status_code in (200, 201):
                return resp.json().get("sha1", "")
        except Exception:
            pass
        return ""

    # ── Docker Compose helpers ───────────────────────────────────────────

    def _compose_cmd(self, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
        cmd = [
            "docker", "compose",
            "-f", str(self.compose_path),
            "-p", self.project_name,
            *args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start infrastructure services (gitea, gitea-init, obs-mock).

        Does NOT start the eval container (it uses a Docker profile).
        """
        if not self.is_docker_available:
            raise RuntimeError("Docker is not available. Install Docker or use local mode.")

        print("  Starting infrastructure services (gitea, obs-mock)...")

        # Start gitea + obs-mock (not eval — it's behind a profile)
        result = self._compose_cmd("up", "-d", "--wait", "gitea", "obs-mock", timeout=180)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start services:\n{result.stderr}\n{result.stdout}"
            )

        # Run gitea-init (one-shot)
        print("  Initializing Gitea...")
        result = self._compose_cmd("up", "gitea-init", timeout=60)
        if result.returncode != 0:
            # Non-fatal: gitea-init may have already run
            print(f"  WARNING: gitea-init returned {result.returncode}: {result.stderr[:200]}")

        # Wait for services to be fully ready
        self._wait_for_gitea()
        self._wait_for_obs_mock()

        self._started = True
        print("  All services ready.")

    def stop(self) -> None:
        """Stop and remove all containers (preserves volumes)."""
        if self._started:
            self._compose_cmd("down", timeout=60)
            self._started = False

    def reset(self) -> None:
        """Full reset: tear down volumes and restart."""
        if self._started:
            self._compose_cmd("down", "-v", timeout=60)
            self._started = False
        self.start()

    # ── Health checks ────────────────────────────────────────────────────

    def _wait_for_gitea(self, timeout: int = 60) -> None:
        """Wait until Gitea API responds."""
        import requests as req

        deadline = time.time() + timeout
        url = f"{self.gitea_url}/api/v1/version"
        while time.time() < deadline:
            try:
                resp = req.get(url, timeout=3, proxies={"http": None, "https": None})
                if resp.status_code == 200:
                    return
            except req.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError(f"Gitea not ready after {timeout}s at {self.gitea_url}")

    def _wait_for_obs_mock(self, timeout: int = 30) -> None:
        """Wait until OBS mock WebSocket accepts connections."""
        import socket

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect((self.obs_ws_host, self.obs_ws_port))
                s.close()
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        raise RuntimeError(
            f"OBS mock not ready after {timeout}s at "
            f"{self.obs_ws_host}:{self.obs_ws_port}"
        )

    def get_service_status(self) -> dict[str, str]:
        """Get status of all sandbox services."""
        result = self._compose_cmd("ps", "--format", "json")
        if result.returncode != 0:
            return {}

        services = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    services[data.get("Service", "unknown")] = data.get(
                        "Health", data.get("Status", "unknown")
                    )
                except json.JSONDecodeError:
                    pass
        return services

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "ASILSandbox":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
