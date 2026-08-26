"""Tests for ASIL sandbox -- uses mocks, no Docker required."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from asil.sandbox import ASILSandbox


def test_sandbox_detects_no_docker():
    """Sandbox should detect when Docker is unavailable."""
    sandbox = ASILSandbox()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert sandbox.is_docker_available is False


def test_sandbox_sandbox_mode_env():
    """Sandbox should detect sandbox mode from environment."""
    sandbox = ASILSandbox()
    with patch.dict("os.environ", {"ASIL_SANDBOX": "true"}):
        assert sandbox.is_sandbox_mode is True
    with patch.dict("os.environ", {"ASIL_SANDBOX": "false"}, clear=False):
        assert sandbox.is_sandbox_mode is False


def test_sandbox_gitea_url_configurable():
    """Gitea URL should be configurable via environment."""
    sandbox = ASILSandbox()
    with patch.dict("os.environ", {"GITEA_URL": "http://my-gitea:3000"}):
        assert sandbox.gitea_url == "http://my-gitea:3000"
    # Default
    with patch.dict("os.environ", {}, clear=True):
        assert sandbox.gitea_url == "http://localhost:3000"


def test_sandbox_start_requires_docker():
    """Starting sandbox should fail gracefully without Docker."""
    sandbox = ASILSandbox()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        try:
            sandbox.start()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Docker is not available" in str(e)


def test_sandbox_context_manager():
    """Sandbox should work as context manager."""
    sandbox = ASILSandbox()

    mock_compose_result = MagicMock()
    mock_compose_result.returncode = 0

    mock_docker_result = MagicMock()
    mock_docker_result.returncode = 0

    with patch("subprocess.run", return_value=mock_docker_result):
        with patch.object(sandbox, "_compose_cmd", return_value=mock_compose_result):
            with patch.object(sandbox, "_wait_for_gitea"):
                with patch.object(sandbox, "_wait_for_obs_mock"):
                    with sandbox:
                        assert sandbox._started is True
                    # After exit, stop should be called
                    assert sandbox._started is False


def test_sandbox_service_status():
    """Service status should parse docker compose ps output."""
    sandbox = ASILSandbox()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '{"Service":"gitea","Health":"healthy"}\n{"Service":"creative-tools","Status":"running"}'

    with patch.object(sandbox, "_compose_cmd", return_value=mock_result):
        status = sandbox.get_service_status()
        assert status["gitea"] == "healthy"
        assert status["creative-tools"] == "running"
