from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bootstrap_rebuttal_docker.sh"


def _fake_environment(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    project = tmp_path / "project"
    (project / "docker").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "results/setup").mkdir(parents=True)
    (project / ".env.example").write_text("OPENAI_API_KEY=\nBOOTSTRAP_VALUE=example\n")
    (project / "docker/docker-compose.yml").write_text("services: {}\n")

    log = tmp_path / "commands.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf "python %s\\n" "$*" >> "$ASIL_BOOTSTRAP_TEST_LOG"\n'
        "report=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--report' ]; then report=$2; shift 2; else shift; fi\n"
        "done\n"
        "if [ -n \"$report\" ]; then mkdir -p \"$(dirname \"$report\")\"; printf '{\"ready\":true,\"gates\":[]}\\n' > \"$report\"; fi\n"
    )
    fake_python.chmod(0o755)

    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf "docker %s\\n" "$*" >> "$ASIL_BOOTSTRAP_TEST_LOG"\n'
    )
    fake_docker.chmod(0o755)

    env = {
        **os.environ,
        "ASIL_BOOTSTRAP_PROJECT_ROOT": str(project),
        "ASIL_BOOTSTRAP_PYTHON_BIN": str(fake_python),
        "ASIL_BOOTSTRAP_DOCKER_BIN": str(fake_docker),
        "ASIL_BOOTSTRAP_TEST_LOG": str(log),
        "ASIL_BOOTSTRAP_SKIP_FLOCK": "1",
    }
    return project, env, log


def _run(tmp_path: Path, *args: str, env_update: dict[str, str] | None = None):
    project, env, log = _fake_environment(tmp_path)
    env.update(env_update or {})
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    commands = log.read_text().splitlines() if log.exists() else []
    return result, project, commands


def test_default_sequence_is_check_build_verify_cleanup(tmp_path: Path) -> None:
    result, project, commands = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    joined = "\n".join(commands)
    assert joined.index("--phase preflight") < joined.index("compose")
    assert "build obs-mock eval" in joined
    assert joined.index("build obs-mock eval") < joined.index("--phase all")
    assert "-p asil-bootstrap down --remove-orphans" in joined
    assert (project / "results/setup/docker_bootstrap_report.json").is_file()


def test_missing_env_is_created_mode_0600(tmp_path: Path) -> None:
    result, project, _ = _run(tmp_path, "--check-only")

    assert result.returncode == 0
    env_path = project / ".env"
    assert env_path.read_text() == (project / ".env.example").read_text()
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_existing_env_is_never_overwritten(tmp_path: Path) -> None:
    project, env, log = _fake_environment(tmp_path)
    env_path = project / ".env"
    env_path.write_text("OPENAI_API_KEY=preserve-me\n")
    env_path.chmod(0o640)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert env_path.read_text() == "OPENAI_API_KEY=preserve-me\n"
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o640
    assert "preserve-me" not in result.stdout + result.stderr + log.read_text()


def test_failure_triggers_only_prefixed_compose_cleanup(tmp_path: Path) -> None:
    project, env, log = _fake_environment(tmp_path)
    failing_python = Path(env["ASIL_BOOTSTRAP_PYTHON_BIN"])
    failing_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "python %s\\n" "$*" >> "$ASIL_BOOTSTRAP_TEST_LOG"\n'
        "exit 7\n"
    )
    failing_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=project, env=env, capture_output=True, text=True
    )

    assert result.returncode == 7
    commands = log.read_text()
    assert "-p asil-bootstrap down --remove-orphans" in commands
    assert "docker rm" not in commands
    assert "docker network rm" not in commands


def test_cli_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    result, _, commands = _run(tmp_path, "--check-only", "--verify-only")

    assert result.returncode == 2
    assert commands == []
    assert "mutually exclusive" in result.stderr


def test_build_only_does_not_run_all_verification(tmp_path: Path) -> None:
    result, _, commands = _run(tmp_path, "--build-only")

    assert result.returncode == 0
    joined = "\n".join(commands)
    assert "build obs-mock eval" in joined
    assert "--phase all" not in joined


def test_verify_only_does_not_build(tmp_path: Path) -> None:
    result, _, commands = _run(tmp_path, "--verify-only")

    assert result.returncode == 0
    joined = "\n".join(commands)
    assert "--phase all" in joined
    assert "build obs-mock eval" not in joined


def test_require_openai_stops_before_build_when_key_missing(tmp_path: Path) -> None:
    result, _, commands = _run(tmp_path, "--require-openai")

    assert result.returncode == 2
    assert "OPENAI_API_KEY" in result.stderr
    assert not any("build obs-mock eval" in command for command in commands)


def test_no_cache_is_forwarded_to_compose_build(tmp_path: Path) -> None:
    result, _, commands = _run(tmp_path, "--build-only", "--no-cache")

    assert result.returncode == 0
    assert any("build --no-cache obs-mock eval" in command for command in commands)
