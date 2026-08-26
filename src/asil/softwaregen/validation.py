"""Executable host and Docker validation for generated extension bundles."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any

from asil.protocol import Action
from asil.softwaregen.audit import audit_bundle
from asil.softwaregen.models import ExtensionBundle
from asil.softwaregen.provider import canonical_sha256, sanitize_data
from asil.softwaregen.runtime import DeclarativeAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def observation_state_sha256(observation: Any) -> str:
    """Hash semantic observation state while excluding volatile capture time."""
    payload = observation.model_dump(mode="json")
    if isinstance(payload.get("meta"), dict):
        payload["meta"].pop("timestamp", None)
    return canonical_sha256(payload)


def _observation_summary(observation: Any) -> dict[str, Any]:
    return {
        "sha256": observation_state_sha256(observation),
        "hash_excludes": ["meta.timestamp"],
        "source": observation.meta.observation_source,
        "element_count": len(observation.interactive_elements),
        "element_ids": [element.id for element in observation.interactive_elements],
        "data_summary": observation.data_summary,
    }


def probe_extension(
    bundle: ExtensionBundle,
    *,
    action: Action | None = None,
    allow_actions: bool = False,
) -> dict[str, Any]:
    audit = audit_bundle(bundle)
    if not audit.ok:
        codes = ", ".join(finding.code for finding in audit.findings if finding.severity == "error")
        raise ValueError(f"Bundle failed audit and cannot be probed: {codes}")
    if action is not None and not allow_actions:
        raise PermissionError("An action file requires the explicit --allow-actions flag.")

    started = time.monotonic()
    adapter = DeclarativeAdapter(bundle)
    before = adapter.observe()
    before_summary = _observation_summary(before)
    action_report: dict[str, Any] | None = None
    final_observation = before
    if action is not None:
        if not adapter.validate_action(action):
            raise ValueError("The supplied semantic action failed adapter validation.")
        final_observation = adapter.execute(action)
        after_summary = _observation_summary(final_observation)
        action_report = {
            "validated": True,
            "action": sanitize_data(action.model_dump(mode="json")),
            "before_sha256": before_summary["sha256"],
            "after_sha256": after_summary["sha256"],
            "state_changed": before_summary["sha256"] != after_summary["sha256"],
        }

    return {
        "ok": True,
        "software_id": bundle.profile.software_id,
        "schema_version": bundle.schema_version,
        "bundle_sha256": canonical_sha256(bundle.model_dump(mode="json")),
        "audit": {
            "ok": audit.ok,
            "errors": audit.error_count,
            "warnings": audit.warning_count,
        },
        "observation": _observation_summary(final_observation),
        "action": action_report,
        "elapsed_s": time.monotonic() - started,
    }


def build_docker_probe_command(
    bundle_path: str | Path,
    *,
    image: str,
    network: str = "",
    env_names: list[str] | None = None,
    action_path: str | Path | None = None,
    allow_actions: bool = False,
) -> list[str]:
    bundle = Path(bundle_path).resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{PROJECT_ROOT}:/workspace:ro",
        "-v",
        f"{bundle}:/softwaregen/extension.json:ro",
        "-w",
        "/workspace",
        "-e",
        "PYTHONPATH=/workspace/src",
    ]
    if network:
        command.extend(["--network", network])
    for name in env_names or []:
        command.extend(["-e", name])
    if action_path is not None:
        action = Path(action_path).resolve()
        command.extend(["-v", f"{action}:/softwaregen/action.json:ro"])
    command.extend([image, "python", "-m", "asil.softwaregen", "probe", "/softwaregen/extension.json"])
    if action_path is not None:
        command.extend(["--action", "/softwaregen/action.json"])
    if allow_actions:
        command.append("--allow-actions")
    command.append("--json")
    return command


def docker_probe_extension(
    bundle_path: str | Path,
    *,
    image: str,
    network: str = "",
    env_names: list[str] | None = None,
    action_path: str | Path | None = None,
    allow_actions: bool = False,
) -> dict[str, Any]:
    command = build_docker_probe_command(
        bundle_path,
        image=image,
        network=network,
        env_names=env_names,
        action_path=action_path,
        allow_actions=allow_actions,
    )
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker probe exited {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    payload = json.loads(result.stdout)
    payload["docker"] = {
        "image": image,
        "network": network,
        "elapsed_s": time.monotonic() - started,
        "command": sanitize_data(command),
    }
    return payload
