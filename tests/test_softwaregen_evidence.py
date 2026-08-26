from __future__ import annotations

import json
from pathlib import Path

import pytest

from asil.softwaregen import (
    DeterministicSoftwareGenProvider,
    InterfacePlan,
    OnboardingProfile,
    build_deployment_evidence_report,
    canonical_sha256,
    generate_extension,
    softwaregen_main,
    write_extension_bundle,
)


def _profile() -> OnboardingProfile:
    return OnboardingProfile.model_validate(
        {
            "software_id": "demo_service",
            "display_name": "Demo Service",
            "version": "1.0",
            "integration_pattern": "service_api",
            "description": "A service used to validate deployment evidence.",
            "evidence": [
                {
                    "id": "ev_list",
                    "kind": "api_spec",
                    "locator": "GET /api/items",
                    "excerpt": "Returns JSON items with stable IDs.",
                    "sample": [{"id": 1, "name": "one"}],
                },
                {
                    "id": "ev_create",
                    "kind": "api_spec",
                    "locator": "POST /api/items",
                    "excerpt": "Creates one item from JSON.",
                },
            ],
            "runtime": {
                "base_url": "http://localhost:8765",
                "allowed_hosts": ["localhost"],
                "allowed_executables": [],
                "filesystem_root": ".",
            },
            "requirements": [],
            "known_limitations": ["Only list and create are covered."],
        }
    )


def _plan() -> InterfacePlan:
    return InterfacePlan.model_validate(
        {
            "summary": "Observe and create items.",
            "observation_views": [
                {
                    "id": "items",
                    "description": "Current items.",
                    "probe": {
                        "transport": "http_json",
                        "method": "GET",
                        "path": "/api/items",
                        "items_pointer": "",
                        "evidence_refs": ["ev_list"],
                    },
                    "element": {
                        "id_prefix": "item:",
                        "id_pointer": "/id",
                        "type": "item",
                        "label_pointer": "/name",
                        "value_fields": {"name": "/name"},
                        "actions": ["create_item"],
                        "evidence_refs": ["ev_list"],
                    },
                    "evidence_refs": ["ev_list"],
                }
            ],
            "operations": [
                {
                    "name": "create_item",
                    "description": "Create an item.",
                    "action_type": "api_call",
                    "target": "demo_service",
                    "parameters": [
                        {"name": "name", "value_type": "string", "required": True}
                    ],
                    "request": {
                        "transport": "http_json",
                        "method": "POST",
                        "path": "/api/items",
                        "body": {"name": "${name}"},
                    },
                    "evidence_refs": ["ev_create"],
                }
            ],
            "limitations": ["Only list and create are covered."],
        }
    )


def _artifact_fixture(tmp_path: Path):
    profile = _profile()
    result = generate_extension(profile, DeterministicSoftwareGenProvider(_plan()))
    artifact_dir = tmp_path / "generated"
    generation_report = write_extension_bundle(result, artifact_dir)
    bundle = result.bundle
    bundle_sha256 = canonical_sha256(bundle.model_dump(mode="json"))
    return profile, bundle, generation_report, artifact_dir, bundle_sha256


def _probe_report(
    bundle_sha256: str,
    element_count: int,
    *,
    action: bool = False,
    docker: bool = False,
    state_changed: bool = True,
) -> dict:
    report = {
        "ok": True,
        "software_id": "demo_service",
        "schema_version": "1.0",
        "bundle_sha256": bundle_sha256,
        "audit": {"ok": True, "errors": 0, "warnings": 0},
        "observation": {
            "sha256": f"{element_count:064x}",
            "source": "rest_api",
            "element_count": element_count,
            "element_ids": [f"item:{index}" for index in range(element_count)],
            "data_summary": f"{element_count} items",
        },
        "action": None,
        "elapsed_s": 0.1,
    }
    if action:
        report["action"] = {
            "validated": True,
            "state_changed": state_changed,
            "before_sha256": "1" * 64,
            "after_sha256": "2" * 64 if state_changed else "1" * 64,
            "action": {"action_type": "api_call", "target": "demo_service", "params": {}},
        }
    if docker:
        report["docker"] = {"image": "asil-eval:local", "command": ["docker", "run"], "elapsed_s": 0.2}
    return report


def _valid_inputs(tmp_path: Path):
    profile, bundle, generation_report, artifact_dir, bundle_hash = _artifact_fixture(tmp_path)
    host = [
        _probe_report(bundle_hash, 2),
        _probe_report(bundle_hash, 3, action=True),
        _probe_report(bundle_hash, 2, action=True),
    ]
    docker = [
        _probe_report(bundle_hash, 2, docker=True),
        _probe_report(bundle_hash, 3, action=True, docker=True),
        _probe_report(bundle_hash, 2, action=True, docker=True),
    ]
    return profile, bundle, generation_report, artifact_dir, host, docker


def test_evidence_report_accepts_matching_audited_host_and_docker_state_changes(tmp_path: Path):
    profile, bundle, generation_report, artifact_dir, host, docker = _valid_inputs(tmp_path)

    report = build_deployment_evidence_report(
        profile=profile,
        bundle=bundle,
        generation_report=generation_report,
        artifact_dir=artifact_dir,
        host_reports=host,
        docker_reports=docker,
    )

    assert report.ready is True
    assert report.host.passed == 3
    assert report.docker.passed == 3
    assert report.host.element_counts == [2, 3, 2]
    assert report.docker.element_counts == [2, 3, 2]
    assert report.generated_stages == [
        "interface plan",
        "extension bundle",
        "action schema",
        "adapter wrapper",
    ]
    assert "task and evaluator design" in report.reviewed_stages
    assert report.artifact_file_hashes_match is True


def test_evidence_report_rejects_probe_for_another_bundle(tmp_path: Path):
    profile, bundle, generation_report, artifact_dir, host, docker = _valid_inputs(tmp_path)
    host[0]["bundle_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="bundle_sha256"):
        build_deployment_evidence_report(
            profile=profile,
            bundle=bundle,
            generation_report=generation_report,
            artifact_dir=artifact_dir,
            host_reports=host,
            docker_reports=docker,
        )


def test_evidence_report_rejects_action_without_validated_state_change(tmp_path: Path):
    profile, bundle, generation_report, artifact_dir, host, docker = _valid_inputs(tmp_path)
    host[1] = _probe_report(
        canonical_sha256(bundle.model_dump(mode="json")),
        3,
        action=True,
        state_changed=False,
    )

    with pytest.raises(ValueError, match="validated state change"):
        build_deployment_evidence_report(
            profile=profile,
            bundle=bundle,
            generation_report=generation_report,
            artifact_dir=artifact_dir,
            host_reports=host,
            docker_reports=docker,
        )


def test_evidence_report_rejects_artifact_hash_mismatch(tmp_path: Path):
    profile, bundle, generation_report, artifact_dir, host, docker = _valid_inputs(tmp_path)
    (artifact_dir / "adapter.py").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact SHA-256"):
        build_deployment_evidence_report(
            profile=profile,
            bundle=bundle,
            generation_report=generation_report,
            artifact_dir=artifact_dir,
            host_reports=host,
            docker_reports=docker,
        )


def test_evidence_report_cli_writes_ready_report(tmp_path: Path, capsys):
    profile, bundle, generation_report, artifact_dir, host, docker = _valid_inputs(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    for index, report in enumerate(host):
        (tmp_path / f"host-{index}.json").write_text(json.dumps(report), encoding="utf-8")
    for index, report in enumerate(docker):
        (tmp_path / f"docker-{index}.json").write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "evidence.json"

    argv = [
        "evidence-report",
        str(profile_path),
        str(artifact_dir / "extension.json"),
        "--generation-report",
        str(artifact_dir / "generation_report.json"),
    ]
    for index in range(3):
        argv.extend(["--host-report", str(tmp_path / f"host-{index}.json")])
    for index in range(3):
        argv.extend(["--docker-report", str(tmp_path / f"docker-{index}.json")])
    argv.extend(["--output", str(output)])

    assert softwaregen_main(argv) == 0
    payload = json.loads(output.read_text())
    assert payload["ready"] is True
    assert payload["host"]["passed"] == 3
    assert payload["docker"]["passed"] == 3
    assert json.loads(capsys.readouterr().out)["ready"] is True
