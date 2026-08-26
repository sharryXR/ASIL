from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from asil.softwaregen import (
    OnboardingProfile,
    ReferenceCatalog,
    load_reference_catalog,
    qualify_profile,
    softwaregen_main,
)


def _profile_payload(
    *,
    software_id: str,
    integration_pattern: str,
    evidence: list[dict],
    runtime: dict | None = None,
) -> dict:
    return {
        "software_id": software_id,
        "display_name": software_id.replace("_", " ").title(),
        "version": "1.0",
        "integration_pattern": integration_pattern,
        "description": "Qualification test profile.",
        "evidence": evidence,
        "runtime": runtime
        or {
            "base_url": "",
            "allowed_hosts": [],
            "allowed_executables": [],
            "filesystem_root": ".",
        },
        "requirements": [],
        "known_limitations": [],
    }


def _gitea_profile_payload() -> dict:
    return _profile_payload(
        software_id="gitea_demo",
        integration_pattern="service_api",
        evidence=[
            {
                "id": "ev_list",
                "kind": "api_spec",
                "locator": "GET /api/items",
                "excerpt": "Returns a JSON list of items with stable IDs.",
                "sample": [{"id": 1, "name": "one"}],
            },
            {
                "id": "ev_create",
                "kind": "api_spec",
                "locator": "POST /api/items",
                "excerpt": "Creates one item from a JSON request body.",
            },
        ],
        runtime={
            "base_url": "http://localhost:3000",
            "allowed_hosts": ["localhost"],
            "allowed_executables": [],
            "filesystem_root": ".",
        },
    )


def _svg_profile_payload() -> dict:
    return _profile_payload(
        software_id="svg_editor",
        integration_pattern="file_backed",
        evidence=[
            {
                "id": "ev_svg",
                "kind": "file_format",
                "locator": "document.svg",
                "excerpt": "SVG is an open XML format whose elements and attributes can be read and edited.",
            }
        ],
    )


def _opaque_profile_payload() -> dict:
    return _profile_payload(
        software_id="opaque_gui",
        integration_pattern="hybrid",
        evidence=[
            {
                "id": "ev_gui",
                "kind": "source_code",
                "locator": "GUI documentation",
                "excerpt": "Only a rendered GUI is documented; no structured read or semantic action interface is exposed.",
            }
        ],
        runtime={
            "base_url": "",
            "allowed_hosts": [],
            "allowed_executables": [],
            "filesystem_root": ".",
        },
    )


def test_reference_catalog_matches_paper_pattern_counts():
    catalog = load_reference_catalog()

    assert len(catalog.applications) == 15
    assert len({app.software_id for app in catalog.applications}) == 15
    assert Counter(app.pattern for app in catalog.applications) == {
        "file_backed": 6,
        "native_script": 4,
        "service_api": 5,
    }


def test_reference_catalog_models_reject_extra_fields():
    payload = load_reference_catalog().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        ReferenceCatalog.model_validate(payload)


def test_reference_catalog_rejects_duplicate_software_ids(tmp_path: Path):
    payload = load_reference_catalog().model_dump(mode="json")
    payload["applications"].append(payload["applications"][0])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate software_id"):
        load_reference_catalog(path)


def test_qualify_service_api_profile_as_direct_declarative():
    report = qualify_profile(OnboardingProfile.model_validate(_gitea_profile_payload()))

    assert report.eligible is True
    assert report.tier == "direct_declarative"
    assert report.available_observation_transports == ["http_json"]
    assert report.available_action_transports == ["http_json"]
    assert report.blockers == []


def test_qualify_open_non_json_file_as_bridge_assisted():
    profile = OnboardingProfile.model_validate(_svg_profile_payload())

    report = qualify_profile(profile)

    assert report.eligible is True
    assert report.tier == "bridge_assisted"
    assert "reviewed structured-output parser or converter" in report.required_human_work
    assert report.available_observation_transports == ["command_json"]


def test_qualify_native_script_with_allowlisted_executable_as_direct_declarative():
    payload = _profile_payload(
        software_id="scriptable_app",
        integration_pattern="native_script",
        evidence=[
            {
                "id": "ev_script",
                "kind": "script",
                "locator": "app --python bridge.py",
                "excerpt": "The reviewed bridge emits JSON state and accepts semantic operation arguments.",
            }
        ],
        runtime={
            "base_url": "",
            "allowed_hosts": [],
            "allowed_executables": ["app"],
            "filesystem_root": ".",
        },
    )

    report = qualify_profile(OnboardingProfile.model_validate(payload))

    assert report.eligible is True
    assert report.tier == "direct_declarative"
    assert report.available_observation_transports == ["command_json"]
    assert report.available_action_transports == ["command_json"]


def test_qualify_json_file_with_evidenced_edit_path_as_direct_declarative():
    payload = _profile_payload(
        software_id="json_workspace",
        integration_pattern="file_backed",
        evidence=[
            {
                "id": "ev_file",
                "kind": "file_format",
                "locator": "workspace.json",
                "excerpt": "The JSON document stores stable object IDs and editable values.",
                "sample": {"items": [{"id": "a", "value": 1}]},
            },
            {
                "id": "ev_tool",
                "kind": "command_help",
                "locator": "workspace-tool set-value",
                "excerpt": "The direct command updates one identified value and emits JSON.",
            },
        ],
        runtime={
            "base_url": "",
            "allowed_hosts": [],
            "allowed_executables": ["workspace-tool"],
            "filesystem_root": "/workspace",
        },
    )

    report = qualify_profile(OnboardingProfile.model_validate(payload))

    assert report.eligible is True
    assert report.tier == "direct_declarative"
    assert report.available_observation_transports == ["json_file"]
    assert report.available_action_transports == ["command_json"]


def test_qualify_opaque_profile_as_out_of_scope():
    profile = OnboardingProfile.model_validate(_opaque_profile_payload())

    report = qualify_profile(profile)

    assert report.eligible is False
    assert report.tier == "out_of_scope"
    assert "no evidenced open read path" in report.blockers
    assert "no evidenced semantic action path" in report.blockers


def test_catalog_cli_emits_fifteen_reference_applications(capsys):
    assert softwaregen_main(["catalog"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["applications"]) == 15


def test_qualify_cli_returns_nonzero_for_out_of_scope_profile(tmp_path: Path, capsys):
    profile = tmp_path / "opaque.json"
    profile.write_text(json.dumps(_opaque_profile_payload()), encoding="utf-8")

    assert softwaregen_main(["qualify", str(profile)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "out_of_scope"
    assert payload["eligible"] is False
