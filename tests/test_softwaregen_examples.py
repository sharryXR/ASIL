from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from asil.protocol import Action
from asil.softwaregen import (
    DeterministicSoftwareGenProvider,
    InterfacePlan,
    generate_extension,
    load_onboarding_profile,
    load_reference_catalog,
    probe_extension,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples" / "softwaregen"
COVERAGE = EXAMPLES / "qualified_coverage.json"


def _materialize_example(stem: str, tmp_path: Path):
    profile = load_onboarding_profile(EXAMPLES / f"{stem}_profile.json")
    plan = InterfacePlan.model_validate_json((EXAMPLES / f"{stem}_plan.json").read_text())
    shutil.copy(EXAMPLES / "reference_state.json", tmp_path / "reference_state.json")
    shutil.copy(EXAMPLES / "reference_state_tool.py", tmp_path / "reference_state_tool.py")

    profile_payload = profile.model_dump(mode="json")
    profile_payload["runtime"]["filesystem_root"] = str(tmp_path)
    plan_payload = plan.model_dump(mode="json")
    for view in plan_payload["observation_views"]:
        probe = view["probe"]
        if probe["transport"] == "json_file":
            probe["path"] = "reference_state.json"
        elif probe["transport"] == "command_json":
            probe["argv"] = ["python", "reference_state_tool.py", "--state", "reference_state.json", "observe"]
    for operation in plan_payload["operations"]:
        operation["request"]["argv"] = [
            "python",
            "reference_state_tool.py",
            "--state",
            "reference_state.json",
            "set-value",
            "${item_id}",
            "${value}",
        ]
    return load_onboarding_profile_from_payload(profile_payload), InterfacePlan.model_validate(plan_payload)


def load_onboarding_profile_from_payload(payload: dict):
    from asil.softwaregen import OnboardingProfile

    return OnboardingProfile.model_validate(payload)


@pytest.mark.parametrize("stem", ["file_backed", "native_script"])
def test_reference_example_generates_audited_bundle_and_executes_state_change(stem: str, tmp_path: Path):
    profile, plan = _materialize_example(stem, tmp_path)
    result = generate_extension(profile, DeterministicSoftwareGenProvider(plan))

    assert result.audit.ok
    before = probe_extension(result.bundle)
    action = Action(
        action_type="set_value",
        target="reference_workspace",
        params={"operation": "set_item_value", "arguments": {"item_id": "item-a", "value": "updated"}},
    )
    after = probe_extension(result.bundle, action=action, allow_actions=True)

    assert before["observation"]["element_count"] == 2
    assert after["ok"] is True
    assert after["action"]["validated"] is True
    assert after["action"]["state_changed"] is True
    state = json.loads((tmp_path / "reference_state.json").read_text())
    assert state["items"][0]["value"] == "updated"


def test_qualified_coverage_matches_reference_catalog():
    coverage = json.loads(COVERAGE.read_text())
    catalog = load_reference_catalog()

    assert coverage["qualified_reference_applications"] == 15
    assert coverage["pattern_counts"] == {"file_backed": 6, "native_script": 4, "service_api": 5}
    assert coverage["claim"] == "recurring access-path evidence, not retrospective generation provenance"
    catalog_path = PROJECT_ROOT / coverage["catalog"]
    assert coverage["catalog_sha256"] == hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    assert sorted(coverage["applications"]) == sorted(app.software_id for app in catalog.applications)


def test_gitea_deployment_evidence_is_ready_and_restores_state():
    path = EXAMPLES / "gitea_deployment_evidence.json"
    report = json.loads(path.read_text())

    assert report["ready"] is True
    assert report["qualification"]["tier"] == "direct_declarative"
    assert report["artifact_file_hashes_match"] is True
    assert report["host"]["element_counts"] == [2, 3, 2]
    assert report["docker"]["element_counts"] == [2, 3, 2]
    assert report["evidence_reference_coverage"]["coverage"] == 1.0
