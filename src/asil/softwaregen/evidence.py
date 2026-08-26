"""Fail-closed assembly of software onboarding deployment evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from asil.softwaregen.audit import audit_bundle
from asil.softwaregen.models import (
    DeploymentEvidenceReport,
    ExtensionBundle,
    OnboardingProfile,
    ProbeEvidenceSummary,
)
from asil.softwaregen.provider import canonical_sha256
from asil.softwaregen.qualification import qualify_profile


_GENERATED_STAGES = ["interface plan", "extension bundle", "action schema", "adapter wrapper"]
_REVIEWED_STAGES = [
    "interface evidence and runtime permissions",
    "custom parser or script bridge when required",
    "task and evaluator design",
    "GUI synchronization and rendering when required",
    "application-specific and unsupported semantics",
]


def _verify_artifact_hashes(generation_report: dict[str, Any], artifact_dir: Path) -> dict[str, str]:
    hashes = generation_report.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Generation report does not contain artifact SHA-256 values.")
    verified: dict[str, str] = {}
    for name, expected in hashes.items():
        path = artifact_dir / str(name)
        if not path.is_file():
            raise ValueError(f"Generation artifact is missing: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Generation artifact SHA-256 mismatch for {name}.")
        verified[str(name)] = actual
    return verified


def _summarize_probe_reports(
    reports: Iterable[dict[str, Any]],
    *,
    environment: str,
    bundle_sha256: str,
) -> ProbeEvidenceSummary:
    rows = list(reports)
    if not rows:
        raise ValueError(f"At least one {environment} probe report is required.")
    element_counts: list[int] = []
    action_reports = 0
    validated_changes = 0
    for index, report in enumerate(rows):
        if report.get("bundle_sha256") != bundle_sha256:
            raise ValueError(f"{environment} report {index} bundle_sha256 does not match the extension bundle.")
        audit = report.get("audit") if isinstance(report.get("audit"), dict) else {}
        if not report.get("ok") or not audit.get("ok") or int(audit.get("errors", 0)) != 0:
            raise ValueError(f"{environment} report {index} did not pass runtime audit.")
        observation = report.get("observation") if isinstance(report.get("observation"), dict) else {}
        count = observation.get("element_count")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"{environment} report {index} lacks a valid observation element_count.")
        element_counts.append(count)
        action = report.get("action")
        if action is not None:
            action_reports += 1
            if not isinstance(action, dict):
                raise ValueError(f"{environment} report {index} contains malformed action evidence.")
            changed = bool(action.get("state_changed"))
            validated = bool(action.get("validated"))
            before = action.get("before_sha256")
            after = action.get("after_sha256")
            if not validated or not changed or not before or not after or before == after:
                raise ValueError(f"{environment} report {index} does not contain a validated state change.")
            validated_changes += 1
    return ProbeEvidenceSummary(
        environment=environment,
        reports=len(rows),
        passed=len(rows),
        action_reports=action_reports,
        validated_state_changes=validated_changes,
        element_counts=element_counts,
    )


def _evidence_coverage(bundle: ExtensionBundle) -> dict[str, Any]:
    available = {item.id for item in bundle.profile.evidence}
    used: set[str] = set()
    for view in bundle.plan.observation_views:
        used.update(view.evidence_refs)
        used.update(view.probe.evidence_refs)
        used.update(view.element.evidence_refs)
    for operation in bundle.plan.operations:
        used.update(operation.evidence_refs)
    missing = sorted(used - available)
    return {
        "available": len(available),
        "used": len(used),
        "coverage": (len(used) / len(available)) if available else 0.0,
        "unused_evidence_ids": sorted(available - used),
        "missing_evidence_ids": missing,
    }


def build_deployment_evidence_report(
    *,
    profile: OnboardingProfile,
    bundle: ExtensionBundle,
    generation_report: dict[str, Any],
    artifact_dir: str | Path,
    host_reports: Iterable[dict[str, Any]],
    docker_reports: Iterable[dict[str, Any]],
) -> DeploymentEvidenceReport:
    if profile.model_dump(mode="json") != bundle.profile.model_dump(mode="json"):
        raise ValueError("The supplied onboarding profile does not match the extension bundle profile.")
    audit = audit_bundle(bundle)
    if not audit.ok:
        raise ValueError("The extension bundle failed static audit.")
    generation_audit = generation_report.get("audit") if isinstance(generation_report.get("audit"), dict) else {}
    if not generation_report.get("ok") or not generation_audit.get("ok") or int(generation_audit.get("error_count", 0)):
        raise ValueError("The generation report contains audit errors.")
    if generation_report.get("profile_sha256") != bundle.provenance.profile_sha256:
        raise ValueError("Generation report profile_sha256 does not match bundle provenance.")
    if generation_report.get("plan_sha256") != bundle.provenance.plan_sha256:
        raise ValueError("Generation report plan_sha256 does not match bundle provenance.")

    verified_hashes = _verify_artifact_hashes(generation_report, Path(artifact_dir))
    bundle_hash = canonical_sha256(bundle.model_dump(mode="json"))
    host = _summarize_probe_reports(host_reports, environment="host", bundle_sha256=bundle_hash)
    docker = _summarize_probe_reports(docker_reports, environment="docker", bundle_sha256=bundle_hash)
    qualification = qualify_profile(profile)
    if not qualification.eligible:
        raise ValueError("The onboarding profile is out of scope for assisted deployment.")
    coverage = _evidence_coverage(bundle)
    if coverage["missing_evidence_ids"]:
        raise ValueError("The extension bundle cites missing evidence IDs.")

    limitations = list(
        dict.fromkeys([*profile.known_limitations, *bundle.plan.limitations, qualification.claim_boundary])
    )
    return DeploymentEvidenceReport(
        ready=True,
        software_id=profile.software_id,
        qualification=qualification,
        provider=bundle.provenance.provider,
        model=bundle.provenance.model,
        api_calls=bundle.provenance.api_calls,
        generation_elapsed_s=bundle.provenance.elapsed_s,
        bundle_sha256=bundle_hash,
        profile_sha256=bundle.provenance.profile_sha256,
        plan_sha256=bundle.provenance.plan_sha256,
        artifact_sha256=verified_hashes,
        artifact_file_hashes_match=True,
        audit=audit,
        evidence_reference_coverage=coverage,
        observation_view_count=len(bundle.plan.observation_views),
        operation_count=len(bundle.plan.operations),
        host=host,
        docker=docker,
        generated_stages=list(_GENERATED_STAGES),
        reviewed_stages=list(_REVIEWED_STAGES),
        limitations=limitations,
        claim_boundary=qualification.claim_boundary,
    )
