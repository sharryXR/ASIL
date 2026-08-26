"""Deterministic assembly and artifact writing for softwaregen bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from asil.softwaregen.audit import audit_bundle
from asil.softwaregen.models import AuditReport, ExtensionBundle, GenerationProvenance, OnboardingProfile
from asil.softwaregen.provider import ProviderOutput, SoftwareGenProvider, canonical_sha256, sanitize_data


@dataclass(frozen=True)
class GenerationResult:
    bundle: ExtensionBundle
    audit: AuditReport
    provider_output: ProviderOutput


def generate_extension(profile: OnboardingProfile, provider: SoftwareGenProvider) -> GenerationResult:
    output = provider.generate_plan(profile)
    profile_payload = profile.model_dump(mode="json")
    plan_payload = output.plan.model_dump(mode="json")
    bundle = ExtensionBundle(
        profile=profile.model_copy(deep=True),
        plan=output.plan.model_copy(deep=True),
        provenance=GenerationProvenance(
            provider=output.provider,
            model=output.model,
            profile_sha256=canonical_sha256(profile_payload),
            plan_sha256=canonical_sha256(plan_payload),
            prompt_sha256=output.prompt_sha256,
            api_calls=output.api_calls,
            elapsed_s=output.elapsed_s,
        ),
    )
    return GenerationResult(bundle=bundle, audit=audit_bundle(bundle), provider_output=output)


def _example_value(parameter: Any) -> Any:
    if parameter.enum:
        return parameter.enum[0]
    if parameter.value_type == "string":
        return f"<{parameter.name}>"
    if parameter.value_type == "integer":
        return int(parameter.minimum) if parameter.minimum is not None else 1
    if parameter.value_type == "number":
        return float(parameter.minimum) if parameter.minimum is not None else 1.0
    if parameter.value_type == "boolean":
        return False
    if parameter.value_type == "object":
        return {}
    if parameter.value_type == "array":
        return []
    return None


def derive_action_schema(bundle: ExtensionBundle) -> dict[str, Any]:
    action_types = list(dict.fromkeys(operation.action_type for operation in bundle.plan.operations))
    targets = list(dict.fromkeys(operation.target for operation in bundle.plan.operations))
    actions: list[dict[str, Any]] = []
    for operation in bundle.plan.operations:
        arguments = {
            parameter.name: {
                "type": parameter.value_type,
                "required": parameter.required,
                **({"enum": parameter.enum} if parameter.enum else {}),
                **({"minimum": parameter.minimum} if parameter.minimum is not None else {}),
                **({"maximum": parameter.maximum} if parameter.maximum is not None else {}),
                **({"description": parameter.description} if parameter.description else {}),
            }
            for parameter in operation.parameters
        }
        example_arguments = {
            parameter.name: _example_value(parameter)
            for parameter in operation.parameters
            if parameter.required
        }
        actions.append(
            {
                "name": operation.name,
                "description": operation.description,
                "evidence_refs": list(operation.evidence_refs),
                "params_schema": {"operation": operation.name, "arguments": arguments},
                "example": {
                    "action_type": operation.action_type,
                    "target": operation.target,
                    "params": {"operation": operation.name, "arguments": example_arguments},
                },
            }
        )
    return {
        "software": bundle.profile.display_name,
        "supported_action_types": action_types,
        "target": targets[0] if len(targets) == 1 else targets,
        "description": bundle.plan.summary,
        "actions": actions,
        "done_action": {
            "description": "Return this action only when the task is complete.",
            "example": {"action_type": "done", "target": "", "params": {}},
        },
        "limitations": list(bundle.plan.limitations),
    }


def _adapter_class_name(software_id: str) -> str:
    return "".join(part.capitalize() for part in software_id.split("_")) + "Adapter"


def generate_adapter_wrapper(bundle: ExtensionBundle) -> str:
    class_name = _adapter_class_name(bundle.profile.software_id)
    return (
        '"""Deterministic wrapper for a reviewed softwaregen extension bundle."""\n\n'
        "from pathlib import Path\n\n"
        "from asil.softwaregen import DeclarativeAdapter, load_extension_bundle\n\n\n"
        f"class {class_name}(DeclarativeAdapter):\n"
        "    def __init__(self, bundle_path: str | Path | None = None) -> None:\n"
        "        path = Path(bundle_path) if bundle_path is not None else Path(__file__).with_name(\"extension.json\")\n"
        "        super().__init__(load_extension_bundle(path))\n"
    )


def load_extension_bundle(path: str | Path) -> ExtensionBundle:
    return ExtensionBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_onboarding_profile(path: str | Path) -> OnboardingProfile:
    return OnboardingProfile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_extension_bundle(
    result: GenerationResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    artifact_names = ("extension.json", "action_schema.json", "adapter.py", "generation_report.json")
    existing = [name for name in artifact_names if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing softwaregen artifacts: {', '.join(existing)}")
    output.mkdir(parents=True, exist_ok=True)

    bundle_payload = result.bundle.model_dump(mode="json")
    artifact_content = {
        "extension.json": _json_text(bundle_payload),
        "action_schema.json": _json_text(derive_action_schema(result.bundle)),
        "adapter.py": generate_adapter_wrapper(result.bundle),
    }
    artifact_hashes = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in artifact_content.items()
    }
    report = {
        "ok": result.audit.ok,
        "audit": result.audit.model_dump(mode="json"),
        "provider": result.bundle.provenance.provider,
        "model": result.bundle.provenance.model,
        "api_calls": result.bundle.provenance.api_calls,
        "elapsed_s": result.bundle.provenance.elapsed_s,
        "profile_sha256": result.bundle.provenance.profile_sha256,
        "plan_sha256": result.bundle.provenance.plan_sha256,
        "prompt_sha256": result.bundle.provenance.prompt_sha256,
        "evidence_count": len(result.bundle.profile.evidence),
        "observation_view_count": len(result.bundle.plan.observation_views),
        "operation_count": len(result.bundle.plan.operations),
        "artifact_sha256": artifact_hashes,
        "provider_trace": sanitize_data(result.provider_output.trace),
    }
    for name, content in artifact_content.items():
        _atomic_write(output / name, content)
    _atomic_write(output / "generation_report.json", _json_text(report))
    return report
