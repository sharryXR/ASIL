"""Grounded LLM-assisted onboarding for open software interfaces."""

from asil.softwaregen.audit import audit_bundle
from asil.softwaregen.models import (
    AuditFinding,
    AuditReport,
    DeploymentEvidenceReport,
    ExtensionBundle,
    InterfacePlan,
    OnboardingProfile,
    ProbeEvidenceSummary,
    QualificationReport,
    ReferenceApplication,
    ReferenceCatalog,
)
from asil.softwaregen.runtime import (
    DeclarativeAdapter,
    render_path_template,
    render_template,
    resolve_json_pointer,
)
from asil.softwaregen.generator import (
    GenerationResult,
    derive_action_schema,
    generate_adapter_wrapper,
    generate_extension,
    load_extension_bundle,
    load_onboarding_profile,
    write_extension_bundle,
)
from asil.softwaregen.provider import (
    DeterministicSoftwareGenProvider,
    OpenAISoftwareGenProvider,
    ProviderOutput,
    canonical_sha256,
    sanitize_data,
)
from asil.softwaregen.cli import main as softwaregen_main
from asil.softwaregen.validation import (
    build_docker_probe_command,
    docker_probe_extension,
    observation_state_sha256,
    probe_extension,
)
from asil.softwaregen.qualification import load_reference_catalog, qualify_profile
from asil.softwaregen.evidence import build_deployment_evidence_report

__all__ = [
    "AuditFinding",
    "AuditReport",
    "DeclarativeAdapter",
    "DeploymentEvidenceReport",
    "DeterministicSoftwareGenProvider",
    "ExtensionBundle",
    "GenerationResult",
    "InterfacePlan",
    "OnboardingProfile",
    "OpenAISoftwareGenProvider",
    "ProviderOutput",
    "ProbeEvidenceSummary",
    "QualificationReport",
    "ReferenceApplication",
    "ReferenceCatalog",
    "audit_bundle",
    "build_docker_probe_command",
    "build_deployment_evidence_report",
    "canonical_sha256",
    "derive_action_schema",
    "docker_probe_extension",
    "generate_adapter_wrapper",
    "generate_extension",
    "load_extension_bundle",
    "load_reference_catalog",
    "load_onboarding_profile",
    "observation_state_sha256",
    "probe_extension",
    "qualify_profile",
    "render_template",
    "render_path_template",
    "resolve_json_pointer",
    "sanitize_data",
    "softwaregen_main",
    "write_extension_bundle",
]
