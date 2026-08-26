"""Typed contracts for grounded, declarative ASIL software extensions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


IntegrationPattern = Literal["file_backed", "native_script", "service_api", "hybrid"]
QualificationTier = Literal["direct_declarative", "bridge_assisted", "out_of_scope"]
EvidenceKind = Literal["api_spec", "command_help", "sample_json", "source_code", "script", "file_format"]
ValueType = Literal["string", "integer", "number", "boolean", "object", "array"]
ASILActionType = Literal["set_value", "invoke_function", "modify_file", "api_call", "navigate", "batch"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    kind: EvidenceKind
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    sample: Any | None = None


class RuntimeBindings(StrictModel):
    base_url: str = ""
    base_url_env: str = Field(default="", pattern=r"^$|^[a-zA-Z_][a-zA-Z0-9_]*$")
    allowed_hosts: list[str] = Field(default_factory=list)
    allowed_executables: list[str] = Field(default_factory=list)
    filesystem_root: str = "."
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout_s: float = Field(default=10.0, gt=0.0, le=120.0)


class OnboardingProfile(StrictModel):
    software_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=1)
    version: str = ""
    integration_pattern: IntegrationPattern
    description: str = Field(min_length=1)
    evidence: list[EvidenceItem] = Field(min_length=1)
    runtime: RuntimeBindings
    requirements: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class QualificationReport(StrictModel):
    software_id: str
    eligible: bool
    tier: QualificationTier
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    available_observation_transports: list[str] = Field(default_factory=list)
    available_action_transports: list[str] = Field(default_factory=list)
    required_human_work: list[str] = Field(default_factory=list)
    claim_boundary: str


class ReferenceApplication(StrictModel):
    software_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=1)
    pattern: Literal["file_backed", "native_script", "service_api"]
    read_paths: list[str] = Field(min_length=1)
    action_paths: list[str] = Field(min_length=1)
    observation_source: str = Field(min_length=1)
    deployment_tier: Literal["direct_declarative", "bridge_assisted"]
    adapter_path: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class ReferenceCatalog(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    claim: str = Field(min_length=1)
    applications: list[ReferenceApplication] = Field(min_length=1)


class ProbeEvidenceSummary(StrictModel):
    environment: Literal["host", "docker"]
    reports: int = Field(ge=1)
    passed: int = Field(ge=0)
    action_reports: int = Field(ge=0)
    validated_state_changes: int = Field(ge=0)
    element_counts: list[int] = Field(default_factory=list)


class DeploymentEvidenceReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    ready: bool
    software_id: str
    qualification: QualificationReport
    provider: str
    model: str
    api_calls: int = Field(ge=0)
    generation_elapsed_s: float = Field(ge=0.0)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: dict[str, str]
    artifact_file_hashes_match: bool
    audit: AuditReport
    evidence_reference_coverage: dict[str, Any]
    observation_view_count: int = Field(ge=1)
    operation_count: int = Field(ge=1)
    host: ProbeEvidenceSummary
    docker: ProbeEvidenceSummary
    generated_stages: list[str]
    reviewed_stages: list[str]
    limitations: list[str]
    claim_boundary: str


class HTTPProbe(StrictModel):
    transport: Literal["http_json"]
    method: Literal["GET"] = "GET"
    path: str = Field(min_length=1)
    query: dict[str, Any] = Field(default_factory=dict)
    items_pointer: str = ""
    evidence_refs: list[str] = Field(min_length=1)


class CommandProbe(StrictModel):
    transport: Literal["command_json"]
    argv: list[str] = Field(min_length=1)
    items_pointer: str = ""
    evidence_refs: list[str] = Field(min_length=1)


class JSONFileProbe(StrictModel):
    transport: Literal["json_file"]
    path: str = Field(min_length=1)
    items_pointer: str = ""
    evidence_refs: list[str] = Field(min_length=1)


ObservationProbe = Annotated[HTTPProbe | CommandProbe | JSONFileProbe, Field(discriminator="transport")]


class ElementMapping(StrictModel):
    id_prefix: str = ""
    id_pointer: str = Field(min_length=1)
    type: str = Field(min_length=1)
    label_pointer: str = ""
    value_fields: dict[str, str] = Field(default_factory=dict)
    metadata_fields: dict[str, str] = Field(default_factory=dict)
    editable: bool = True
    actions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)


class ObservationView(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    description: str = Field(min_length=1)
    probe: ObservationProbe
    element: ElementMapping
    evidence_refs: list[str] = Field(min_length=1)


class ParameterSpec(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    value_type: ValueType
    required: bool = True
    description: str = ""
    enum: list[Any] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None


class HTTPActionRequest(StrictModel):
    transport: Literal["http_json"]
    method: Literal["POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(min_length=1)
    query: dict[str, Any] = Field(default_factory=dict)
    body: Any = Field(default_factory=dict)


class CommandActionRequest(StrictModel):
    transport: Literal["command_json"]
    argv: list[str] = Field(min_length=1)
    stdin: Any | None = None


ActionRequest = Annotated[HTTPActionRequest | CommandActionRequest, Field(discriminator="transport")]


class OperationSpec(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    description: str = Field(min_length=1)
    action_type: ASILActionType
    target: str = Field(min_length=1)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    request: ActionRequest
    evidence_refs: list[str] = Field(min_length=1)


class InterfacePlan(StrictModel):
    summary: str = Field(min_length=1)
    observation_views: list[ObservationView] = Field(min_length=1)
    operations: list[OperationSpec] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class GenerationProvenance(StrictModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")
    api_calls: int = Field(default=0, ge=0)
    elapsed_s: float = Field(default=0.0, ge=0.0)


class ExtensionBundle(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    profile: OnboardingProfile
    plan: InterfacePlan
    provenance: GenerationProvenance


class AuditFinding(StrictModel):
    severity: Literal["error", "warning"]
    code: str = Field(min_length=1)
    location: str = ""
    message: str = Field(min_length=1)


class AuditReport(StrictModel):
    findings: list[AuditFinding] = Field(default_factory=list)

    @computed_field
    @property
    def ok(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @computed_field
    @property
    def error_count(self) -> int:
        return sum(finding.severity == "error" for finding in self.findings)

    @computed_field
    @property
    def warning_count(self) -> int:
        return sum(finding.severity == "warning" for finding in self.findings)
