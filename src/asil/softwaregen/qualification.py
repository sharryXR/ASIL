"""Conservative applicability checks for assisted ASIL onboarding."""

from __future__ import annotations

from pathlib import Path

from asil.softwaregen.models import OnboardingProfile, QualificationReport, ReferenceCatalog


DEFAULT_REFERENCE_CATALOG = Path(__file__).with_name("reference_catalog.json")
_FIXED_REVIEW_STAGES = [
    "interface evidence and runtime permission review",
    "task and evaluator design",
    "GUI synchronization and rendering when required",
    "application-specific and unsupported semantics",
]
_CLAIM_BOUNDARY = (
    "Assisted onboarding applies to evidenced open interfaces; qualification does not prove a complete "
    "application integration and does not automate tasks, evaluators, rendering, or opaque/perceptual semantics."
)


def load_reference_catalog(path: str | Path | None = None) -> ReferenceCatalog:
    catalog_path = Path(path) if path is not None else DEFAULT_REFERENCE_CATALOG
    catalog = ReferenceCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    identifiers = [application.software_id for application in catalog.applications]
    duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicates:
        raise ValueError(f"Reference catalog contains duplicate software_id values: {', '.join(duplicates)}")
    return catalog


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def qualify_profile(profile: OnboardingProfile) -> QualificationReport:
    evidence = list(profile.evidence)
    api_read = any(item.kind == "api_spec" and _contains_any(item.locator, ("get ", "list", "read")) for item in evidence)
    api_action = any(
        item.kind == "api_spec" and _contains_any(item.locator, ("post ", "put ", "patch ", "delete ", "create", "update"))
        for item in evidence
    )
    command_items = [item for item in evidence if item.kind in {"command_help", "script"}]
    command_evidence = bool(command_items)
    command_ready = command_evidence and bool(profile.runtime.allowed_executables)
    command_read = command_ready and any(
        _contains_any(
            f"{item.locator} {item.excerpt}",
            ("observe", "list", "get ", "read", "dump", "state"),
        )
        for item in command_items
    )
    command_action = command_ready and any(
        _contains_any(
            f"{item.locator} {item.excerpt}",
            ("set", "create", "update", "delete", "edit", "action", "operation", "accepts"),
        )
        for item in command_items
    )
    file_evidence = [item for item in evidence if item.kind in {"file_format", "sample_json"}]
    json_file = any(
        item.kind == "sample_json"
        or item.sample is not None
        or item.locator.lower().endswith((".json", ".ipynb"))
        for item in file_evidence
    )
    open_file = bool(file_evidence)
    file_action = any(
        _contains_any(f"{item.locator} {item.excerpt}", ("edit", "editable", "update", "mutate", "write"))
        for item in file_evidence
    )

    observations: list[str] = []
    actions: list[str] = []
    reasons: list[str] = []
    blockers: list[str] = []

    if api_read and profile.runtime.allowed_hosts and (profile.runtime.base_url or profile.runtime.base_url_env):
        observations.append("http_json")
        reasons.append("profile evidences a host-allowlisted JSON API read path")
    if api_action and profile.runtime.allowed_hosts and (profile.runtime.base_url or profile.runtime.base_url_env):
        actions.append("http_json")
        reasons.append("profile evidences a host-allowlisted semantic API action path")
    if command_read:
        observations.append("command_json")
        reasons.append("profile evidences a direct structured-output script or command with an executable allowlist")
    if command_action:
        actions.append("command_json")
        reasons.append("profile evidences a direct semantic command or script action with an executable allowlist")
    if json_file:
        observations.append("json_file")
        reasons.append("profile evidences a structured JSON file below the reviewed filesystem root")

    if observations and actions:
        tier = "direct_declarative"
        eligible = True
        required = list(_FIXED_REVIEW_STAGES)
    elif open_file and file_action:
        tier = "bridge_assisted"
        eligible = True
        observations = ["command_json"]
        actions = ["command_json"]
        reasons.append("an open non-JSON format can be normalized through a reviewed bridge")
        required = ["reviewed structured-output parser or converter", *_FIXED_REVIEW_STAGES]
    else:
        tier = "out_of_scope"
        eligible = False
        required = list(_FIXED_REVIEW_STAGES)
        if not observations and not (open_file and file_action):
            blockers.append("no evidenced open read path")
        if not actions and not (open_file and file_action):
            blockers.append("no evidenced semantic action path")
        if api_read and not profile.runtime.allowed_hosts:
            blockers.append("API evidence lacks an immutable allowed-host binding")
        if command_evidence and not profile.runtime.allowed_executables:
            blockers.append("command or script evidence lacks an executable allowlist")

    return QualificationReport(
        software_id=profile.software_id,
        eligible=eligible,
        tier=tier,
        reasons=list(dict.fromkeys(reasons)),
        blockers=list(dict.fromkeys(blockers)),
        available_observation_transports=list(dict.fromkeys(observations)),
        available_action_transports=list(dict.fromkeys(actions)),
        required_human_work=list(dict.fromkeys(required)),
        claim_boundary=_CLAIM_BOUNDARY,
    )
