"""Static grounding and safety audit for declarative software extensions."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit

from asil.softwaregen.models import (
    AuditFinding,
    AuditReport,
    CommandActionRequest,
    CommandProbe,
    ExtensionBundle,
    HTTPActionRequest,
    HTTPProbe,
    JSONFileProbe,
)


_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_ENV_PLACEHOLDER = re.compile(r"\$\{ENV:([a-zA-Z_][a-zA-Z0-9_]*)\}")
_SHELL_EXECUTABLES = {"ash", "bash", "csh", "dash", "fish", "ksh", "sh", "tcsh", "zsh"}
_SENSITIVE_HEADERS = {"authorization", "cookie", "proxy-authorization", "x-api-key", "api-key"}


def _finding(code: str, location: str, message: str, *, severity: str = "error") -> AuditFinding:
    return AuditFinding(severity=severity, code=code, location=location, message=message)


def _duplicates(values: Iterable[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _unsafe_relative_path(path: str, *, require_leading_slash: bool) -> bool:
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        return True
    clean_path = parsed.path
    if require_leading_slash and not clean_path.startswith("/"):
        return True
    return ".." in PurePosixPath(clean_path).parts


def _audit_command(
    argv: list[str],
    *,
    allowed_executables: set[str],
    location: str,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    executable = os.path.basename(argv[0]) if argv else ""
    if executable not in allowed_executables:
        findings.append(
            _finding(
                "executable_not_allowed",
                location,
                f"Executable `{executable}` is not in runtime.allowed_executables.",
            )
        )
    if executable in _SHELL_EXECUTABLES:
        findings.append(
            _finding("shell_execution_forbidden", location, "Shell executables are not supported by softwaregen.")
        )
    return findings


def _audit_evidence_refs(
    refs: list[str],
    *,
    evidence_ids: set[str],
    location: str,
) -> list[AuditFinding]:
    return [
        _finding("unknown_evidence", location, f"Evidence reference `{ref}` is absent from the profile.")
        for ref in refs
        if ref not in evidence_ids
    ]


def audit_bundle(bundle: ExtensionBundle) -> AuditReport:
    findings: list[AuditFinding] = []
    profile = bundle.profile
    plan = bundle.plan
    evidence_ids = {evidence.id for evidence in profile.evidence}

    for duplicate in sorted(_duplicates(evidence.id for evidence in profile.evidence)):
        findings.append(_finding("duplicate_evidence", "profile.evidence", f"Duplicate evidence ID `{duplicate}`."))

    base_url = urlsplit(profile.runtime.base_url)
    if profile.runtime.base_url:
        if base_url.scheme not in {"http", "https"} or not base_url.hostname:
            findings.append(_finding("invalid_base_url", "profile.runtime.base_url", "Base URL must be HTTP(S)."))
        elif base_url.hostname not in set(profile.runtime.allowed_hosts):
            findings.append(
                _finding(
                    "host_not_allowed",
                    "profile.runtime.base_url",
                    f"Host `{base_url.hostname}` is absent from runtime.allowed_hosts.",
                )
            )

    for header_name, header_value in profile.runtime.headers.items():
        if header_name.lower() in _SENSITIVE_HEADERS and not _ENV_PLACEHOLDER.search(header_value):
            findings.append(
                _finding(
                    "literal_sensitive_header",
                    f"profile.runtime.headers.{header_name}",
                    f"Sensitive header `{header_name}` must reference a runtime environment variable.",
                )
            )

    view_ids = [view.id for view in plan.observation_views]
    for duplicate in sorted(_duplicates(view_ids)):
        findings.append(_finding("duplicate_view", "plan.observation_views", f"Duplicate view ID `{duplicate}`."))

    operation_ids = [operation.name for operation in plan.operations]
    for duplicate in sorted(_duplicates(operation_ids)):
        findings.append(_finding("duplicate_operation", "plan.operations", f"Duplicate operation `{duplicate}`."))
    operation_id_set = set(operation_ids)

    allowed_executables = {os.path.basename(value) for value in profile.runtime.allowed_executables}
    for index, view in enumerate(plan.observation_views):
        location = f"plan.observation_views[{index}]"
        findings.extend(_audit_evidence_refs(view.evidence_refs, evidence_ids=evidence_ids, location=location))
        findings.extend(
            _audit_evidence_refs(
                view.probe.evidence_refs,
                evidence_ids=evidence_ids,
                location=f"{location}.probe",
            )
        )
        findings.extend(
            _audit_evidence_refs(
                view.element.evidence_refs,
                evidence_ids=evidence_ids,
                location=f"{location}.element",
            )
        )
        for action_name in view.element.actions:
            if action_name not in operation_id_set:
                findings.append(
                    _finding(
                        "unknown_element_action",
                        f"{location}.element.actions",
                        f"Element action `{action_name}` has no matching operation.",
                    )
                )

        if isinstance(view.probe, HTTPProbe):
            if not (profile.runtime.base_url or profile.runtime.base_url_env):
                findings.append(_finding("missing_base_url", f"{location}.probe", "HTTP probe requires a base URL."))
            if _unsafe_relative_path(view.probe.path, require_leading_slash=True):
                findings.append(
                    _finding("unsafe_http_path", f"{location}.probe.path", f"Unsafe HTTP path `{view.probe.path}`.")
                )
        elif isinstance(view.probe, CommandProbe):
            findings.extend(
                _audit_command(
                    view.probe.argv,
                    allowed_executables=allowed_executables,
                    location=f"{location}.probe.argv",
                )
            )
        elif isinstance(view.probe, JSONFileProbe):
            if _unsafe_relative_path(view.probe.path, require_leading_slash=False) or PurePosixPath(view.probe.path).is_absolute():
                findings.append(
                    _finding("unsafe_file_path", f"{location}.probe.path", f"Unsafe JSON file path `{view.probe.path}`.")
                )

    for index, operation in enumerate(plan.operations):
        location = f"plan.operations[{index}]"
        findings.extend(_audit_evidence_refs(operation.evidence_refs, evidence_ids=evidence_ids, location=location))
        parameter_names = [parameter.name for parameter in operation.parameters]
        for duplicate in sorted(_duplicates(parameter_names)):
            findings.append(
                _finding("duplicate_parameter", f"{location}.parameters", f"Duplicate parameter `{duplicate}`.")
            )

        request = operation.request
        if isinstance(request, HTTPActionRequest):
            if not (profile.runtime.base_url or profile.runtime.base_url_env):
                findings.append(_finding("missing_base_url", f"{location}.request", "HTTP action requires a base URL."))
            if _unsafe_relative_path(request.path, require_leading_slash=True):
                findings.append(
                    _finding("unsafe_http_path", f"{location}.request.path", f"Unsafe HTTP path `{request.path}`.")
                )
        elif isinstance(request, CommandActionRequest):
            findings.extend(
                _audit_command(
                    request.argv,
                    allowed_executables=allowed_executables,
                    location=f"{location}.request.argv",
                )
            )

        request_payload = request.model_dump(mode="json")
        template_strings = list(_walk_strings(request_payload))
        placeholders: set[str] = set()
        malformed = False
        for text in template_strings:
            matches = list(_PLACEHOLDER.finditer(text))
            placeholders.update(match.group(1) for match in matches)
            residue = _PLACEHOLDER.sub("", text)
            malformed = malformed or "${" in residue
        declared = set(parameter_names)
        for placeholder in sorted(placeholders - declared):
            findings.append(
                _finding(
                    "unknown_placeholder",
                    f"{location}.request",
                    f"Placeholder `{placeholder}` has no declared parameter.",
                )
            )
        for parameter in sorted(declared - placeholders):
            findings.append(
                _finding(
                    "unused_parameter",
                    f"{location}.parameters",
                    f"Parameter `{parameter}` is never used by the request template.",
                )
            )
        if malformed:
            findings.append(
                _finding("malformed_placeholder", f"{location}.request", "Request contains a malformed placeholder.")
            )

    return AuditReport(findings=findings)
