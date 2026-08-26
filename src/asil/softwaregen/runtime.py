"""Safe interpreter for declarative ASIL observation and action bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from asil.adapter import ASILAdapter
from asil.protocol import Action, Element, Observation
from asil.softwaregen.audit import audit_bundle
from asil.softwaregen.models import (
    CommandActionRequest,
    CommandProbe,
    ExtensionBundle,
    HTTPActionRequest,
    HTTPProbe,
    JSONFileProbe,
    OperationSpec,
    ParameterSpec,
)


_EXACT_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_ENV_PLACEHOLDER = re.compile(r"\$\{ENV:([a-zA-Z_][a-zA-Z0-9_]*)\}")


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer, raising KeyError for missing segments."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON Pointer must be empty or start with '/': {pointer}")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise KeyError(part)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"JSON Pointer segment `{part}` not found in `{pointer}`") from exc
    return current


def render_template(value: Any, params: dict[str, Any]) -> Any:
    """Recursively substitute `${parameter}` while preserving exact-value types."""
    if isinstance(value, str):
        exact = _EXACT_PLACEHOLDER.fullmatch(value)
        if exact:
            name = exact.group(1)
            if name not in params:
                raise ValueError(f"Template parameter `{name}` is missing.")
            return params[name]

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in params:
                raise ValueError(f"Template parameter `{name}` is missing.")
            return str(params[name])

        rendered = _PLACEHOLDER.sub(replace, value)
        if "${" in rendered:
            raise ValueError(f"Template contains an unsupported placeholder: {value}")
        return rendered
    if isinstance(value, list):
        return [render_template(child, params) for child in value]
    if isinstance(value, dict):
        return {key: render_template(child, params) for key, child in value.items()}
    return value


def render_path_template(template: str, params: dict[str, Any]) -> str:
    """Render URL path parameters as single encoded segments."""
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            raise ValueError(f"Template parameter `{name}` is missing.")
        value = params[name]
        if isinstance(value, (dict, list)):
            raise ValueError(f"Path parameter `{name}` must be scalar.")
        text = str(value)
        if text in {".", ".."}:
            raise ValueError(f"Path parameter `{name}` contains a traversal segment.")
        if "/" in text or "\\" in text:
            raise ValueError(f"Path parameter `{name}` contains path separators.")
        return quote(text, safe="-._~")

    rendered = _PLACEHOLDER.sub(replace, template)
    if "${" in rendered:
        raise ValueError(f"Path template contains an unsupported placeholder: {template}")
    return rendered


def _render_env_template(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise RuntimeError(f"Required runtime environment variable `{name}` is not set.")
        return resolved

    rendered = _ENV_PLACEHOLDER.sub(replace, value)
    if "${ENV:" in rendered:
        raise RuntimeError(f"Malformed environment placeholder in runtime value: {value}")
    return rendered


def _value_matches_type(value: Any, value_type: str) -> bool:
    if value_type == "string":
        return isinstance(value, str)
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type == "object":
        return isinstance(value, dict)
    if value_type == "array":
        return isinstance(value, list)
    return False


def _validate_parameter(parameter: ParameterSpec, value: Any) -> str | None:
    if not _value_matches_type(value, parameter.value_type):
        return f"Parameter `{parameter.name}` must have type {parameter.value_type}."
    if parameter.enum and value not in parameter.enum:
        return f"Parameter `{parameter.name}` must be one of {parameter.enum}."
    if parameter.minimum is not None and isinstance(value, (int, float)) and value < parameter.minimum:
        return f"Parameter `{parameter.name}` must be >= {parameter.minimum}."
    if parameter.maximum is not None and isinstance(value, (int, float)) and value > parameter.maximum:
        return f"Parameter `{parameter.name}` must be <= {parameter.maximum}."
    return None


class DeclarativeAdapter(ASILAdapter):
    """Interpret a zero-audit-error extension bundle without generated code."""

    supported_action_types = [
        "set_value",
        "invoke_function",
        "modify_file",
        "api_call",
        "navigate",
        "batch",
    ]

    def __init__(self, bundle: ExtensionBundle, *, session: requests.Session | None = None) -> None:
        report = audit_bundle(bundle)
        if not report.ok:
            codes = ", ".join(finding.code for finding in report.findings if finding.severity == "error")
            raise ValueError(f"Extension bundle failed static audit: {codes}")
        self.bundle = bundle
        self.app_name = bundle.profile.display_name
        self.app_version = bundle.profile.version
        self._session = session or requests.Session()
        self._operations = {operation.name: operation for operation in bundle.plan.operations}
        self._last_action_response: Any = None

    @property
    def source_path(self) -> Path | None:
        for view in self.bundle.plan.observation_views:
            if isinstance(view.probe, JSONFileProbe):
                return self._resolve_file(view.probe.path)
        return None

    def _headers(self) -> dict[str, str]:
        return {
            key: _render_env_template(value)
            for key, value in self.bundle.profile.runtime.headers.items()
        }

    def _resolve_file(self, relative_path: str) -> Path:
        root = Path(self.bundle.profile.runtime.filesystem_root).expanduser().resolve()
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"File path escapes approved root: {relative_path}")
        return candidate

    def _http_request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: Any | None = None,
    ) -> Any:
        runtime = self.bundle.profile.runtime
        base_url = os.environ.get(runtime.base_url_env, "") if runtime.base_url_env else ""
        base_url = base_url or runtime.base_url
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("The resolved runtime base URL is not valid HTTP(S).")
        if parsed.hostname not in set(runtime.allowed_hosts):
            raise ValueError(f"Resolved host `{parsed.hostname}` is not an approved runtime host.")
        url = base_url.rstrip("/") + "/" + path.lstrip("/")
        response = self._session.request(
            method,
            url,
            params=query or None,
            json=body if method != "GET" else None,
            headers=self._headers(),
            timeout=runtime.request_timeout_s,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def _run_command(self, argv: list[str], *, stdin: Any | None = None) -> Any:
        runtime = self.bundle.profile.runtime
        input_text = None if stdin is None else json.dumps(stdin, ensure_ascii=False)
        result = subprocess.run(
            argv,
            cwd=Path(runtime.filesystem_root).expanduser().resolve(),
            input=input_text,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
            timeout=runtime.request_timeout_s,
        )
        if not result.stdout.strip():
            return None
        return json.loads(result.stdout)

    def _read_probe(self, probe: HTTPProbe | CommandProbe | JSONFileProbe) -> tuple[Any, str]:
        if isinstance(probe, HTTPProbe):
            return self._http_request(probe.method, probe.path, query=probe.query), "rest_api"
        if isinstance(probe, CommandProbe):
            return self._run_command(probe.argv), "script_api"
        if isinstance(probe, JSONFileProbe):
            return json.loads(self._resolve_file(probe.path).read_text(encoding="utf-8")), "file_parse"
        raise TypeError(f"Unsupported observation probe: {type(probe).__name__}")

    def observe(self) -> Observation:
        elements: list[Element] = []
        sources: list[str] = []
        view_counts: list[str] = []
        for view in self.bundle.plan.observation_views:
            payload, source = self._read_probe(view.probe)
            sources.append(source)
            selected = resolve_json_pointer(payload, view.probe.items_pointer)
            items = selected if isinstance(selected, list) else [selected]
            view_counts.append(f"{view.id}={len(items)}")
            for item in items:
                item_id = resolve_json_pointer(item, view.element.id_pointer)
                if item_id is None or str(item_id) == "":
                    raise ValueError(f"View `{view.id}` produced an empty stable element ID.")
                label = (
                    resolve_json_pointer(item, view.element.label_pointer)
                    if view.element.label_pointer
                    else item_id
                )
                value = {
                    key: resolve_json_pointer(item, pointer)
                    for key, pointer in view.element.value_fields.items()
                }
                metadata = {
                    key: resolve_json_pointer(item, pointer)
                    for key, pointer in view.element.metadata_fields.items()
                }
                elements.append(
                    Element(
                        id=f"{view.element.id_prefix}{item_id}",
                        type=view.element.type,
                        label=str(label),
                        value=value,
                        editable=view.element.editable,
                        actions=list(view.element.actions),
                        metadata={"view_id": view.id, **metadata},
                    )
                )

        unique_sources = list(dict.fromkeys(sources))
        observation_source = unique_sources[0] if len(unique_sources) == 1 else "hybrid_api"
        first_view = self.bundle.plan.observation_views[0].id
        return self._build_observation(
            source=observation_source,
            elements=elements,
            app_state={"current_view": first_view},
            navigation={
                "available_views": [
                    {"id": view.id, "label": view.id, "description": view.description}
                    for view in self.bundle.plan.observation_views
                ],
                "current_path": first_view,
                "reachable_from_here": [view.id for view in self.bundle.plan.observation_views],
            },
            data_summary=f"{self.app_name}: {len(elements)} elements ({', '.join(view_counts)})",
        )

    def _validated_operation(self, action: Action) -> tuple[OperationSpec, dict[str, Any]]:
        if set(action.params) - {"operation", "arguments"}:
            raise ValueError("Action params may contain only `operation` and `arguments`.")
        operation_name = action.params.get("operation")
        if not isinstance(operation_name, str) or operation_name not in self._operations:
            raise ValueError(f"Unknown semantic operation: {operation_name!r}")
        operation = self._operations[operation_name]
        if action.action_type != operation.action_type:
            raise ValueError(
                f"Operation `{operation_name}` requires action_type `{operation.action_type}`."
            )
        if action.target != operation.target:
            raise ValueError(f"Operation `{operation_name}` requires target `{operation.target}`.")
        arguments = action.params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Action `arguments` must be an object.")
        declared = {parameter.name: parameter for parameter in operation.parameters}
        unknown = sorted(set(arguments) - set(declared))
        if unknown:
            raise ValueError(f"Unknown action argument(s): {', '.join(unknown)}")
        missing = sorted(
            parameter.name
            for parameter in operation.parameters
            if parameter.required and parameter.name not in arguments
        )
        if missing:
            raise ValueError(f"required action argument(s) missing: {', '.join(missing)}")
        for name, value in arguments.items():
            error = _validate_parameter(declared[name], value)
            if error:
                raise ValueError(error)
        return operation, arguments

    def validate_action(self, action: Action) -> bool:
        try:
            self._validated_operation(action)
        except (TypeError, ValueError):
            return False
        return True

    def execute(self, action: Action) -> Observation:
        operation, arguments = self._validated_operation(action)
        request = operation.request
        if isinstance(request, HTTPActionRequest):
            self._last_action_response = self._http_request(
                request.method,
                render_path_template(request.path, arguments),
                query=render_template(request.query, arguments),
                body=render_template(request.body, arguments),
            )
        elif isinstance(request, CommandActionRequest):
            self._last_action_response = self._run_command(
                render_template(request.argv, arguments),
                stdin=render_template(request.stdin, arguments),
            )
        else:
            raise TypeError(f"Unsupported action request: {type(request).__name__}")
        return self.observe()
