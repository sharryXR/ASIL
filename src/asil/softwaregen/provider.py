"""Grounded structured-output providers for ASIL software onboarding."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Protocol

from dotenv import load_dotenv

from asil.softwaregen.models import InterfacePlan, OnboardingProfile


_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "set-cookie",
    "token",
    "access_token",
    "api_key",
    "apikey",
    "password",
    "secret",
}
_BEARER_VALUE = re.compile(r"(?i)\b(?:bearer|token)\s+[^\s,;]+")


def canonical_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sanitize_data(value: Any, *, key: str = "") -> Any:
    """Redact secret-bearing keys and common inline authorization values."""
    if key.lower() in _SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(child_key): sanitize_data(child, key=str(child_key)) for child_key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_data(child) for child in value]
    if isinstance(value, tuple):
        return [sanitize_data(child) for child in value]
    if isinstance(value, str):
        return _BEARER_VALUE.sub("<redacted>", value)
    return value


def _profile_prompt_payload(profile: OnboardingProfile) -> dict[str, Any]:
    payload = profile.model_dump(mode="json")
    headers = payload.get("runtime", {}).get("headers", {})
    payload["runtime"]["header_names"] = sorted(headers)
    payload["runtime"]["headers"] = "<environment-backed>"
    return sanitize_data(payload)


def build_generation_prompt(profile: OnboardingProfile) -> tuple[str, str]:
    instructions = (
        "You compile documented open software interfaces into an ASIL InterfacePlan. "
        "Use only the supplied evidence. Every observation view, probe, element mapping, and operation "
        "must cite relevant evidence IDs. Prefer stable task-relevant state and the deepest feasible open "
        "interface. Generate semantic non-GUI operations, never coordinates, shell commands, package "
        "installation, arbitrary Python, or benchmark evaluator logic. HTTP paths must be relative to the "
        "provided base URL. Command argv entries must use only allowed executables. If a capability is not "
        "documented, put it in limitations instead of inventing it. Return a valid InterfacePlan."
    )
    prompt_payload = {
        "profile": _profile_prompt_payload(profile),
        "contract_notes": {
            "observation": "Map JSON collections to stable ASIL elements with JSON Pointers.",
            "actions": "Each operation uses typed parameters and one evidenced HTTP or direct-command request.",
            "templates": "Action request templates may use only ${parameter_name} placeholders.",
        },
        "output_schema": InterfacePlan.model_json_schema(),
        "exact_field_rules": [
            "Use summary, observation_views, operations, and limitations as the only top-level keys.",
            "Use id, probe, element, and evidence_refs on every observation view; do not use name, mapping, or evidence aliases.",
            "Every probe and action request must include transport=http_json or transport=command_json.",
            "Use value_type on parameters; do not use a field named type for parameter types.",
            "Use plain strings in limitations.",
        ],
    }
    return instructions, json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    texts: list[str] = []
    output = getattr(response, "output", []) or []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content", [])
        for part in content or []:
            text = getattr(part, "text", None)
            if isinstance(part, dict):
                text = part.get("text", text)
            if text:
                texts.append(str(text))
    return "\n".join(texts)


def _pointer_alias(value: Any) -> Any:
    if isinstance(value, str) and value and not value.startswith("/"):
        return f"/{value}"
    return value


def normalize_interface_plan_payload(value: Any, profile: OnboardingProfile) -> Any:
    """Normalize a small set of common gateway/model aliases before strict validation."""
    if not isinstance(value, dict):
        return value
    payload = json.loads(json.dumps(value))
    payload.setdefault("summary", payload.get("description") or f"Generated interface plan for {profile.display_name}.")
    limitations = payload.get("limitations", [])
    if isinstance(limitations, list):
        payload["limitations"] = [
            item.get("description", "") if isinstance(item, dict) else item
            for item in limitations
        ]

    for view in payload.get("observation_views", []) if isinstance(payload.get("observation_views"), list) else []:
        if not isinstance(view, dict):
            continue
        view.setdefault("id", view.get("name"))
        view.setdefault("evidence_refs", view.get("evidence", []))
        probe = view.get("probe")
        if isinstance(probe, dict):
            kind = str(probe.get("kind", ""))
            if "transport" not in probe:
                if kind in {"http_request", "http", "rest_api"} or "method" in probe:
                    probe["transport"] = "http_json"
                elif kind in {"command", "command_json", "script"} or "argv" in probe:
                    probe["transport"] = "command_json"
                elif kind in {"json_file", "file"}:
                    probe["transport"] = "json_file"
            probe.setdefault("evidence_refs", probe.get("evidence", []))
            if "items_pointer" not in probe and "items_path" in probe:
                probe["items_pointer"] = _pointer_alias(probe["items_path"])
        mapping = view.get("element") or view.get("element_mapping") or view.get("mapping")
        if isinstance(mapping, dict):
            view["element"] = mapping
            mapping.setdefault("id_prefix", mapping.get("prefix", ""))
            if "id_pointer" not in mapping and "id_field" in mapping:
                mapping["id_pointer"] = _pointer_alias(mapping["id_field"])
            if "label_pointer" not in mapping and "label_field" in mapping:
                mapping["label_pointer"] = _pointer_alias(mapping["label_field"])
            mapping.setdefault("type", mapping.get("element_type", "item"))
            mapping.setdefault("evidence_refs", mapping.get("evidence", view.get("evidence_refs", [])))
            fields = mapping.get("value_fields")
            if isinstance(fields, list):
                mapping["value_fields"] = {str(field): _pointer_alias(field) for field in fields}
            elif isinstance(fields, dict):
                mapping["value_fields"] = {
                    str(key): _pointer_alias(pointer)
                    for key, pointer in fields.items()
                }

    for operation in payload.get("operations", []) if isinstance(payload.get("operations"), list) else []:
        if not isinstance(operation, dict):
            continue
        operation.setdefault("evidence_refs", operation.get("evidence", []))
        operation.setdefault("target", profile.software_id)
        request = operation.get("request")
        if isinstance(request, dict):
            kind = str(request.get("kind", ""))
            if "transport" not in request:
                if kind in {"http_request", "http", "rest_api"} or "method" in request:
                    request["transport"] = "http_json"
                elif kind in {"command", "command_json", "script"} or "argv" in request:
                    request["transport"] = "command_json"
            operation.setdefault(
                "action_type",
                "api_call" if request.get("transport") == "http_json" else "invoke_function",
            )
        for parameter in operation.get("parameters", []) if isinstance(operation.get("parameters"), list) else []:
            if isinstance(parameter, dict):
                parameter.setdefault("value_type", parameter.get("type"))
    return payload


@dataclass(frozen=True)
class ProviderOutput:
    plan: InterfacePlan
    provider: str
    model: str
    prompt_sha256: str
    api_calls: int
    elapsed_s: float
    trace: dict[str, Any] = field(default_factory=dict)


class SoftwareGenProvider(Protocol):
    def generate_plan(self, profile: OnboardingProfile) -> ProviderOutput:
        ...


class DeterministicSoftwareGenProvider:
    """Offline provider for tests and reviewed plan files."""

    def __init__(self, plan: InterfacePlan | dict[str, Any], *, model: str = "reviewed-plan") -> None:
        self.plan = plan if isinstance(plan, InterfacePlan) else InterfacePlan.model_validate(plan)
        self.model = model

    def generate_plan(self, profile: OnboardingProfile) -> ProviderOutput:
        started = time.monotonic()
        prompt_hash = canonical_sha256(_profile_prompt_payload(profile))
        return ProviderOutput(
            plan=self.plan.model_copy(deep=True),
            provider="deterministic",
            model=self.model,
            prompt_sha256=prompt_hash,
            api_calls=0,
            elapsed_s=time.monotonic() - started,
            trace={"mode": "reviewed_plan", "profile_sha256": prompt_hash},
        )


class OpenAISoftwareGenProvider:
    """OpenAI Responses API provider with bounded structured-output retries."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        reasoning_effort: str = "medium",
        max_retries: int = 2,
        client: Any | None = None,
        trace_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self._provided_client = client
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def _client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        load_dotenv()
        import openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI softwaregen provider.")
        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_API_BASE", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    def _write_trace(self, trace: dict[str, Any]) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"softwaregen-{int(time.time() * 1000)}-{canonical_sha256(trace)[:12]}.json"
        path.write_text(json.dumps(sanitize_data(trace), indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    def generate_plan(self, profile: OnboardingProfile) -> ProviderOutput:
        instructions, prompt = build_generation_prompt(profile)
        prompt_hash = canonical_sha256({"instructions": instructions, "prompt": prompt})
        client = self._client()
        started = time.monotonic()
        api_calls = 0
        errors: list[str] = []
        last_response_text = ""
        request = {
            "model": self.model,
            "instructions": instructions,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "reasoning": {"effort": self.reasoning_effort, "summary": "concise"},
        }

        for attempt in range(self.max_retries + 1):
            try:
                api_calls += 1
                active_request = request
                if attempt == 0:
                    response = client.responses.parse(**request, text_format=InterfacePlan)
                    parsed = getattr(response, "output_parsed", None)
                    if parsed is None:
                        last_response_text = _extract_response_text(response)
                        parsed = json.loads(last_response_text)
                else:
                    retry_prompt = json.dumps(
                        {
                            "repair_instruction": (
                                "The previous response failed local schema validation. Return only one JSON object "
                                "using the exact output_schema and exact_field_rules already provided."
                            ),
                            "original_request": json.loads(prompt),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    active_request = {
                        **request,
                        "input": [{"role": "user", "content": [{"type": "input_text", "text": retry_prompt}]}],
                        "text": {"format": {"type": "json_object"}},
                    }
                    response = client.responses.create(**active_request)
                    last_response_text = _extract_response_text(response)
                    parsed = json.loads(last_response_text)
                if isinstance(parsed, InterfacePlan):
                    plan = parsed
                else:
                    plan = InterfacePlan.model_validate(normalize_interface_plan_payload(parsed, profile))
                elapsed = time.monotonic() - started
                trace = sanitize_data(
                    {
                        "provider": "openai",
                        "model": self.model,
                        "attempt": attempt,
                        "api_calls": api_calls,
                        "prompt_sha256": prompt_hash,
                        "request": active_request,
                        "plan": plan.model_dump(mode="json"),
                        "errors": errors,
                    }
                )
                self._write_trace(trace)
                return ProviderOutput(
                    plan=plan,
                    provider="openai",
                    model=self.model,
                    prompt_sha256=prompt_hash,
                    api_calls=api_calls,
                    elapsed_s=elapsed,
                    trace=trace,
                )
            except Exception as exc:  # pragma: no cover - retry behavior is API dependent.
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt >= self.max_retries:
                    trace = sanitize_data(
                        {
                            "provider": "openai",
                            "model": self.model,
                            "api_calls": api_calls,
                            "prompt_sha256": prompt_hash,
                            "errors": errors,
                            "last_response_text": last_response_text,
                        }
                    )
                    self._write_trace(trace)
                    raise
                time.sleep(1.0 + attempt)
        raise RuntimeError("OpenAI softwaregen provider exhausted retries.")
