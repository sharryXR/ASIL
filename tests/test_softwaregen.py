from __future__ import annotations

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading

import pytest
from pydantic import ValidationError

from asil.protocol import Action
from asil.softwaregen import (
    AuditReport,
    DeclarativeAdapter,
    DeterministicSoftwareGenProvider,
    ExtensionBundle,
    OpenAISoftwareGenProvider,
    OnboardingProfile,
    audit_bundle,
    build_docker_probe_command,
    canonical_sha256,
    derive_action_schema,
    generate_adapter_wrapper,
    generate_extension,
    observation_state_sha256,
    softwaregen_main,
    render_template,
    render_path_template,
    resolve_json_pointer,
    sanitize_data,
    write_extension_bundle,
)


def _profile_payload() -> dict:
    return {
        "software_id": "demo_service",
        "display_name": "Demo Service",
        "version": "1.0",
        "integration_pattern": "service_api",
        "description": "A local JSON service used to test grounded onboarding.",
        "evidence": [
            {
                "id": "ev_items_get",
                "kind": "api_spec",
                "locator": "GET /api/items",
                "excerpt": "Returns {items: [{id, title, done}]}.",
                "sample": {"items": [{"id": 1, "title": "First", "done": False}]},
            },
            {
                "id": "ev_items_create",
                "kind": "api_spec",
                "locator": "POST /api/items",
                "excerpt": "Creates an item from a title and returns it.",
                "sample": {"id": 2, "title": "Second", "done": False},
            },
            {
                "id": "ev_cli",
                "kind": "command_help",
                "locator": "democtl --help",
                "excerpt": "democtl list --json and democtl add --title TITLE --json",
            },
        ],
        "runtime": {
            "base_url": "http://127.0.0.1:8765",
            "allowed_hosts": ["127.0.0.1", "localhost"],
            "allowed_executables": ["democtl"],
            "filesystem_root": ".",
            "headers": {"Authorization": "Bearer ${ENV:DEMO_TOKEN}"},
            "request_timeout_s": 5.0,
        },
        "requirements": ["Expose item state and an evidenced create action."],
        "known_limitations": ["The API has no documented delete operation."],
    }


def _bundle_payload() -> dict:
    return {
        "schema_version": "1.0",
        "profile": _profile_payload(),
        "plan": {
            "summary": "Read and create items through the documented local API.",
            "observation_views": [
                {
                    "id": "items",
                    "description": "All visible items.",
                    "probe": {
                        "transport": "http_json",
                        "method": "GET",
                        "path": "/api/items",
                        "query": {},
                        "items_pointer": "/items",
                        "evidence_refs": ["ev_items_get"],
                    },
                    "element": {
                        "id_prefix": "item:",
                        "id_pointer": "/id",
                        "type": "item",
                        "label_pointer": "/title",
                        "value_fields": {"title": "/title", "done": "/done"},
                        "metadata_fields": {},
                        "editable": True,
                        "actions": ["create_item"],
                        "evidence_refs": ["ev_items_get"],
                    },
                    "evidence_refs": ["ev_items_get"],
                }
            ],
            "operations": [
                {
                    "name": "create_item",
                    "description": "Create a new item.",
                    "action_type": "api_call",
                    "target": "demo_service",
                    "parameters": [
                        {"name": "title", "value_type": "string", "required": True},
                    ],
                    "request": {
                        "transport": "http_json",
                        "method": "POST",
                        "path": "/api/items",
                        "query": {},
                        "body": {"title": "${title}"},
                    },
                    "evidence_refs": ["ev_items_create"],
                }
            ],
            "limitations": ["Delete is not generated because it is not evidenced."],
        },
        "provenance": {
            "provider": "deterministic",
            "model": "fixture",
            "profile_sha256": "0" * 64,
            "plan_sha256": "1" * 64,
        },
    }


def _bundle() -> ExtensionBundle:
    return ExtensionBundle.model_validate(_bundle_payload())


def _codes(report: AuditReport) -> set[str]:
    return {finding.code for finding in report.findings}


def test_softwaregen_models_accept_grounded_local_http_bundle():
    profile = OnboardingProfile.model_validate(_profile_payload())
    bundle = _bundle()

    assert profile.software_id == "demo_service"
    assert bundle.plan.operations[0].request.transport == "http_json"
    assert audit_bundle(bundle).ok is True


def test_softwaregen_audit_rejects_unknown_evidence_reference():
    payload = _bundle_payload()
    payload["plan"]["operations"][0]["evidence_refs"] = ["ev_invented"]

    report = audit_bundle(ExtensionBundle.model_validate(payload))

    assert report.ok is False
    assert "unknown_evidence" in _codes(report)


def test_softwaregen_audit_rejects_http_path_traversal():
    payload = _bundle_payload()
    payload["plan"]["observation_views"][0]["probe"]["path"] = "/api/../admin/secrets"

    report = audit_bundle(ExtensionBundle.model_validate(payload))

    assert report.ok is False
    assert "unsafe_http_path" in _codes(report)


def test_softwaregen_audit_rejects_non_allowlisted_command():
    payload = _bundle_payload()
    payload["plan"]["observation_views"][0]["probe"] = {
        "transport": "command_json",
        "argv": ["curl", "http://remote.invalid"],
        "items_pointer": "/items",
        "evidence_refs": ["ev_cli"],
    }

    report = audit_bundle(ExtensionBundle.model_validate(payload))

    assert report.ok is False
    assert "executable_not_allowed" in _codes(report)


def test_softwaregen_audit_rejects_unknown_action_placeholder():
    payload = _bundle_payload()
    payload["plan"]["operations"][0]["request"]["body"] = {"title": "${invented}"}

    report = audit_bundle(ExtensionBundle.model_validate(payload))

    assert report.ok is False
    assert "unknown_placeholder" in _codes(report)


def test_softwaregen_audit_rejects_literal_sensitive_header_values():
    payload = _bundle_payload()
    payload["profile"]["runtime"]["headers"] = {"Authorization": "Bearer literal-secret"}

    report = audit_bundle(ExtensionBundle.model_validate(payload))

    assert report.ok is False
    assert "literal_sensitive_header" in _codes(report)


def test_softwaregen_audit_rejects_duplicate_view_and_operation_ids():
    payload = _bundle_payload()
    payload["plan"]["observation_views"].append(deepcopy(payload["plan"]["observation_views"][0]))
    payload["plan"]["operations"].append(deepcopy(payload["plan"]["operations"][0]))

    report = audit_bundle(ExtensionBundle.model_validate(payload))

    assert report.ok is False
    assert {"duplicate_view", "duplicate_operation"} <= _codes(report)


def test_softwaregen_observation_mapping_requires_stable_id_pointer():
    payload = _bundle_payload()
    payload["plan"]["observation_views"][0]["element"]["id_pointer"] = ""

    with pytest.raises(ValidationError):
        ExtensionBundle.model_validate(payload)


def test_softwaregen_models_reject_unreviewed_extra_fields():
    payload = _bundle_payload()
    payload["plan"]["operations"][0]["unreviewed_execution_code"] = "do_anything()"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExtensionBundle.model_validate(payload)


def test_softwaregen_runtime_resolves_rfc6901_json_pointers():
    document = {"items": [{"a/b": {"~value": 7}}]}

    assert resolve_json_pointer(document, "") is document
    assert resolve_json_pointer(document, "/items/0/a~1b/~0value") == 7

    with pytest.raises(KeyError):
        resolve_json_pointer(document, "/items/3")


def test_softwaregen_runtime_renders_recursive_templates_without_losing_types():
    template = {
        "title": "${title}",
        "count": "${count}",
        "label": "item-${count}",
        "nested": ["${enabled}", {"owner": "${owner}"}],
    }
    params = {"title": "Roadmap", "count": 3, "enabled": True, "owner": "asil"}

    rendered = render_template(template, params)

    assert rendered == {
        "title": "Roadmap",
        "count": 3,
        "label": "item-3",
        "nested": [True, {"owner": "asil"}],
    }
    with pytest.raises(ValueError, match="missing"):
        render_template("${missing}", params)


def test_softwaregen_runtime_renders_http_path_parameters_without_path_injection():
    assert render_path_template(
        "/api/repos/${owner}/${repo}",
        {"owner": "team name", "repo": "project-1"},
    ) == "/api/repos/team%20name/project-1"

    with pytest.raises(ValueError, match="path separators"):
        render_path_template("/api/repos/${owner}/${repo}", {"owner": "team/admin", "repo": "x"})
    with pytest.raises(ValueError, match="traversal"):
        render_path_template("/api/repos/${owner}/${repo}", {"owner": "..", "repo": "x"})


class _DemoJSONHandler(BaseHTTPRequestHandler):
    items: list[dict] = []

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path == "/api/items":
            self._write_json(200, {"items": list(type(self).items)})
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.path != "/api/items":
            self._write_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        item = {"id": len(type(self).items) + 1, "title": payload["title"], "done": False}
        type(self).items.append(item)
        self._write_json(201, item)

    def log_message(self, format: str, *args) -> None:
        del format, args


@pytest.fixture
def demo_json_service():
    _DemoJSONHandler.items = [{"id": 1, "title": "First", "done": False}]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DemoJSONHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _runtime_bundle(*, base_url: str, headers: dict[str, str] | None = None) -> ExtensionBundle:
    payload = _bundle_payload()
    payload["profile"]["runtime"]["base_url"] = base_url
    payload["profile"]["runtime"]["headers"] = headers or {}
    return ExtensionBundle.model_validate(payload)


def test_softwaregen_runtime_observes_http_json_and_executes_semantic_action(demo_json_service: str):
    adapter = DeclarativeAdapter(_runtime_bundle(base_url=demo_json_service))

    before = adapter.observe()
    after = adapter.execute(
        Action(
            action_type="api_call",
            target="demo_service",
            params={"operation": "create_item", "arguments": {"title": "Second"}},
        )
    )

    assert before.meta.observation_source == "rest_api"
    assert [(element.id, element.label) for element in before.interactive_elements] == [("item:1", "First")]
    assert [(element.id, element.value["title"]) for element in after.interactive_elements] == [
        ("item:1", "First"),
        ("item:2", "Second"),
    ]


def test_softwaregen_observation_state_hash_excludes_volatile_timestamp(demo_json_service: str):
    adapter = DeclarativeAdapter(_runtime_bundle(base_url=demo_json_service))

    first = adapter.observe()
    second = adapter.observe()

    assert first.meta.timestamp != second.meta.timestamp
    assert observation_state_sha256(first) == observation_state_sha256(second)


def test_softwaregen_runtime_resolves_base_url_from_env_and_rechecks_host_allowlist(
    demo_json_service: str,
    monkeypatch,
):
    payload = _bundle_payload()
    payload["profile"]["runtime"].update(
        {"base_url": "", "base_url_env": "DEMO_SERVICE_URL", "headers": {}}
    )
    bundle = ExtensionBundle.model_validate(payload)

    monkeypatch.setenv("DEMO_SERVICE_URL", demo_json_service)
    assert DeclarativeAdapter(bundle).observe().interactive_elements[0].id == "item:1"

    monkeypatch.setenv("DEMO_SERVICE_URL", "http://example.invalid:8765")
    with pytest.raises(ValueError, match="approved"):
        DeclarativeAdapter(bundle).observe()


def test_softwaregen_runtime_rejects_invalid_action_arguments_before_io(demo_json_service: str):
    adapter = DeclarativeAdapter(_runtime_bundle(base_url=demo_json_service))
    missing = Action(
        action_type="api_call",
        target="demo_service",
        params={"operation": "create_item", "arguments": {}},
    )
    extra = Action(
        action_type="api_call",
        target="demo_service",
        params={"operation": "create_item", "arguments": {"title": "Second", "invented": True}},
    )
    wrong_type = Action(
        action_type="api_call",
        target="demo_service",
        params={"operation": "create_item", "arguments": {"title": 9}},
    )

    assert adapter.validate_action(missing) is False
    assert adapter.validate_action(extra) is False
    assert adapter.validate_action(wrong_type) is False
    with pytest.raises(ValueError, match="required"):
        adapter.execute(missing)
    assert _DemoJSONHandler.items == [{"id": 1, "title": "First", "done": False}]


def test_softwaregen_runtime_observes_direct_command_json(tmp_path: Path):
    script = tmp_path / "emit_items.py"
    script.write_text(
        "import json\nprint(json.dumps({'items': [{'id': 4, 'title': 'CLI item', 'done': True}]}))\n",
        encoding="utf-8",
    )
    payload = _bundle_payload()
    payload["profile"]["runtime"].update(
        {
            "allowed_executables": [os.path.basename(sys.executable)],
            "filesystem_root": str(tmp_path),
            "headers": {},
        }
    )
    payload["plan"]["observation_views"][0]["probe"] = {
        "transport": "command_json",
        "argv": [sys.executable, str(script)],
        "items_pointer": "/items",
        "evidence_refs": ["ev_cli"],
    }

    observation = DeclarativeAdapter(ExtensionBundle.model_validate(payload)).observe()

    assert observation.meta.observation_source == "script_api"
    assert observation.interactive_elements[0].id == "item:4"
    assert observation.interactive_elements[0].value["done"] is True


def test_softwaregen_runtime_observes_json_file_below_approved_root(tmp_path: Path):
    (tmp_path / "state.json").write_text(
        json.dumps({"items": [{"id": 8, "title": "File item", "done": False}]}),
        encoding="utf-8",
    )
    payload = _bundle_payload()
    payload["profile"]["runtime"].update({"filesystem_root": str(tmp_path), "headers": {}})
    payload["plan"]["observation_views"][0]["probe"] = {
        "transport": "json_file",
        "path": "state.json",
        "items_pointer": "/items",
        "evidence_refs": ["ev_items_get"],
    }

    observation = DeclarativeAdapter(ExtensionBundle.model_validate(payload)).observe()

    assert observation.meta.observation_source == "file_parse"
    assert observation.interactive_elements[0].label == "File item"


def test_softwaregen_generation_keeps_runtime_outside_model_control():
    profile = OnboardingProfile.model_validate(_profile_payload())
    plan = _bundle().plan
    provider = DeterministicSoftwareGenProvider(plan, model="fixture-plan")

    result = generate_extension(profile, provider)

    assert result.bundle.profile.runtime == profile.runtime
    assert result.bundle.plan == plan
    assert result.bundle.provenance.provider == "deterministic"
    assert result.bundle.provenance.model == "fixture-plan"
    assert result.audit.ok is True


def test_softwaregen_derives_action_schema_from_validated_operations():
    schema = derive_action_schema(_bundle())

    assert schema["software"] == "Demo Service"
    assert schema["supported_action_types"] == ["api_call"]
    assert schema["target"] == "demo_service"
    assert schema["actions"][0]["name"] == "create_item"
    assert schema["actions"][0]["params_schema"]["arguments"]["title"] == {
        "type": "string",
        "required": True,
    }
    assert schema["actions"][0]["example"]["params"]["operation"] == "create_item"


def test_softwaregen_adapter_wrapper_and_hashes_are_deterministic():
    bundle = _bundle()

    first = generate_adapter_wrapper(bundle)
    second = generate_adapter_wrapper(bundle)

    assert first == second
    assert "class DemoServiceAdapter(DeclarativeAdapter):" in first
    assert "extension.json" in first
    assert canonical_sha256(bundle.model_dump(mode="json")) == canonical_sha256(
        bundle.model_dump(mode="json")
    )


def test_softwaregen_sanitizes_secrets_recursively():
    payload = {
        "Authorization": "Bearer top-secret",
        "nested": {"api_key": "sk-secret", "safe": "visible"},
        "headers": {"X-Trace": "safe", "Cookie": "session=secret"},
    }

    sanitized = sanitize_data(payload)
    serialized = json.dumps(sanitized)

    assert "top-secret" not in serialized
    assert "sk-secret" not in serialized
    assert "session=secret" not in serialized
    assert sanitized["nested"]["safe"] == "visible"


class _FakeParsedResponse:
    def __init__(self, plan) -> None:
        self.output_parsed = plan


class _FakeResponsesAPI:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeParsedResponse(self.plan)


class _FakeOpenAIClient:
    def __init__(self, plan) -> None:
        self.responses = _FakeResponsesAPI(plan)


class _FakeTextResponse:
    def __init__(self, payload: dict) -> None:
        self.output_text = json.dumps(payload)


class _FallbackResponsesAPI:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.parse_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        raise RuntimeError("gateway ignored structured output")

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakeTextResponse(self.payload)


class _FallbackOpenAIClient:
    def __init__(self, payload: dict) -> None:
        self.responses = _FallbackResponsesAPI(payload)


def test_softwaregen_openai_provider_uses_structured_output_without_sending_header_values():
    payload = _profile_payload()
    payload["runtime"]["headers"] = {"Authorization": "Bearer never-send-this"}
    profile = OnboardingProfile.model_validate(payload)
    client = _FakeOpenAIClient(_bundle().plan)
    provider = OpenAISoftwareGenProvider(model="test-model", client=client, max_retries=0)

    output = provider.generate_plan(profile)

    assert output.plan == _bundle().plan
    assert output.api_calls == 1
    request_text = json.dumps(client.responses.calls[0], default=str)
    assert "never-send-this" not in request_text
    assert "<environment-backed>" in request_text
    assert client.responses.calls[0]["text_format"] is type(_bundle().plan)


def test_softwaregen_openai_provider_falls_back_to_explicit_json_schema_within_call_budget():
    client = _FallbackOpenAIClient(_bundle_payload()["plan"])
    provider = OpenAISoftwareGenProvider(model="test-model", client=client, max_retries=2)

    output = provider.generate_plan(OnboardingProfile.model_validate(_profile_payload()))

    assert output.plan == _bundle().plan
    assert output.api_calls == 2
    assert len(client.responses.parse_calls) == 1
    assert len(client.responses.create_calls) == 1
    fallback_request = client.responses.create_calls[0]
    assert fallback_request["text"] == {"format": {"type": "json_object"}}
    assert "output_schema" in fallback_request["input"][0]["content"][0]["text"]


def test_softwaregen_writes_complete_artifact_bundle_and_refuses_implicit_overwrite(tmp_path: Path):
    profile = OnboardingProfile.model_validate(_profile_payload())
    result = generate_extension(profile, DeterministicSoftwareGenProvider(_bundle().plan))
    output_dir = tmp_path / "generated"

    report = write_extension_bundle(result, output_dir)

    assert {
        "extension.json",
        "action_schema.json",
        "adapter.py",
        "generation_report.json",
    } == {path.name for path in output_dir.iterdir()}
    assert ExtensionBundle.model_validate_json((output_dir / "extension.json").read_text()) == result.bundle
    assert report["ok"] is True
    assert set(report["artifact_sha256"]) == {"extension.json", "action_schema.json", "adapter.py"}
    for artifact_name, expected_sha256 in report["artifact_sha256"].items():
        assert hashlib.sha256((output_dir / artifact_name).read_bytes()).hexdigest() == expected_sha256
    with pytest.raises(FileExistsError):
        write_extension_bundle(result, output_dir)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_softwaregen_cli_generates_offline_bundle_and_audits_it(tmp_path: Path, capsys):
    profile_path = _write_json(tmp_path / "profile.json", _profile_payload())
    plan_path = _write_json(tmp_path / "plan.json", _bundle_payload()["plan"])
    output_dir = tmp_path / "generated"

    generate_code = softwaregen_main(
        [
            "generate",
            str(profile_path),
            "--provider",
            "deterministic",
            "--plan-file",
            str(plan_path),
            "--output",
            str(output_dir),
        ]
    )
    generated_output = json.loads(capsys.readouterr().out)
    audit_code = softwaregen_main(["audit", str(output_dir / "extension.json")])
    audit_output = json.loads(capsys.readouterr().out)

    assert generate_code == 0
    assert generated_output["ok"] is True
    assert audit_code == 0
    assert audit_output["ok"] is True


def test_softwaregen_cli_writes_generate_error_inside_output_directory(tmp_path: Path, capsys):
    profile_path = _write_json(tmp_path / "profile.json", _profile_payload())
    output_dir = tmp_path / "failed-generation"
    output_dir.mkdir()

    code = softwaregen_main(
        [
            "generate",
            str(profile_path),
            "--provider",
            "deterministic",
            "--output",
            str(output_dir),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 2
    assert output["ok"] is False
    assert (output_dir / "generation_error.json").exists()


def test_softwaregen_cli_audit_returns_nonzero_for_unsafe_bundle(tmp_path: Path, capsys):
    payload = _bundle_payload()
    payload["plan"]["operations"][0]["request"]["path"] = "/api/../admin"
    bundle_path = _write_json(tmp_path / "unsafe.json", payload)

    code = softwaregen_main(["audit", str(bundle_path)])
    output = json.loads(capsys.readouterr().out)

    assert code == 2
    assert output["ok"] is False
    assert any(finding["code"] == "unsafe_http_path" for finding in output["findings"])


def test_softwaregen_cli_probe_observation_only(demo_json_service: str, tmp_path: Path, capsys):
    bundle_path = _write_json(
        tmp_path / "extension.json",
        _runtime_bundle(base_url=demo_json_service).model_dump(mode="json"),
    )

    code = softwaregen_main(["probe", str(bundle_path)])
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["ok"] is True
    assert output["observation"]["element_count"] == 1
    assert output["action"] is None


def test_softwaregen_cli_probe_refuses_action_without_explicit_flag(
    demo_json_service: str,
    tmp_path: Path,
    capsys,
):
    bundle_path = _write_json(
        tmp_path / "extension.json",
        _runtime_bundle(base_url=demo_json_service).model_dump(mode="json"),
    )
    action_path = _write_json(
        tmp_path / "action.json",
        {
            "action_type": "api_call",
            "target": "demo_service",
            "params": {"operation": "create_item", "arguments": {"title": "Blocked"}},
        },
    )

    code = softwaregen_main(["probe", str(bundle_path), "--action", str(action_path)])
    output = json.loads(capsys.readouterr().out)

    assert code == 2
    assert output["ok"] is False
    assert "--allow-actions" in output["error"]
    assert len(_DemoJSONHandler.items) == 1


def test_softwaregen_cli_probe_records_explicit_action_state_delta(
    demo_json_service: str,
    tmp_path: Path,
    capsys,
):
    bundle_path = _write_json(
        tmp_path / "extension.json",
        _runtime_bundle(base_url=demo_json_service).model_dump(mode="json"),
    )
    action_path = _write_json(
        tmp_path / "action.json",
        {
            "action_type": "api_call",
            "target": "demo_service",
            "params": {"operation": "create_item", "arguments": {"title": "Allowed"}},
        },
    )

    code = softwaregen_main(
        ["probe", str(bundle_path), "--action", str(action_path), "--allow-actions"]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["ok"] is True
    assert output["action"]["validated"] is True
    assert output["action"]["state_changed"] is True
    assert output["action"]["before_sha256"] != output["action"]["after_sha256"]
    assert output["observation"]["element_count"] == 2


def test_softwaregen_docker_probe_command_reuses_public_probe_entrypoint(tmp_path: Path):
    bundle_path = _write_json(tmp_path / "extension.json", _bundle_payload())
    action_path = _write_json(tmp_path / "action.json", {"action_type": "done", "target": "", "params": {}})

    command = build_docker_probe_command(
        bundle_path,
        image="asil-eval:test",
        network="asil-test-net",
        env_names=["DEMO_TOKEN"],
        action_path=action_path,
        allow_actions=True,
    )

    assert command[:3] == ["docker", "run", "--rm"]
    assert ["--network", "asil-test-net"] == command[command.index("--network") : command.index("--network") + 2]
    assert "asil-eval:test" in command
    assert command[-6:] == [
        "probe",
        "/softwaregen/extension.json",
        "--action",
        "/softwaregen/action.json",
        "--allow-actions",
        "--json",
    ]
