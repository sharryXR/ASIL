import json
from pathlib import Path

from asil.eval.evaluator import evaluate_observation
from asil.protocol import Action


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "thunderbird"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def test_from_evaluation_context_creates_default_mailbox(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)

    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert adapter.source_path == tmp_path / "thunderbird_mailbox.json"
    assert adapter.get_context()["mailbox_path"] == str(tmp_path / "thunderbird_mailbox.json")
    assert obs.meta.app_name == "Thunderbird"
    assert obs.meta.observation_source == "json_mailbox"
    assert "folder:Inbox" in elements
    assert "message:msg_client_followup" in elements
    assert "message_view:msg_client_followup" in elements
    assert elements["folder:Inbox"].value["selected"] is True
    assert elements["message:msg_client_followup"].value["selected"] is True
    assert obs.app_state.current_view == "mail_3pane"
    assert obs.app_state.active_document == "Inbox"


def test_setup_state_compose_focus_exposes_draft_pane(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)

    adapter.setup_state("compose_focus")
    obs = adapter.observe()

    compose = next(element for element in obs.interactive_elements if element.id == "compose:draft")
    assert compose.type == "compose"
    assert compose.value["to"] == "team@acme.test"
    assert compose.value["subject"] == "Status update"
    assert compose.value["is_open"] is True
    assert obs.app_state.current_view == "compose"


def test_observe_reselects_visible_message_for_mail_3pane_surface(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    state = adapter._read_state()
    state["selected_folder"] = "Projects"
    state["selected_message_id"] = "msg_client_followup"
    adapter._write_state(state)

    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert obs.app_state.current_view == "mail_3pane"
    assert elements["folder:Projects"].value["selected"] is True
    assert elements["message:msg_design_mockups"].value["selected"] is True
    assert "message_view:msg_design_mockups" in elements


def test_execute_reorganizes_mailbox_and_updates_visible_message_state(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)

    action = Action(
        action_type="invoke_function",
        target="thunderbird",
        params={
            "operations": [
                {"action": "select_message", "id": "msg_client_followup"},
                {"action": "set_starred", "id": "msg_client_followup", "starred": True},
                {"action": "add_tag", "id": "msg_client_followup", "tag": "Follow Up"},
                {"action": "move_message", "id": "msg_client_followup", "destination": "Projects"},
                {"action": "switch_folder", "folder": "Projects"},
            ]
        },
    )

    obs = adapter.execute(action)
    elements = {element.id: element for element in obs.interactive_elements}

    moved = elements["message:msg_client_followup"]
    preview = elements["message_view:msg_client_followup"]
    assert elements["folder:Projects"].value["selected"] is True
    assert moved.value["folder"] == "Projects"
    assert moved.value["starred"] is True
    assert "Follow Up" in moved.value["tags"]
    assert moved.value["selected"] is True
    assert preview.value["subject"] == "Client follow-up"
    assert preview.value["folder"] == "Projects"


def test_execute_can_create_and_send_a_reply(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)

    action = Action(
        action_type="invoke_function",
        target="thunderbird",
        params={
            "operations": [
                {"action": "select_message", "id": "msg_budget_review"},
                {"action": "create_draft_reply", "source_id": "msg_budget_review"},
                {"action": "update_draft", "changes": {"body": "Approved for Thursday.\nThanks,\nMorgan"}},
                {"action": "send_draft"},
                {"action": "switch_folder", "folder": "Sent"},
            ]
        },
    )

    obs = adapter.execute(action)
    elements = {element.id: element for element in obs.interactive_elements}

    sent = next(element for element in obs.interactive_elements if element.id.startswith("message:sent_"))
    assert elements["folder:Sent"].value["selected"] is True
    assert sent.value["folder"] == "Sent"
    assert sent.value["subject"] == "Re: Budget review notes"
    assert "Approved for Thursday." in sent.value["body"]
    assert sent.value["read"] is True
    assert all(element.id != "compose:draft" for element in obs.interactive_elements)


def test_validate_action_checks_thunderbird_contract(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)

    good = Action(action_type="invoke_function", target="thunderbird", params={"operations": []})
    bad_type = Action(action_type="modify_file", target="thunderbird", params={"operations": []})
    bad_target = Action(action_type="invoke_function", target="mail", params={"operations": []})

    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad_type)
    assert not adapter.validate_action(bad_target)


def test_rendering_fallback_is_honest_when_rendering_mail_state(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "real thunderbird window" in artifact.description.lower()


def test_render_to_png_uses_real_thunderbird_window(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)

    calls = {}

    def fake_launch(command, **kwargs):
        calls["command"] = command
        calls["launch_kwargs"] = kwargs
        return object()

    def fake_capture(output_path, **kwargs):
        calls["output_path"] = Path(output_path)
        calls["capture_kwargs"] = kwargs
        return Path(output_path)

    def fake_terminate(proc, **kwargs):
        calls["terminated"] = proc

    def fake_send_keys(title_pattern, keys, **kwargs):
        calls.setdefault("navigation", []).append(
            {"title_pattern": title_pattern, "keys": list(keys), "kwargs": kwargs}
        )
        return "window-id"

    def fake_click_window_relative(title_pattern, x_offset, y_offset, **kwargs):
        calls.setdefault("clicks", []).append(
            {
                "title_pattern": title_pattern,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "kwargs": kwargs,
            }
        )
        return "window-id"

    monkeypatch.setattr("asil.adapters.thunderbird.shutil.which", lambda name: "/usr/bin/thunderbird")
    monkeypatch.setattr("asil.adapters.thunderbird.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.thunderbird.capture_window_to_png", fake_capture)
    monkeypatch.setattr("asil.adapters.thunderbird.send_keys_to_window", fake_send_keys)
    monkeypatch.setattr("asil.adapters.thunderbird.click_window_relative", fake_click_window_relative)
    monkeypatch.setattr("asil.adapters.thunderbird.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "thunderbird.png")

    assert calls["command"][0] == "/usr/bin/thunderbird"
    assert calls["command"][1] == "-profile"
    assert calls["command"][2].endswith("_thunderbird_profile")
    assert len(calls["command"]) == 4
    assert calls["command"][3] == "--new-instance"
    assert calls["launch_kwargs"] == {}
    assert calls["output_path"] == tmp_path / "thunderbird.png"
    assert calls["capture_kwargs"] == {
        "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
        "timeout": 60.0,
        "margin": 12,
        "settle_delay": 6.0,
        "min_width": 900,
        "min_height": 600,
    }
    assert calls["navigation"] == [
        {
            "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
            "keys": ["Right"],
            "kwargs": {
                "timeout": 30.0,
                "min_width": 900,
                "min_height": 600,
            },
        }
    ]
    assert calls["clicks"] == [
        {
            "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
            "x_offset": 64,
            "y_offset": 92,
            "kwargs": {
                "timeout": 30.0,
                "min_width": 900,
                "min_height": 600,
            },
        },
        {
            "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
            "x_offset": 86,
            "y_offset": 121,
            "kwargs": {
                "timeout": 30.0,
                "min_width": 900,
                "min_height": 600,
            },
        }
    ]
    assert out == tmp_path / "thunderbird.png"


def test_render_to_png_uses_compose_window_when_draft_is_open(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("compose_focus")

    calls = {}

    def fake_launch(command, **kwargs):
        calls["command"] = command
        calls["launch_kwargs"] = kwargs
        return object()

    def fake_capture(output_path, **kwargs):
        calls["output_path"] = Path(output_path)
        calls["capture_kwargs"] = kwargs
        return Path(output_path)

    def fake_terminate(proc, **kwargs):
        calls["terminated"] = proc

    def fake_send_keys(title_pattern, keys, **kwargs):
        calls.setdefault("navigation", []).append(
            {"title_pattern": title_pattern, "keys": list(keys), "kwargs": kwargs}
        )
        return "window-id"

    def fake_click_window_relative(title_pattern, x_offset, y_offset, **kwargs):
        calls.setdefault("clicks", []).append(
            {
                "title_pattern": title_pattern,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "kwargs": kwargs,
            }
        )
        return "window-id"

    def fake_type_text(title_pattern, text, **kwargs):
        calls.setdefault("typed", []).append(
            {"title_pattern": title_pattern, "text": text, "kwargs": kwargs}
        )
        return "window-id"

    def fake_wait_for_window(title_pattern, **kwargs):
        calls["waited_for"] = {"title_pattern": title_pattern, "kwargs": kwargs}
        return "window-id"

    monkeypatch.setattr("asil.adapters.thunderbird.shutil.which", lambda name: "/usr/bin/thunderbird")
    monkeypatch.setattr("asil.adapters.thunderbird.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.thunderbird.capture_window_to_png", fake_capture)
    monkeypatch.setattr("asil.adapters.thunderbird.send_keys_to_window", fake_send_keys)
    monkeypatch.setattr("asil.adapters.thunderbird.click_window_relative", fake_click_window_relative)
    monkeypatch.setattr("asil.adapters.thunderbird.type_text_to_window", fake_type_text)
    monkeypatch.setattr("asil.adapters.thunderbird.wait_for_window", fake_wait_for_window)
    monkeypatch.setattr("asil.adapters.thunderbird.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("asil.adapters.thunderbird.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "thunderbird_compose.png")

    assert calls["command"][0] == "/usr/bin/thunderbird"
    assert calls["command"][1] == "-profile"
    assert len(calls["command"]) == 4
    assert calls["command"][3] == "--new-instance"
    assert calls["launch_kwargs"] == {}
    assert calls["waited_for"] == {
        "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
        "kwargs": {
            "timeout": 45.0,
            "min_width": 700,
            "min_height": 500,
        },
    }
    assert calls["navigation"] == [
        {
            "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
            "keys": ["ctrl+n"],
            "kwargs": {
                "timeout": 30.0,
                "min_width": 600,
                "min_height": 500,
            },
        },
        {
            "title_pattern": r"Write:.*|.*\(no subject\).*Thunderbird",
            "keys": ["Return"],
            "kwargs": {
                "timeout": 30.0,
                "min_width": 500,
                "min_height": 400,
                "window_class_pattern": "Thunderbird|thunderbird",
            },
        },
    ]
    assert calls["clicks"] == [
        {
            "title_pattern": r"Write:.*|.*\(no subject\).*Thunderbird",
            "x_offset": 250,
            "y_offset": 120,
            "kwargs": {
                "timeout": 30.0,
                "min_width": 500,
                "min_height": 400,
                "window_class_pattern": "Thunderbird|thunderbird",
            },
        },
        {
            "title_pattern": r"Write:.*|.*\(no subject\).*Thunderbird",
            "x_offset": 250,
            "y_offset": 160,
            "kwargs": {
                "timeout": 30.0,
                "min_width": 500,
                "min_height": 400,
                "window_class_pattern": "Thunderbird|thunderbird",
            },
        },
        {
            "title_pattern": r"Write:.*|.*\(no subject\).*Thunderbird",
            "x_offset": 240,
            "y_offset": 245,
            "kwargs": {
                "timeout": 30.0,
                "min_width": 500,
                "min_height": 400,
                "window_class_pattern": "Thunderbird|thunderbird",
            },
        },
    ]
    assert calls["typed"][0]["text"] == "team@acme.test"
    assert calls["typed"][1]["text"] == "Status update"
    assert "weekly status update" in calls["typed"][2]["text"]
    assert all(item["kwargs"]["min_width"] == 500 for item in calls["typed"])
    assert all(item["kwargs"]["min_height"] == 400 for item in calls["typed"])
    assert calls["capture_kwargs"] == {
        "title_pattern": r"Write:.*|.*\(no subject\).*Thunderbird",
        "timeout": 60.0,
        "margin": 12,
        "settle_delay": 6.0,
        "min_width": 500,
        "min_height": 400,
    }
    assert out == tmp_path / "thunderbird_compose.png"


def test_render_to_png_falls_back_to_compose_when_no_message_is_selected(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    state = adapter._read_state()
    state["selected_message_id"] = None
    adapter._write_state(state)

    launches = []
    captures = []

    def fake_launch(command, **kwargs):
        proc = object()
        launches.append((command, kwargs, proc))
        return proc

    def fake_capture(output_path, **kwargs):
        captures.append((Path(output_path), kwargs))
        return Path(output_path)

    def fake_terminate(proc, **kwargs):
        return None

    def fake_send_keys(title_pattern, keys, **kwargs):
        return "window-id"

    def fake_click_window_relative(title_pattern, x_offset, y_offset, **kwargs):
        return "window-id"

    def fake_type_text(title_pattern, text, **kwargs):
        return "window-id"

    monkeypatch.setattr("asil.adapters.thunderbird.shutil.which", lambda name: "/usr/bin/thunderbird")
    monkeypatch.setattr("asil.adapters.thunderbird.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.thunderbird.capture_window_to_png", fake_capture)
    monkeypatch.setattr("asil.adapters.thunderbird.send_keys_to_window", fake_send_keys)
    monkeypatch.setattr("asil.adapters.thunderbird.click_window_relative", fake_click_window_relative)
    monkeypatch.setattr("asil.adapters.thunderbird.type_text_to_window", fake_type_text)
    monkeypatch.setattr("asil.adapters.thunderbird.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "thunderbird_fallback.png")

    assert len(launches) == 1
    assert launches[0][0] == [
        "/usr/bin/thunderbird",
        "-profile",
        str(tmp_path / "_thunderbird_profile"),
        "--new-instance",
    ]
    assert captures == [
        (
            tmp_path / "thunderbird_fallback.png",
            {
                "title_pattern": ".*Mozilla Thunderbird|.* - Thunderbird",
                "timeout": 60.0,
                "margin": 12,
                "settle_delay": 6.0,
                "min_width": 900,
                "min_height": 600,
            },
        )
    ]
    assert out == tmp_path / "thunderbird_fallback.png"


def test_sync_from_gui_updates_open_draft_from_compose_window(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("compose_focus")

    monkeypatch.setattr(
        adapter,
        "_read_saved_draft_state",
        lambda: {
            "to": "team@acme.test",
            "subject": "Updated subject",
            "body": "Updated body",
            "is_open": True,
        },
    )

    adapter.sync_from_gui()
    state = adapter._read_state()

    assert state["draft"] == {
        "to": "team@acme.test",
        "subject": "Updated subject",
        "body": "Updated body",
        "source_id": None,
        "is_open": True,
    }


def test_sync_from_gui_preserves_existing_draft_text_when_readback_is_blank(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("compose_focus")

    monkeypatch.setattr(
        adapter,
        "_read_saved_draft_state",
        lambda: {"to": "", "subject": "", "body": "", "is_open": True},
    )

    before = adapter._read_state()["draft"].copy()
    adapter.sync_from_gui()
    after = adapter._read_state()["draft"]

    assert after["to"] == before["to"]
    assert after["subject"] == before["subject"]
    assert after["body"] == before["body"]
    assert after["is_open"] is True


def test_normalize_compose_body_strips_html_spacing_artifacts(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    raw = "Approved for Thursday.\r\n\r\n    \r\nThanks,\r\n\r\n    \r\nMorgan"

    assert adapter._normalize_compose_body(raw) == "Approved for Thursday.\nThanks,\nMorgan"


def test_read_saved_draft_state_normalizes_saved_body(tmp_path: Path, monkeypatch):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    monkeypatch.setattr("asil.adapters.thunderbird.send_keys_to_window", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.adapters.thunderbird.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapter,
        "_latest_saved_draft_message",
        lambda: {
            "to": "finance@acme.test",
            "subject": "Budget review notes",
            "body": "Approved for Thursday.\r\n\r\n    \r\nThanks,\r\n\r\n    \r\nMorgan",
        },
    )

    state = adapter._read_saved_draft_state()

    assert state["body"] == "Approved for Thursday.\nThanks,\nMorgan"


def test_latest_saved_draft_message_reads_real_drafts_mbox(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path)
    profile_dir = adapter._ensure_profile()
    drafts_path = profile_dir / "Mail" / "Local Folders" / "Drafts"
    drafts_path.write_text(
        "\n".join(
            [
                "From - Tue Apr 21 14:18:49 2026",
                "To: team@acme.test",
                "Subject: Status update",
                "Content-Type: text/html; charset=UTF-8",
                "",
                "<html><body><p>Draft the weekly status update for the team.</p></body></html>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert adapter._latest_saved_draft_message() == {
        "to": "team@acme.test",
        "subject": "Status update",
        "body": "Draft the weekly status update for the team.",
    }


def test_thunderbird_action_schema_describes_supported_operations():
    root = Path(__file__).resolve().parent.parent / "src" / "asil" / "action_schemas"
    schema = json.loads((root / "thunderbird.json").read_text(encoding="utf-8"))

    action_names = {item["name"] for item in schema["actions"]}
    assert schema["software"] == "Thunderbird"
    assert schema["target"] == "thunderbird"
    assert schema["supported_action_types"] == ["invoke_function"]
    assert {
        "switch_folder",
        "select_message",
        "set_starred",
        "set_read",
        "move_message",
        "archive_message",
        "delete_message",
        "restore_message",
        "add_tag",
        "remove_tag",
        "create_draft_reply",
        "update_draft",
        "send_draft",
    }.issubset(action_names)


def test_thunderbird_example_set_contains_20_tasks():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "thunderbird"
    tasks = sorted(path for path in root.glob("thunderbird_*.json") if path.stem.removeprefix("thunderbird_").isdigit())

    assert len(tasks) == 20
    ids = [json.loads(task.read_text(encoding="utf-8"))["id"] for task in tasks]
    assert ids == [f"thunderbird_{idx:02d}" for idx in range(1, 21)]


def test_thunderbird_examples_align_to_compose_surface():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "thunderbird"

    for path in sorted(path for path in root.glob("thunderbird_*.json") if path.stem.removeprefix("thunderbird_").isdigit()):
        task = json.loads(path.read_text(encoding="utf-8"))
        gui_expectations = task["gui_expectations"]
        checkpoints = task["evaluator"]["paths"][0]["checkpoints"]
        operations = task["_asil"]["actions"][0]["params"]["operations"]
        operation_names = [operation["action"] for operation in operations]

        assert gui_expectations["success_surface"] == "compose_window"
        assert gui_expectations["visible_change_summary"]
        assert all(checkpoint["gui_visible_required"] for checkpoint in checkpoints)
        assert task["_asil"]["initial_state"] == "compose_focus"
        assert set(gui_expectations["checkpoint_visibility"].values()) == {"visible_in_compose_window"}
        assert [checkpoint["id"] for checkpoint in checkpoints] == ["draft_to", "draft_subject", "draft_body", "draft_open"]
        assert operation_names == ["update_draft"]


def test_thunderbird_examples_stay_on_compose_drafts():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "thunderbird"

    compose_tasks = 0
    for path in sorted(path for path in root.glob("thunderbird_*.json") if path.stem.removeprefix("thunderbird_").isdigit()):
        task = json.loads(path.read_text(encoding="utf-8"))
        checkpoints = task["evaluator"]["paths"][0]["checkpoints"]
        rules = [checkpoint["rule"] for checkpoint in checkpoints]
        compose_tasks += 1
        assert any(
            rule.get("element_value", {}).get("id") == "compose:draft"
            for rule in rules
        )
        assert all("element_contains" not in rule for rule in rules)
        assert all(
            not rule.get("element_value", {}).get("id", "").startswith("folder:")
            and not rule.get("element_contains", {}).get("id", "").startswith("folder:")
            and not rule.get("no_element_matches", {}).get("id", "").startswith("folder:")
            for rule in rules
        )

    assert compose_tasks == 20


def test_representative_thunderbird_tasks_evaluate_successfully(tmp_path: Path):
    from asil.adapters.thunderbird import ThunderbirdAdapter

    for task_id in ("thunderbird_01", "thunderbird_10", "thunderbird_18"):
        task = _task(task_id)
        adapter = ThunderbirdAdapter.from_evaluation_context(tmp_path / task_id)
        adapter.setup_state(task["_asil"].get("initial_state", "default"))

        observation = adapter.observe()
        for action_data in task["_asil"]["actions"]:
            action = Action(**action_data)
            observation = adapter.execute(action)

        report = evaluate_observation(
            observation,
            validation=task["_asil"].get("validation"),
            evaluator=task.get("evaluator"),
        )
        assert report.success, task_id
