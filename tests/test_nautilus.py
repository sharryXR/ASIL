import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asil.eval.evaluator import evaluate_observation
from asil.eval.task_audit import audit_task_file
from asil.protocol import Action


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "nautilus"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def test_from_evaluation_context_creates_seeded_workspace(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    obs = adapter.observe()

    assert adapter.source_path == (tmp_path / "nautilus-workspace").resolve()
    assert obs.meta.app_name == "Nautilus"
    assert obs.app_state.current_view == "browser"
    elements = {element.id: element for element in obs.interactive_elements}
    assert "entry:Inbox" in elements
    assert "entry:Projects" in elements
    assert "entry:todo.txt" in elements


def test_execute_updates_workspace_and_visible_entries(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    action = Action(
        action_type="invoke_function",
        target="nautilus",
        params={
            "operations": [
                {"action": "open_directory", "path": "Projects/alpha"},
                {"action": "rename_entry", "path": "draft-plan.md", "new_name": "release-plan.md"},
                {"action": "copy_entry", "path": "release-plan.md", "destination_dir": "../../Inbox"},
            ]
        },
    )

    obs = adapter.execute(action)
    elements = {element.id: element for element in obs.interactive_elements}

    assert "entry:Projects/alpha/release-plan.md" in elements
    assert "workspace:Projects/alpha/release-plan.md" in elements
    assert (adapter.workspace_path / "Projects" / "alpha" / "release-plan.md").exists()
    assert (adapter.workspace_path / "Inbox" / "release-plan.md").exists()


def test_execute_open_directory_survives_symlinked_root(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    real_root = tmp_path / "real-root"
    alias_root = tmp_path / "alias-root"
    real_root.mkdir()
    alias_root.symlink_to(real_root, target_is_directory=True)

    adapter = NautilusAdapter.from_evaluation_context(alias_root, mock=True)
    obs = adapter.execute(
        Action(
            action_type="invoke_function",
            target="nautilus",
            params={"operations": [{"action": "open_directory", "path": "Inbox"}]},
        )
    )

    location = next(element for element in obs.interactive_elements if element.id == "location")
    assert location.value["path"] == "Inbox"


def test_resolve_slash_targets_workspace_root(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.setup_state("projects_alpha")
    obs = adapter.execute(
        Action(
            action_type="invoke_function",
            target="nautilus",
            params={"operations": [{"action": "open_directory", "path": "/"}]},
        )
    )

    location = next(element for element in obs.interactive_elements if element.id == "location")
    assert location.value["path"] == "/"
    assert any(element.id == "entry:Inbox" for element in obs.interactive_elements)


def test_rendering_reports_real_window_capture(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    artifact = adapter.describe_rendering()

    assert artifact.kind == "app_window"
    assert artifact.actual_page is True
    assert artifact.backend == "x11-window-capture"


@patch("asil.adapters.nautilus.terminate_process")
@patch("asil.adapters.nautilus.capture_window_to_png")
@patch("asil.adapters.nautilus.subprocess.run")
@patch("asil.adapters.nautilus.launch_gui_process")
@patch("asil.adapters.nautilus.ensure_user_access")
@patch("asil.adapters.nautilus.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
def test_render_to_png_uses_window_class_capture(
    _mock_which,
    mock_ensure_access,
    mock_launch,
    _mock_run,
    mock_capture,
    mock_terminate,
    tmp_path: Path,
):
    from asil.adapters.nautilus import NautilusAdapter

    mock_launch.return_value = object()
    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    output = tmp_path / "nautilus.png"

    adapter.render_to_png(output)

    assert mock_ensure_access.call_count == 2
    capture_kwargs = mock_capture.call_args.kwargs
    assert capture_kwargs["title_pattern"] == ".*"
    assert capture_kwargs["window_class_pattern"] == "org.gnome.Nautilus|Org.gnome.Nautilus|nautilus"
    mock_terminate.assert_called_once()


@patch("asil.adapters.nautilus.terminate_process")
@patch("asil.adapters.nautilus.capture_window_to_png")
@patch("asil.adapters.nautilus.type_text_to_window")
@patch("asil.adapters.nautilus.click_window_relative")
@patch("asil.adapters.nautilus.subprocess.run")
@patch("asil.adapters.nautilus.launch_gui_process")
@patch("asil.adapters.nautilus.ensure_user_access")
@patch("asil.adapters.nautilus.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
def test_render_to_png_clicks_search_button_for_search_state(
    _mock_which,
    _mock_ensure_access,
    mock_launch,
    _mock_run,
    mock_click,
    mock_type,
    _mock_capture,
    mock_terminate,
    tmp_path: Path,
):
    from asil.adapters.nautilus import NautilusAdapter

    mock_launch.return_value = object()
    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.setup_state("search_notes")

    adapter.render_to_png(tmp_path / "search.png")

    mock_click.assert_called_once()
    mock_type.assert_called_once()
    assert mock_type.call_args.args[1] == "notes"
    mock_terminate.assert_called_once()


@patch("asil.adapters.nautilus.terminate_process")
@patch("asil.adapters.nautilus.capture_window_to_png")
@patch("asil.adapters.nautilus.subprocess.run")
@patch("asil.adapters.nautilus.launch_gui_process")
@patch("asil.adapters.nautilus.ensure_user_access")
@patch("asil.adapters.nautilus.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
def test_render_to_png_sets_hidden_file_preferences(
    _mock_which,
    _mock_ensure_access,
    mock_launch,
    mock_run,
    _mock_capture,
    mock_terminate,
    tmp_path: Path,
):
    from asil.adapters.nautilus import NautilusAdapter

    mock_launch.return_value = object()
    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.setup_state("hidden_root")

    adapter.render_to_png(tmp_path / "hidden.png")

    invoked = [" ".join(call.args[0]) for call in mock_run.call_args_list]
    assert any("org.gnome.nautilus.preferences show-hidden-files true" in cmd for cmd in invoked)
    assert any("org.gtk.Settings.FileChooser show-hidden true" in cmd for cmd in invoked)
    mock_terminate.assert_called_once()


def test_nautilus_schema_describes_file_manager_operations():
    schema_path = Path(__file__).resolve().parent.parent / "src" / "asil" / "action_schemas" / "nautilus.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["software"] == "Nautilus"
    assert schema["supported_action_types"] == ["invoke_function"]
    assert "description" in schema
    assert "rename_entry" in json.dumps(schema["actions"][0]["params_schema"])
    assert "move_entry" in json.dumps(schema["actions"][0]["params_schema"])
    assert schema["actions"][0]["examples"][0]["action"]["target"] == "nautilus"


def test_nautilus_example_set_contains_20_tasks():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "nautilus"
    tasks = sorted(path for path in root.glob("nautilus_*.json") if path.stem.removeprefix("nautilus_").isdigit())

    assert len(tasks) == 20
    ids = [json.loads(task.read_text(encoding="utf-8"))["id"] for task in tasks]
    assert ids == [f"nautilus_{idx:02d}" for idx in range(1, 21)]


def test_nautilus_tasks_pass_static_gui_audit():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "nautilus"

    for task_path in sorted(path for path in root.glob("nautilus_*.json") if path.stem.removeprefix("nautilus_").isdigit()):
        report = audit_task_file(task_path)
        assert report.ok, f"{task_path.name}: {report.errors}"


def test_representative_nautilus_tasks_evaluate_successfully(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    for task_id in ("nautilus_01", "nautilus_11", "nautilus_12", "nautilus_20"):
        task = _task(task_id)
        adapter = NautilusAdapter.from_evaluation_context(tmp_path / task_id, mock=True)
        adapter.setup_state(task["_asil"].get("initial_state", "default"))

        observation = adapter.observe()
        for action_data in task["_asil"]["actions"]:
            observation = adapter.execute(Action(**action_data))

        report = evaluate_observation(
            observation,
            validation=task["_asil"].get("validation"),
            evaluator=task.get("evaluator"),
        )
        assert report.success, task_id


def test_sync_from_gui_updates_location_and_bookmarks(tmp_path: Path, monkeypatch):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.setup_state("default")

    captured_sessions = []
    monkeypatch.setattr(
        adapter,
        "_read_gui_current_dir",
        lambda session=None: captured_sessions.append(session) or adapter.workspace_path / "Archive",
    )
    monkeypatch.setattr(adapter, "_read_gui_bookmarks", lambda: ["Archive", "Projects/alpha"])

    session = SimpleNamespace(last_capture_window_id="0x410001")
    adapter.sync_from_gui(session)
    state = adapter._load_state()

    assert captured_sessions == [session]
    assert state["current_dir"] == "Archive"
    assert state["bookmarks"] == ["Archive", "Projects/alpha"]


def test_read_gui_current_dir_uses_captured_window_and_waits_for_location_bar(tmp_path: Path, monkeypatch):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    calls: list[tuple[str, object]] = []

    def fake_send(keys, *, preferred_window_id=None):
        calls.append(("keys", (tuple(keys), preferred_window_id)))
        return preferred_window_id or "0x410099"

    monkeypatch.setattr(adapter, "_send_keys_to_live_window", fake_send)
    monkeypatch.setattr("asil.adapters.nautilus.time.sleep", lambda seconds: calls.append(("sleep", seconds)))
    monkeypatch.setattr(
        "asil.adapters.nautilus.read_clipboard_text",
        lambda: f"file://{adapter.workspace_path}/Projects/alpha",
    )

    current_dir = adapter._read_gui_current_dir(SimpleNamespace(last_capture_window_id="0x410001"))

    assert current_dir == adapter.workspace_path / "Projects" / "alpha"
    assert calls == [
        ("keys", (("ctrl+l",), "0x410001")),
        ("sleep", 0.25),
        ("keys", (("ctrl+c",), "0x410001")),
        ("sleep", 0.15),
        ("keys", (("Escape",), "0x410001")),
    ]


def test_send_keys_to_live_window_recovers_from_stale_captured_window(tmp_path: Path, monkeypatch):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    direct_commands: list[list[str]] = []
    recovered_windows: list[dict[str, object]] = []

    monkeypatch.setattr("asil.adapters.nautilus.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("asil.adapters.nautilus.ensure_virtual_display", lambda: {"DISPLAY": ":199"})

    def fake_run(command, **_kwargs):
        direct_commands.append(command)
        if command == ["/usr/bin/xdotool", "windowactivate", "--sync", "0x410001"]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    def fake_wait_for_window(_title, **kwargs):
        recovered_windows.append(kwargs)
        return "0x410099"

    monkeypatch.setattr("asil.adapters.nautilus.subprocess.run", fake_run)
    monkeypatch.setattr("asil.adapters.nautilus.wait_for_window", fake_wait_for_window, raising=False)

    window_id = adapter._send_keys_to_live_window(["ctrl+l"], preferred_window_id="0x410001")

    assert window_id == "0x410099"
    assert direct_commands == [
        ["/usr/bin/xdotool", "windowactivate", "--sync", "0x410001"],
        ["/usr/bin/xdotool", "windowactivate", "--sync", "0x410099"],
        [
            "/usr/bin/xdotool",
            "key",
            "--clearmodifiers",
            "ctrl+l",
        ],
    ]
    assert len(recovered_windows) == 1
    assert recovered_windows[0]["window_class_pattern"] == "org.gnome.Nautilus|Org.gnome.Nautilus|nautilus"


def test_send_keys_to_live_window_targets_active_focus_not_synthetic_window_events(tmp_path: Path, monkeypatch):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path, mock=True)
    commands: list[list[str]] = []

    monkeypatch.setattr("asil.adapters.nautilus.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("asil.adapters.nautilus.ensure_virtual_display", lambda: {"DISPLAY": ":199"})
    monkeypatch.setattr(
        "asil.adapters.nautilus.subprocess.run",
        lambda command, **_kwargs: commands.append(command) or subprocess.CompletedProcess(command, 0),
    )

    window_id = adapter._send_keys_to_live_window(["ctrl+l"], preferred_window_id="0x410001")

    assert window_id == "0x410001"
    assert commands == [
        ["/usr/bin/xdotool", "windowactivate", "--sync", "0x410001"],
        ["/usr/bin/xdotool", "key", "--clearmodifiers", "ctrl+l"],
    ]


def test_read_gui_current_dir_decodes_gnome_file_clipboard_uri(tmp_path: Path, monkeypatch):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path / "root with spaces", mock=True)
    encoded_path = str(adapter.workspace_path / "Projects" / "alpha").replace(" ", "%20")

    monkeypatch.setattr(adapter, "_send_keys_to_live_window", lambda _keys, preferred_window_id=None: "0x1")
    monkeypatch.setattr("asil.adapters.nautilus.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "asil.adapters.nautilus.read_clipboard_text",
        lambda: f"x-special/gnome-copied-files\ncopy\nfile://{encoded_path}\n",
    )

    assert adapter._read_gui_current_dir() == adapter.workspace_path / "Projects" / "alpha"


def test_read_gui_bookmarks_decodes_percent_encoded_file_uris(tmp_path: Path):
    from asil.adapters.nautilus import NautilusAdapter

    adapter = NautilusAdapter.from_evaluation_context(tmp_path / "root with spaces", mock=True)
    bookmarks_file = adapter.state_path.parent / "_nautilus_home" / ".config" / "gtk-3.0" / "bookmarks"
    bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = str(adapter.workspace_path / "Projects" / "alpha").replace(" ", "%20")
    bookmarks_file.write_text(f"file://{encoded_path} Alpha\n", encoding="utf-8")

    assert adapter._read_gui_bookmarks() == ["Projects/alpha"]
