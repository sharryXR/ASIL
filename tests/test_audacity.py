import json
import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from asil.adapters.audacity import AudacityAdapter
from asil.eval.task_audit import audit_task_file
from asil.eval.runner import TaskDefinition, run_evaluation
from asil.protocol import Action


def _sample_state() -> dict:
    return {
        "project_name": "podcast_mix",
        "sample_rate_hz": 48000,
        "transport": {"playback": "stopped", "record_armed": False},
        "selection": {"start": 2.5, "end": 6.0, "focused_track_id": "track_voice"},
        "export": {"format": "wav", "filename": "podcast_mix.wav", "directory": "/tmp/exports"},
        "tracks": [
            {
                "id": "track_voice",
                "name": "Voiceover",
                "kind": "audio",
                "mute": False,
                "solo": False,
                "gain_db": -1.5,
                "pan": 0.0,
                "height_px": 148,
                "clips": [
                    {"id": "clip_voice_1", "name": "Intro Take", "start": 0.0, "end": 8.0, "color": "blue"}
                ],
            },
            {
                "id": "track_music",
                "name": "Music Bed",
                "kind": "audio",
                "mute": True,
                "solo": False,
                "gain_db": -8.0,
                "pan": -0.25,
                "height_px": 124,
                "clips": [
                    {"id": "clip_music_1", "name": "Theme Loop", "start": 0.0, "end": 15.0, "color": "green"}
                ],
            },
        ],
        "labels": [
            {"id": "label_intro", "text": "Intro", "start": 0.0, "end": 2.0},
            {"id": "label_edit", "text": "Tighten pause", "start": 4.0, "end": 4.8},
        ],
        "history": ["Open project", "Trim intro", "Rename tracks"],
    }


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "audacity"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


@pytest.fixture
def audacity_project(tmp_path: Path) -> Path:
    project_path = tmp_path / "audacity_project.json"
    project_path.write_text(json.dumps(_sample_state(), indent=2) + "\n", encoding="utf-8")
    return project_path


def test_observe_returns_tracks_labels_and_project_state(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)

    obs = adapter.observe()

    assert obs.meta.app_name == "Audacity"
    assert obs.meta.observation_source == "file_parse"
    assert obs.app_state.current_view == "multitrack_timeline"
    assert obs.app_state.active_document == "podcast_mix"
    assert obs.environment.system["sample_rate_hz"] == 48000.0
    ids = {element.id for element in obs.interactive_elements}
    assert {"track:track_voice", "track:track_music", "clip:clip_voice_1", "label:label_intro", "selection", "export_settings"}.issubset(ids)


def test_observe_exposes_visible_track_and_clip_properties(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)

    obs = adapter.observe()

    track = next(element for element in obs.interactive_elements if element.id == "track:track_music")
    clip = next(element for element in obs.interactive_elements if element.id == "clip:clip_music_1")
    assert track.value["name"] == "Music Bed"
    assert track.value["mute"] is True
    assert track.value["gain_db"] == -8.0
    assert clip.value["track_id"] == "track_music"
    assert clip.value["end"] == 15.0


def test_execute_updates_track_label_selection_and_export(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    action = Action(
        action_type="modify_file",
        target=str(audacity_project),
        params={
            "operations": [
                {"action": "set_track_value", "track_id": "track_music", "field": "name", "value": "Underscore"},
                {"action": "set_track_value", "track_id": "track_music", "field": "mute", "value": False},
                {"action": "set_selection", "start": 1.0, "end": 3.5, "focused_track_id": "track_music"},
                {"action": "set_label", "label_id": "label_edit", "text": "Shorten pause", "start": 3.9, "end": 4.3},
                {"action": "set_export", "field": "filename", "value": "episode_final.mp3"},
            ]
        },
    )

    obs = adapter.execute(action)

    track = next(element for element in obs.interactive_elements if element.id == "track:track_music")
    selection = next(element for element in obs.interactive_elements if element.id == "selection")
    label = next(element for element in obs.interactive_elements if element.id == "label:label_edit")
    export = next(element for element in obs.interactive_elements if element.id == "export_settings")
    assert track.value["name"] == "Underscore"
    assert track.value["mute"] is False
    assert selection.value == {"start": 1.0, "end": 3.5, "focused_track_id": "track_music"}
    assert label.value["text"] == "Shorten pause"
    assert export.value["filename"] == "episode_final.mp3"


def test_execute_can_add_label_and_adjust_clip_timing(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    action = Action(
        action_type="set_value",
        target="audacity_project",
        params={
            "operations": [
                {"action": "add_label", "label_id": "label_outro", "text": "Outro", "start": 12.0, "end": 14.0},
                {"action": "set_clip_value", "clip_id": "clip_voice_1", "field": "end", "value": 7.5},
            ]
        },
    )

    obs = adapter.execute(action)

    label = next(element for element in obs.interactive_elements if element.id == "label:label_outro")
    clip = next(element for element in obs.interactive_elements if element.id == "clip:clip_voice_1")
    assert label.value["text"] == "Outro"
    assert clip.value["end"] == 7.5


def test_execute_can_add_delete_and_reorder_tracks(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    action = Action(
        action_type="modify_file",
        target=str(audacity_project),
        params={
            "operations": [
                {
                    "action": "add_track",
                    "track_id": "track_room_tone",
                    "name": "Room Tone",
                    "kind": "audio",
                    "clips": [],
                },
                {"action": "delete_track", "track_id": "track_music"},
                {"action": "move_track", "track_id": "track_room_tone", "index": 0},
            ]
        },
    )

    obs = adapter.execute(action)
    ids = [element.id for element in obs.interactive_elements if element.id.startswith("track:")]
    labels = {element.id: element.label for element in obs.interactive_elements if element.id.startswith("track:")}

    assert ids[0] == "track:track_room_tone"
    assert "track:track_music" not in ids
    assert labels["track:track_room_tone"] == "Room Tone"


def test_validate_action_accepts_only_audacity_file_actions(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    good_modify = Action(action_type="modify_file", target=str(audacity_project), params={})
    good_set = Action(action_type="set_value", target="audacity_project", params={})
    bad = Action(action_type="api_call", target="obs-websocket", params={})

    assert adapter.validate_action(good_modify)
    assert adapter.validate_action(good_set)
    assert not adapter.validate_action(bad)


def test_describe_rendering_reports_real_window_capture(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)

    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "real audacity window" in artifact.description.lower()


def test_editor_window_score_prefers_main_audacity_window(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    preferred_titles = adapter._preferred_window_titles()

    main_score = adapter._editor_window_score("Audacity", preferred_titles, 938, 669)
    child_score = adapter._editor_window_score("Voiceover", preferred_titles, 940, 694)

    assert main_score > child_score


def test_is_editor_title_rejects_recovery_and_welcome_surfaces(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)

    assert adapter._is_editor_title("Audacity") is True
    assert adapter._is_editor_title("Welcome to Audacity!") is False
    assert adapter._is_editor_title("Automatic Crash Recovery") is False
    assert adapter._is_editor_title("Project Recovery") is False


@patch("asil.adapters.audacity.terminate_process")
@patch("asil.adapters.audacity.ensure_audio_backend")
@patch("asil.adapters.audacity.launch_gui_process")
@patch("asil.adapters.audacity.shutil.which", return_value="/usr/bin/audacity")
@patch.object(AudacityAdapter, "_capture_editor_window")
@patch.object(AudacityAdapter, "_prime_editor_window")
def test_render_to_png_uses_real_window_capture(
    mock_prime_editor_window,
    mock_capture_editor_window,
    mock_which,
    mock_launch,
    mock_audio_backend,
    mock_terminate,
    audacity_project: Path,
):
    adapter = AudacityAdapter(project_path=audacity_project)
    mock_launch.return_value = object()
    mock_capture_editor_window.side_effect = lambda output_path, **kwargs: (
        setattr(adapter, "_last_capture_complete", False) or Path(output_path)
    )
    adapter._prepare_gui_home = lambda _home_root: {"DISPLAY": ":99"}  # type: ignore[method-assign]
    adapter._script_pipe_paths = lambda username="asilgui": (Path("/tmp/to"), Path("/tmp/from"))  # type: ignore[method-assign]

    output = adapter.render_to_png("/tmp/audacity.png")

    assert output == Path("/tmp/audacity.png")
    mock_which.assert_called_once_with("audacity")
    mock_launch.assert_called_once()
    launch_args, launch_kwargs = mock_launch.call_args
    assert launch_args[0][0] == "/usr/bin/audacity"
    assert len(launch_args[0]) == 1
    assert launch_kwargs["run_as_user"] == "asilgui"
    mock_audio_backend.assert_called_once_with(run_as_user="asilgui")
    mock_prime_editor_window.assert_called_once_with()
    capture_args, capture_kwargs = mock_capture_editor_window.call_args
    assert capture_args == (Path("/tmp/audacity.png"),)
    assert set(capture_kwargs["preferred_titles"]) == {"podcast_mix", "voiceover", "music bed", "audacity_project"}


def test_sync_from_gui_updates_tracks_from_runtime_snapshot(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)

    monkeypatch.setattr(
        adapter,
        "_read_gui_track_snapshot",
        lambda: [
            {"id": "track_voice", "name": "Host Narration"},
            {"id": "track_cta_fx", "name": "CTA FX"},
        ],
    )

    adapter.sync_from_gui()
    state = adapter._read_state()

    assert [track["id"] for track in state["tracks"]] == ["track_voice", "track_cta_fx"]
    assert state["tracks"][0]["name"] == "Host Narration"
    assert state["tracks"][1]["name"] == "CTA FX"
    assert adapter._last_capture_complete is False


def test_sync_from_gui_preserves_track_id_for_renamed_visible_track(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)

    monkeypatch.setattr(
        adapter,
        "_read_gui_track_snapshot",
        lambda: [
            {"name": "Voiceover"},
            {"name": "Narration Bed"},
        ],
    )

    adapter.sync_from_gui()
    state = adapter._read_state()

    assert [track["id"] for track in state["tracks"]] == ["track_voice", "track_music"]
    assert state["tracks"][1]["name"] == "Narration Bed"


def test_sync_from_gui_handles_delete_and_rename_in_track_order(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)

    monkeypatch.setattr(
        adapter,
        "_read_gui_track_snapshot",
        lambda: [
            {"name": "Narration"},
        ],
    )

    adapter.sync_from_gui()
    state = adapter._read_state()

    assert [track["id"] for track in state["tracks"]] == ["track_voice"]
    assert state["tracks"][0]["name"] == "Narration"


def test_sync_from_gui_handles_delete_and_add_without_reusing_deleted_track_id(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)

    monkeypatch.setattr(
        adapter,
        "_read_gui_track_snapshot",
        lambda: [
            {"name": "Main Narration"},
            {"name": "Ambience"},
        ],
    )

    adapter.sync_from_gui()
    state = adapter._read_state()

    assert [track["id"] for track in state["tracks"]] == ["track_voice", "track_ambience"]
    assert state["tracks"][0]["name"] == "Main Narration"
    assert state["tracks"][1]["name"] == "Ambience"


def test_sync_from_gui_prefers_later_exact_match_over_renaming_first_track(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)

    monkeypatch.setattr(
        adapter,
        "_read_gui_track_snapshot",
        lambda: [
            {"name": "Music Bed"},
        ],
    )

    adapter.sync_from_gui()
    state = adapter._read_state()

    assert [track["id"] for track in state["tracks"]] == ["track_music"]
    assert state["tracks"][0]["name"] == "Music Bed"


@pytest.mark.parametrize(
    ("raw_line", "expected"),
    [
        ("Voiceover WI 1 0", "Voiceover"),
        ("Narration BEW 1 0", "Narration Bed"),
        ("Narration W 1 0", "Narration"),
        ("Main Narratiw 1 0", "Main Narration"),
        ("Ambience WI 1 0", "Ambience"),
        ("WW MAE OR", ""),
        ("I", ""),
        ("Mono 44100hz 0 S", ""),
    ],
)
def test_candidate_track_name_from_ocr_line_cleans_common_audacity_noise(raw_line: str, expected: str):
    assert AudacityAdapter._candidate_track_name_from_ocr_line(raw_line) == expected


def test_get_gui_session_spec_primes_editor_window(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)
    calls: list[str] = []

    monkeypatch.setattr("asil.adapters.audacity.shutil.which", lambda name: "/usr/bin/audacity")
    monkeypatch.setattr("asil.adapters.audacity.ensure_user_access", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.adapters.audacity.ensure_audio_backend", lambda **kwargs: None)
    monkeypatch.setattr(adapter, "_prepare_gui_home", lambda _home_root: {"DISPLAY": ":99"})
    monkeypatch.setattr(adapter, "_script_pipe_paths", lambda username="asilgui": (Path("/tmp/to"), Path("/tmp/from")))
    monkeypatch.setattr(adapter, "_dismiss_welcome_dialog", lambda timeout=10.0: calls.append(f"dismiss:{timeout}"))
    monkeypatch.setattr(adapter, "_wait_for_editor_window_id", lambda timeout=60.0: calls.append(f"wait:{timeout}") or "0x1")
    monkeypatch.setattr(adapter, "_focus_editor_window", lambda window_id: calls.append(f"focus:{window_id}"))
    monkeypatch.setattr(adapter, "_wait_for_script_pipe", lambda timeout=20.0: calls.append(f"pipe:{timeout}") or ("/tmp/to", "/tmp/from"))
    monkeypatch.setattr(adapter, "_apply_state_via_script_pipe", lambda *_args, **_kwargs: calls.append("apply_state"))

    spec = adapter.get_gui_session_spec()
    assert spec.post_launch_callback is not None
    spec.post_launch_callback()

    assert calls == ["dismiss:10.0", "wait:60.0", "focus:0x1", "pipe:20.0", "apply_state", "dismiss:2.0"]


def test_run_script_pipe_command_reads_until_blank_line(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    pipe_in = io.StringIO("OK: NewMonoTrack\nBatchCommand finished: OK\n\n")
    pipe_out = io.StringIO()

    response = adapter._run_script_pipe_command(pipe_out, pipe_in, "NewMonoTrack:")

    assert pipe_out.getvalue() == "NewMonoTrack:\n"
    assert response == "OK: NewMonoTrack\nBatchCommand finished: OK"


def test_parse_track_snapshot_from_pipe_response_extracts_exact_track_fields(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    response = """
[
  {"name": "Lead VO", "mute": false, "solo": true, "gain": -3, "pan": 0},
  {"Name": "Theme Bed", "Mute": true, "Solo": false, "Gain": -8.5, "Pan": -0.25}
]
BatchCommand finished: OK
""".strip()

    snapshot = adapter._parse_track_snapshot_from_pipe_response(response)

    assert snapshot == [
        {"name": "Lead VO", "mute": False, "solo": True, "gain_db": -3.0, "pan": 0.0},
        {"name": "Theme Bed", "mute": True, "solo": False, "gain_db": -8.5, "pan": -0.25},
    ]


def test_read_gui_track_snapshot_prefers_script_pipe_over_ocr(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)

    monkeypatch.setattr(
        adapter,
        "_read_gui_track_snapshot_via_script_pipe",
        lambda: [{"name": "Lead VO"}, {"name": "Theme Bed"}],
    )
    monkeypatch.setattr(adapter, "_capture_editor_window", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ocr fallback used")))

    snapshot = adapter._read_gui_track_snapshot()

    assert snapshot == [{"name": "Lead VO"}, {"name": "Theme Bed"}]


def test_cleanup_script_pipe_files_removes_stale_fifos(audacity_project: Path, tmp_path: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)
    pipe_to = tmp_path / "audacity_script_pipe.to.1000"
    pipe_from = tmp_path / "audacity_script_pipe.from.1000"
    pipe_to.touch()
    pipe_from.touch()

    monkeypatch.setattr(adapter, "_script_pipe_paths", lambda username="asilgui": (pipe_to, pipe_from))

    adapter._cleanup_script_pipe_files()

    assert not pipe_to.exists()
    assert not pipe_from.exists()


def test_wait_for_script_pipe_returns_when_paths_exist(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)
    pipe_to = Path("/tmp/audacity_script_pipe.to.1000")
    pipe_from = Path("/tmp/audacity_script_pipe.from.1000")

    monkeypatch.setattr(adapter, "_script_pipe_paths", lambda username="asilgui": (pipe_to, pipe_from))
    monkeypatch.setattr(adapter, "_script_pipe_ready_error", lambda *_args: None)
    monkeypatch.setattr("asil.adapters.audacity.Path.exists", lambda self: self in {pipe_to, pipe_from})
    monkeypatch.setattr("asil.adapters.audacity.time.sleep", lambda _seconds: None)
    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr("asil.adapters.audacity.time.time", lambda: next(times))

    observed_to, observed_from = adapter._wait_for_script_pipe(timeout=1.0)

    assert observed_to == pipe_to
    assert observed_from == pipe_from


def test_wait_for_script_pipe_repairs_permission_denied_once(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)
    pipe_to = Path("/tmp/audacity_script_pipe.to.1000")
    pipe_from = Path("/tmp/audacity_script_pipe.from.1000")
    ready_errors = iter([f"Permission denied opening {pipe_to}", None])
    repairs: list[tuple[Path, Path]] = []

    monkeypatch.setattr(adapter, "_script_pipe_paths", lambda username="asilgui": (pipe_to, pipe_from))
    monkeypatch.setattr(adapter, "_script_pipe_ready_error", lambda *_args: next(ready_errors))
    monkeypatch.setattr(adapter, "_make_script_pipe_accessible", lambda *args: repairs.append(args))
    monkeypatch.setattr("asil.adapters.audacity.Path.exists", lambda self: self in {pipe_to, pipe_from})
    monkeypatch.setattr("asil.adapters.audacity.time.sleep", lambda _seconds: None)
    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr("asil.adapters.audacity.time.time", lambda: next(times))

    observed_to, observed_from = adapter._wait_for_script_pipe(timeout=1.0)

    assert observed_to == pipe_to
    assert observed_from == pipe_from
    assert repairs == [(pipe_to, pipe_from)]


def test_apply_state_via_script_pipe_opens_write_before_read(audacity_project: Path, monkeypatch):
    adapter = AudacityAdapter(project_path=audacity_project)
    opened: list[tuple[str, str]] = []

    class _FakeWriter(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeReader(io.StringIO):
        def __init__(self):
            super().__init__("OK: NewMonoTrack\nBatchCommand finished: OK\n\nOK: SetTrackStatus\nBatchCommand finished: OK\n\n")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open_pair(pipe_to, pipe_from):
        opened.append((str(pipe_to), "w"))
        opened.append((str(pipe_from), "r"))
        return _FakeWriter(), _FakeReader()

    def fake_open(path, mode="r", *args, **kwargs):
        opened.append((str(path), mode))
        if "w" in mode:
            return _FakeWriter()
        return _FakeReader()

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(adapter, "_open_script_pipe_pair", fake_open_pair)
    monkeypatch.setattr(adapter, "_read_state", lambda: {"tracks": [{"name": "Voiceover"}]})
    adapter._apply_state_via_script_pipe("/tmp/to", "/tmp/from")

    assert opened[:2] == [("/tmp/to", "w"), ("/tmp/from", "r")]


def test_open_script_pipe_pair_never_blocks_while_reopening_checked_fifos(
    audacity_project: Path,
    monkeypatch,
):
    adapter = AudacityAdapter(project_path=audacity_project)
    opened: list[tuple[str, int]] = []
    blocking_calls: list[tuple[int, bool]] = []
    fds = iter([101, 102])

    def fake_os_open(path, flags):
        opened.append((str(path), flags))
        assert flags & os.O_NONBLOCK
        return next(fds)

    monkeypatch.setattr("asil.adapters.audacity.os.open", fake_os_open)
    monkeypatch.setattr(
        "asil.adapters.audacity.os.set_blocking",
        lambda fd, blocking: blocking_calls.append((fd, blocking)),
    )
    monkeypatch.setattr(
        "asil.adapters.audacity.os.fdopen",
        lambda fd, *args, **kwargs: (fd, args, kwargs),
    )

    write_handle, read_handle = adapter._open_script_pipe_pair("/tmp/to", "/tmp/from")

    assert opened == [
        ("/tmp/to", os.O_WRONLY | os.O_NONBLOCK),
        ("/tmp/from", os.O_RDONLY | os.O_NONBLOCK),
    ]
    assert blocking_calls == [(101, True), (102, True)]
    assert write_handle[0] == 101
    assert read_handle[0] == 102


def test_capture_window_by_id_recovers_when_window_id_goes_stale(audacity_project: Path, monkeypatch, tmp_path: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    out = tmp_path / "audacity.png"

    first = {"done": False}

    def fake_window_geometry(window_id: str):
        if window_id == "0xc00092" and not first["done"]:
            first["done"] = True
            raise RuntimeError("stale window")
        if window_id == "0xc00099":
            return (0, 0, 640, 360)
        raise RuntimeError(f"unexpected window id {window_id}")

    def fake_search_window_ids(_pattern: str):
        return ["0xc00099"]

    def fake_window_title(window_id: str):
        return {"0xc00099": "Audacity"}.get(window_id, "")

    monkeypatch.setattr(adapter, "_window_geometry", fake_window_geometry)
    monkeypatch.setattr(adapter, "_search_window_ids", fake_search_window_ids)
    monkeypatch.setattr(adapter, "_window_title", fake_window_title)
    monkeypatch.setattr(adapter, "_gui_env", lambda: {"DISPLAY": ":99"})
    def fake_which(name: str):
        if name == "import":
            return "/usr/bin/import"
        if name == "xdotool":
            return "/usr/bin/xdotool"
        if name == "xwininfo":
            return "/usr/bin/xwininfo"
        if name == "xdpyinfo":
            return "/usr/bin/xdpyinfo"
        return f"/usr/bin/{name}"

    monkeypatch.setattr("asil.adapters.audacity.shutil.which", fake_which)
    monkeypatch.setattr("asil.adapters.audacity.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("asil.adapters.audacity.assert_png_not_blank", lambda _path: None)

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/xdpyinfo":
            return None
        if cmd[0] == "/usr/bin/import" and "-display" in cmd and "-window" in cmd:
            Image.new("RGB", (800, 600), "white").save(Path(cmd[-1]))
            return None
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("asil.adapters.audacity.subprocess.run", fake_run)

    adapter._capture_window_by_id("0xc00092", out, preferred_titles={"podcast_mix"})

    assert out.exists()


@patch("asil.adapters.audacity.terminate_process")
@patch("asil.adapters.audacity.ensure_audio_backend")
@patch("asil.adapters.audacity.launch_gui_process")
@patch("asil.adapters.audacity.shutil.which", return_value="/usr/bin/audacity")
@patch.object(AudacityAdapter, "_capture_editor_window")
@patch.object(AudacityAdapter, "_prime_editor_window")
def test_render_to_png_prefers_editor_window_capture_over_generic_title_matching(
    mock_prime_editor_window,
    mock_capture_editor_window,
    mock_which,
    mock_launch,
    mock_audio_backend,
    mock_terminate,
    audacity_project: Path,
):
    adapter = AudacityAdapter(project_path=audacity_project)
    mock_launch.return_value = object()
    adapter._prepare_gui_home = lambda _home_root: {"DISPLAY": ":99"}  # type: ignore[method-assign]
    adapter._script_pipe_paths = lambda username="asilgui": (Path("/tmp/to"), Path("/tmp/from"))  # type: ignore[method-assign]

    output = adapter.render_to_png("/tmp/audacity-editor.png")

    assert output == Path("/tmp/audacity-editor.png")
    mock_which.assert_called_once_with("audacity")
    mock_launch.assert_called_once()
    mock_audio_backend.assert_called_once_with(run_as_user="asilgui")
    mock_prime_editor_window.assert_called_once_with()
    capture_args, capture_kwargs = mock_capture_editor_window.call_args
    assert capture_args == (Path("/tmp/audacity-editor.png"),)
    assert set(capture_kwargs["preferred_titles"]) == {"podcast_mix", "voiceover", "music bed", "audacity_project"}
    mock_terminate.assert_called_once_with(mock_launch.return_value)


def test_task_definition_loads_audacity_example():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "audacity"
    tasks = TaskDefinition.from_json(root / "audacity_01.json")

    assert tasks[0].software == "audacity"
    assert tasks[0].initial_state == "default"
    assert tasks[0].gui_expectations["success_surface"] == "multitrack_timeline"


def test_audacity_examples_have_twenty_tasks_and_gui_visible_rules():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "audacity"
    files = sorted(path for path in root.glob("audacity_*.json") if path.stem.removeprefix("audacity_").isdigit())

    assert len(files) == 20
    for path in files:
        task = json.loads(path.read_text(encoding="utf-8"))
        checkpoints = task["evaluator"]["paths"][0]["checkpoints"]
        assert checkpoints
        assert all(checkpoint["gui_visible_required"] for checkpoint in checkpoints)
        assert task["_asil"]["software"] == "audacity"
        assert task["gui_expectations"]["success_surface"] == "multitrack_timeline"
        assert task["related_apps"] == ["audacity"]


def test_audacity_examples_focus_on_track_header_changes_only():
    banned_snippets = [
        '"id": "selection"',
        '"id": "transport"',
        '"id": "export_settings"',
        '"type": "label"',
        '"field": "mute"',
        '"field": "solo"',
        '"field": "gain_db"',
        '"field": "pan"',
        '"action": "set_selection"',
        '"action": "set_export"',
        '"action": "set_transport"',
        '"action": "set_label"',
        '"action": "add_label"',
        '"action": "set_clip_value"',
        '"key": "mute"',
        '"key": "solo"',
        '"key": "gain_db"',
        '"key": "pan"',
        '"key": "start"',
        '"key": "end"',
        '"key": "format"',
        '"key": "filename"',
        '"key": "directory"',
        '"key": "record_armed"',
        '"key": "playback"',
    ]

    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "audacity"
    for path in sorted(path for path in root.glob("audacity_*.json") if path.stem.removeprefix("audacity_").isdigit()):
        task = json.loads(path.read_text(encoding="utf-8"))
        task_json = json.dumps(task, sort_keys=True)
        assert any(
            token in task_json
            for token in ['"action": "set_track_value"', '"action": "add_track"', '"action": "delete_track"']
        ), path.name
        assert '"visible_in_track_headers"' in task_json, path.name
        for snippet in banned_snippets:
            assert snippet not in task_json, f"{path.name} still includes forbidden benchmark surface {snippet}"


def test_audacity_examples_use_visible_track_name_rules():
    simple_task = _task("audacity_01")
    single_track_task = _task("audacity_10")
    dual_track_task = _task("audacity_20")

    simple_rules = [checkpoint["rule"] for checkpoint in simple_task["evaluator"]["paths"][0]["checkpoints"]]
    single_track_rules = [checkpoint["rule"] for checkpoint in single_track_task["evaluator"]["paths"][0]["checkpoints"]]
    dual_track_rules = [checkpoint["rule"] for checkpoint in dual_track_task["evaluator"]["paths"][0]["checkpoints"]]

    assert {"element_value": {"id": "track:track_music", "key": "name", "expected": "Narration Bed"}} in simple_rules
    assert {"any_element_matches": {"type": "track", "id": "track:track_sponsor_read", "value": {"name": "Sponsor Read"}}} in single_track_rules
    assert {"any_element_matches": {"type": "track", "id": "track:track_cta_fx", "value": {"name": "CTA FX"}}} in dual_track_rules
    assert {"element_value": {"id": "track:track_voice", "key": "name", "expected": "Host Narration"}} in dual_track_rules


def test_audacity_schema_includes_add_delete_and_move_track_examples():
    schema_path = Path(__file__).resolve().parent.parent / "src" / "asil" / "action_schemas" / "audacity.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_json = json.dumps(schema, sort_keys=True)

    assert '"action": "add_track"' in schema_json
    assert '"action": "delete_track"' in schema_json
    assert '"action": "move_track"' in schema_json


def test_audacity_examples_pass_static_gui_audit():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "audacity"

    for task_path in sorted(path for path in root.glob("audacity_*.json") if path.stem.removeprefix("audacity_").isdigit()):
        report = audit_task_file(task_path)
        assert report.ok, f"{task_path.name}: {report.errors}"


def test_run_evaluation_executes_audacity_ground_truth(audacity_project: Path):
    adapter = AudacityAdapter(project_path=audacity_project)
    task = TaskDefinition(
        id="audacity_eval",
        software="audacity",
        difficulty="simple",
        description="Rename the music track",
        actions=[
            {
                "action_type": "modify_file",
                "target": "{{project_path}}",
                "params": {
                    "operations": [
                        {"action": "set_track_value", "track_id": "track_music", "field": "name", "value": "Narration Bed"}
                    ]
                },
            }
        ],
        validation={"element_value": {"id": "track:track_music", "key": "name", "expected": "Narration Bed"}},
        initial_state="default",
    )

    results = run_evaluation(adapter, [task], isolate_tasks=True)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].score == 1.0
