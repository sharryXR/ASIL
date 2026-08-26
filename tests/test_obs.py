"""Tests for OBS adapter — uses mock WebSocket, no OBS required."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from asil.adapters.obs import CompatOBSWSClient, OBSAdapter
from asil.protocol import Action


def _mock_ws() -> MagicMock:
    ws = MagicMock()
    ws.call.side_effect = _mock_call
    return ws


def _mock_call(request_type, data=None):
    responses = {
        "GetSceneList": {
            "scenes": [
                {"sceneName": "Main", "sceneIndex": 0},
                {"sceneName": "Gaming", "sceneIndex": 1},
            ],
            "currentProgramSceneName": "Main",
        },
        "GetCurrentPreviewScene": {"currentPreviewSceneName": "Gaming"},
        "GetStudioModeEnabled": {"studioModeEnabled": True},
        "GetSceneItemList": {
            "sceneItems": [
                {
                    "sceneItemId": 1,
                    "sourceName": "Webcam",
                    "inputKind": "video_capture",
                    "sceneItemEnabled": True,
                    "sceneItemLocked": False,
                },
                {
                    "sceneItemId": 2,
                    "sourceName": "Desktop",
                    "inputKind": "display_capture",
                    "sceneItemEnabled": False,
                    "sceneItemLocked": False,
                },
            ]
        },
        "GetStreamStatus": {"outputActive": False, "outputDuration": 0},
        "GetRecordStatus": {"outputActive": False},
        "GetVideoSettings": {"baseWidth": 1920, "baseHeight": 1080, "fpsNumerator": 30, "fpsDenominator": 1},
    }
    return responses.get(request_type, {})


def test_observe_returns_scenes_and_sources():
    adapter = OBSAdapter(ws=_mock_ws())
    obs = adapter.observe()

    assert obs.meta.app_name == "OBS Studio"
    assert obs.meta.observation_source == "rest_api"
    # 2 sources (Webcam, Desktop) x 2 scenes = 4 elements, but deduplication depends on impl
    # At minimum we expect at least 2 elements
    assert len(obs.interactive_elements) >= 2


def test_observe_extracts_source_properties():
    adapter = OBSAdapter(ws=_mock_ws())
    obs = adapter.observe()

    webcam = next((e for e in obs.interactive_elements if "Webcam" in e.label), None)
    assert webcam is not None
    assert webcam.value["visible"] is True
    assert webcam.type == "video_capture"


def test_observe_shows_stream_status():
    adapter = OBSAdapter(ws=_mock_ws())
    obs = adapter.observe()
    assert obs.app_state.current_view == "Main"
    preview = next(e for e in obs.interactive_elements if e.id == "preview_scene")
    studio = next(e for e in obs.interactive_elements if e.id == "studio_mode")
    video = next(e for e in obs.interactive_elements if e.id == "video_settings")
    assert preview.value["name"] == "Gaming"
    assert studio.value["enabled"] is True
    assert video.value["base_width"] == 1920


def test_execute_api_call():
    ws = _mock_ws()
    adapter = OBSAdapter(ws=ws)
    action = Action(
        action_type="api_call",
        target="obs-websocket",
        params={"method": "SetCurrentProgramScene", "args": {"sceneName": "Gaming"}},
    )
    adapter.execute(action)
    ws.call.assert_any_call("SetCurrentProgramScene", {"sceneName": "Gaming"})


def test_setup_state_initializes_non_success_obs_states():
    ws = _mock_ws()
    adapter = OBSAdapter(ws=ws)

    adapter.setup_state("scene_intermission")
    adapter.setup_state("recording_active")
    adapter.setup_state("mic_muted")
    adapter.setup_state("streaming_active")

    ws.call.assert_any_call("SetCurrentProgramScene", {"sceneName": "Intermission"})
    ws.call.assert_any_call("StartRecord", {})
    ws.call.assert_any_call("SetInputMute", {"inputName": "Mic/Aux", "inputMuted": True})
    ws.call.assert_any_call("StartStream", {})


def test_validate_action():
    adapter = OBSAdapter(ws=_mock_ws())
    good = Action(action_type="api_call", target="obs", params={})
    bad = Action(action_type="modify_file", target="x.svg", params={})
    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad)


def test_compat_v4_request_ignores_async_events():
    client = CompatOBSWSClient(protocol="v4")
    fake_ws = MagicMock()
    fake_ws.recv.side_effect = [
        '{"update-type":"SceneItemAdded","item-name":"Webcam"}',
        '{"message-id":"1","status":"ok","current-scene":"Main Scene","scenes":[]}',
    ]
    client._v4_ws = fake_ws

    resp = client._v4_request("GetSceneList", {})

    assert resp["current-scene"] == "Main Scene"
    fake_ws.send.assert_called_once()


@patch.object(CompatOBSWSClient, "_connect_v4")
def test_compat_auto_falls_back_to_v4_when_v5_hello_times_out(mock_connect_v4):
    fake_ws = MagicMock()
    fake_ws.recv.side_effect = TimeoutError("timed out waiting for OBS v5 hello")

    with patch("websockets.sync.client.connect", return_value=fake_ws):
        client = CompatOBSWSClient(protocol="auto")
        client._ensure_connected()

    mock_connect_v4.assert_called_once()
    fake_ws.recv.assert_called_once_with(timeout=5.0)


def test_describe_rendering_reports_real_window_capture():
    adapter = OBSAdapter(ws=_mock_ws(), use_real_gui=True)
    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "real obs studio window" in artifact.description.lower()


def test_gui_session_spec_launches_real_obs_binary():
    adapter = OBSAdapter(ws=_mock_ws(), use_real_gui=True)

    with patch("asil.adapters.obs.shutil.which", return_value="/usr/bin/obs"), patch.object(
        adapter, "_prepare_real_obs_home"
    ) as mock_prepare, patch.object(
        adapter, "_real_obs_extra_env", return_value={"HOME": "/tmp/obs-home"}
    ):
        spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.launch_command[:1] == ("/usr/bin/obs",)
    assert spec.extra_env["HOME"] == "/tmp/obs-home"
    assert spec.ui_ready_probe is not None
    mock_prepare.assert_called_once()


def test_wait_for_visible_obs_window_retries_until_capture_is_nonblank(tmp_path):
    adapter = OBSAdapter(ws=_mock_ws(), use_real_gui=True)

    session = MagicMock()
    session.capture.side_effect = [
        RuntimeError("blank screenshot"),
        True,
    ]

    with patch("asil.adapters.obs.time.sleep") as mock_sleep:
        adapter._wait_for_visible_obs_window(session, timeout=1.0, poll_interval=0.1)

    assert session.capture.call_count == 2
    assert adapter._ui_ready is True
    mock_sleep.assert_called_once_with(0.1)


def test_real_gui_ws_reuses_existing_obs_socket_without_relaunch():
    adapter = OBSAdapter(ws=None, use_real_gui=True)

    with patch.object(adapter, "_wait_for_socket") as mock_wait, patch.object(
        adapter, "_launch_real_obs"
    ) as mock_launch, patch("asil.adapters.obs.CompatOBSWSClient") as mock_client:
        client_instance = mock_client.return_value
        client_instance.call.return_value = {"scenes": [], "currentProgramSceneName": "Main Scene"}
        _ = adapter.ws

    mock_wait.assert_called_once_with(timeout=2.0)
    mock_launch.assert_not_called()
    mock_client.assert_called_once()


def test_launch_real_obs_uses_real_gui_env():
    adapter = OBSAdapter(ws=_mock_ws(), use_real_gui=True)

    with patch("asil.adapters.obs.shutil.which", return_value="/usr/bin/obs"), patch.object(
        adapter, "_prepare_real_obs_home"
    ) as mock_prepare, patch.object(
        adapter, "_real_obs_extra_env", return_value={"HOME": "/tmp/obs-home"}
    ) as mock_env, patch("asil.adapters.obs.launch_gui_process") as mock_launch, patch.object(
        adapter, "_wait_for_socket"
    ) as mock_wait:
        adapter._launch_real_obs()

    mock_prepare.assert_called_once()
    mock_env.assert_called_once()
    mock_launch.assert_called_once()
    assert mock_launch.call_args.kwargs["extra_env"] == {"HOME": "/tmp/obs-home"}
    mock_wait.assert_called_once()


@patch("asil.adapters.obs.capture_html_to_png")
def test_render_to_png_uses_html_screenshot_backend_only_in_explicit_mock_mode(mock_capture):
    adapter = OBSAdapter(ws=_mock_ws(), render_mock_ui=True)
    adapter.render_to_png("/tmp/obs.png")

    args, _ = mock_capture.call_args
    assert "OBS Mock Workspace" in args[0]
    assert "Gaming" in args[0]
    assert "Studio Mode" in args[0]
    assert args[1] == Path("/tmp/obs.png")


@patch("asil.adapters.obs.capture_window_to_png")
@patch("asil.adapters.obs.send_keys_to_window")
@patch("asil.adapters.obs.time.sleep")
@patch.object(OBSAdapter, "_launch_real_obs")
def test_render_to_png_real_gui_uses_window_capture(mock_launch, mock_sleep, mock_keys, mock_capture):
    adapter = OBSAdapter(ws=_mock_ws(), use_real_gui=True)
    mock_capture.return_value = Path("/tmp/obs-real.png")

    out = adapter.render_to_png("/tmp/obs-real.png")

    assert out == Path("/tmp/obs-real.png")
    mock_launch.assert_called_once()
    mock_sleep.assert_called_once_with(4.0)
    mock_keys.assert_called_once_with("OBS", ["Escape"], timeout=20.0)
    mock_capture.assert_called_once_with(
        Path("/tmp/obs-real.png"),
        title_pattern="OBS",
        timeout=45.0,
        margin=12,
        settle_delay=2.0,
    )
