import importlib.util
from pathlib import Path


def _load_obs_mock_server():
    path = Path(__file__).resolve().parent.parent / "docker" / "obs_mock_server.py"
    spec = importlib.util.spec_from_file_location("asil_obs_mock_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_obs_mock_server_reports_studio_mode_and_preview_scene():
    module = _load_obs_mock_server()
    module._reset_state()

    module._handle_request("SetStudioModeEnabled", {"studioModeEnabled": True})
    module._handle_request("SetCurrentPreviewScene", {"sceneName": "BRB"})

    assert module._handle_request("GetStudioModeEnabled", {}) == {"studioModeEnabled": True}
    assert module._handle_request("GetCurrentPreviewScene", {}) == {"currentPreviewSceneName": "BRB"}


def test_obs_mock_server_transitions_preview_scene_into_program_when_studio_mode_is_enabled():
    module = _load_obs_mock_server()
    module._reset_state()

    module._handle_request("SetStudioModeEnabled", {"studioModeEnabled": True})
    module._handle_request("SetCurrentPreviewScene", {"sceneName": "Intermission"})
    module._handle_request("TriggerStudioModeTransition", {})

    scene_list = module._handle_request("GetSceneList", {})
    assert scene_list["currentProgramSceneName"] == "Intermission"
