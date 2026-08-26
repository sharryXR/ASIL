"""ASIL adapter for OBS Studio — Pattern C (obs-websocket API)."""

from __future__ import annotations
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import time
import uuid
import html
from pathlib import Path
from typing import Any, Protocol

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    capture_html_to_png,
    capture_window_to_png,
    html_page,
    launch_gui_process,
    send_keys_to_window,
    terminate_process,
)


class WSClient(Protocol):
    """Minimal interface for obs-websocket client (for type checking + mocking)."""
    def call(self, request_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]: ...


class MockOBSWSClient:
    """In-memory OBS websocket mock for ground-truth evaluation without a running OBS instance."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "currentProgramSceneName": "Main Scene",
            "currentPreviewSceneName": "",
            "studioModeEnabled": False,
            "currentSceneCollectionName": "Default",
            "scenes": [
                {"sceneName": "Main Scene", "sceneIndex": 0},
                {"sceneName": "Intermission", "sceneIndex": 1},
                {"sceneName": "BRB", "sceneIndex": 2},
            ],
            "sceneItems": {
                "Main Scene": [
                    {"sourceName": "Webcam", "sceneItemId": 1, "sceneItemEnabled": True,
                     "sceneItemLocked": False, "inputKind": "v4l2_input"},
                    {"sourceName": "Game Capture", "sceneItemId": 2, "sceneItemEnabled": True,
                     "sceneItemLocked": False, "inputKind": "game_capture"},
                ],
                "Intermission": [],
                "BRB": [],
            },
            "inputs": [
                {"inputName": "Desktop Audio", "inputKind": "wasapi_output_capture",
                 "inputMuted": False, "inputVolumeDb": 0.0, "inputVolumeMul": 1.0},
                {"inputName": "Mic/Aux", "inputKind": "wasapi_input_capture",
                 "inputMuted": False, "inputVolumeDb": 0.0, "inputVolumeMul": 1.0},
            ],
            "streamActive": False,
            "recordActive": False,
            "videoSettings": {
                "fpsNumerator": 30, "fpsDenominator": 1,
                "baseWidth": 1920, "baseHeight": 1080,
                "outputWidth": 1920, "outputHeight": 1080,
            },
            "recordDirectory": "/tmp/obs_recording",
            "lastResponse": {},
        }

    def _ensure_scene(self, scene_name: str) -> None:
        if not scene_name:
            return
        s = self._state
        if any(scene["sceneName"] == scene_name for scene in s["scenes"]):
            return
        s["scenes"].append({"sceneName": scene_name, "sceneIndex": len(s["scenes"])})
        s["sceneItems"][scene_name] = [dict(item) for item in s["sceneItems"].get("Main Scene", [])]

    def _ensure_input(self, input_name: str, input_kind: str = "audio_input") -> dict[str, Any]:
        s = self._state
        for inp in s["inputs"]:
            if inp["inputName"] == input_name:
                return inp
        new_input = {
            "inputName": input_name,
            "inputKind": input_kind,
            "inputMuted": False,
            "inputVolumeDb": 0.0,
            "inputVolumeMul": 1.0,
        }
        s["inputs"].append(new_input)
        return new_input

    def call(self, request_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        d = data or {}
        s = self._state

        if request_type == "GetSceneList":
            return {
                "currentProgramSceneName": s["currentProgramSceneName"],
                "scenes": s["scenes"],
            }
        if request_type == "GetCurrentProgramScene":
            resp = {"currentProgramSceneName": s["currentProgramSceneName"]}
            s["lastResponse"] = resp
            return resp
        if request_type == "GetCurrentPreviewScene":
            return {"currentPreviewSceneName": s["currentPreviewSceneName"]}
        if request_type == "GetStudioModeEnabled":
            return {"studioModeEnabled": s["studioModeEnabled"]}
        if request_type == "SetCurrentProgramScene":
            scene_name = d.get("sceneName", s["currentProgramSceneName"])
            self._ensure_scene(scene_name)
            s["currentProgramSceneName"] = scene_name
            return {}
        if request_type == "SetCurrentPreviewScene":
            scene_name = d.get("sceneName", "")
            self._ensure_scene(scene_name)
            s["currentPreviewSceneName"] = scene_name
            return {}
        if request_type == "TriggerStudioModeTransition":
            if s["currentPreviewSceneName"]:
                s["currentProgramSceneName"] = s["currentPreviewSceneName"]
            return {}
        if request_type == "SetStudioModeEnabled":
            s["studioModeEnabled"] = d.get("studioModeEnabled", False)
            return {}
        if request_type == "GetSceneItemList":
            scene = d.get("sceneName", s["currentProgramSceneName"])
            return {"sceneItems": s["sceneItems"].get(scene, [])}
        if request_type == "SetSceneItemEnabled":
            scene = d.get("sceneName", s["currentProgramSceneName"])
            self._ensure_scene(scene)
            item_id = d.get("sceneItemId")
            enabled = d.get("sceneItemEnabled", True)
            for item in s["sceneItems"].get(scene, []):
                if item["sceneItemId"] == item_id:
                    item["sceneItemEnabled"] = enabled
            return {}
        if request_type == "GetStreamStatus":
            return {"outputActive": s["streamActive"], "outputDuration": 0}
        if request_type == "StartStream":
            s["streamActive"] = True
            return {}
        if request_type == "StopStream":
            s["streamActive"] = False
            return {}
        if request_type == "GetRecordStatus":
            return {"outputActive": s["recordActive"], "outputDuration": 0}
        if request_type == "StartRecord":
            s["recordActive"] = True
            return {}
        if request_type == "StopRecord":
            s["recordActive"] = False
            return {}
        if request_type == "GetInputList":
            return {"inputs": [{"inputName": i["inputName"], "inputKind": i["inputKind"]}
                                for i in s["inputs"]]}
        if request_type == "GetInputMute":
            name = d.get("inputName", "")
            return {"inputMuted": self._ensure_input(name)["inputMuted"]}
        if request_type == "SetInputMute":
            name = d.get("inputName", "")
            self._ensure_input(name)["inputMuted"] = d.get("inputMuted", False)
            return {}
        if request_type == "GetInputVolume":
            name = d.get("inputName", "")
            inp = self._ensure_input(name)
            return {"inputVolumeDb": inp["inputVolumeDb"],
                    "inputVolumeMul": inp["inputVolumeMul"]}
        if request_type == "SetInputVolume":
            name = d.get("inputName", "")
            inp = self._ensure_input(name)
            if "inputVolumeDb" in d:
                inp["inputVolumeDb"] = d["inputVolumeDb"]
            if "inputVolumeMul" in d:
                inp["inputVolumeMul"] = d["inputVolumeMul"]
            return {}
        if request_type == "GetVideoSettings":
            return dict(s["videoSettings"])
        if request_type == "SetVideoSettings":
            s["videoSettings"].update(d)
            return {}
        if request_type == "GetSceneCollectionList":
            return {
                "currentSceneCollectionName": s["currentSceneCollectionName"],
                "sceneCollections": ["Default", "Gaming Setup", "Podcast"],
            }
        if request_type == "SetCurrentSceneCollection":
            s["currentSceneCollectionName"] = d.get("sceneCollectionName",
                                                     s["currentSceneCollectionName"])
            return {}
        if request_type == "SetRecordDirectory":
            s["recordDirectory"] = d.get("recordDirectory", s["recordDirectory"])
            return {}
        if request_type == "CreateScene":
            name = d.get("sceneName", "")
            if name and not any(sc["sceneName"] == name for sc in s["scenes"]):
                s["scenes"].append({"sceneName": name, "sceneIndex": len(s["scenes"])})
                s["sceneItems"][name] = []
            return {}
        if request_type == "RemoveScene":
            name = d.get("sceneName", "")
            s["scenes"] = [sc for sc in s["scenes"] if sc["sceneName"] != name]
            s["sceneItems"].pop(name, None)
            return {}
        if request_type == "CreateInput":
            scene = d.get("sceneName", s["currentProgramSceneName"])
            source_name = d.get("inputName", d.get("sourceName", ""))
            kind = d.get("inputKind", "source")
            item_id = len(s["sceneItems"].get(scene, [])) + 1
            s["sceneItems"].setdefault(scene, []).append({
                "sourceName": source_name, "sceneItemId": item_id,
                "sceneItemEnabled": True, "sceneItemLocked": False, "inputKind": kind,
            })
            s["inputs"].append({"inputName": source_name, "inputKind": kind,
                                 "inputMuted": False, "inputVolumeDb": 0.0, "inputVolumeMul": 1.0})
            return {"sceneItemId": item_id}
        if request_type == "SetSceneItemTransform":
            return {}
        # Unknown — return empty
        return {}


class SimpleOBSWSClient:
    """Lightweight WebSocket client for obs-websocket v5 protocol.

    Connects to the obs-mock server (or any obs-websocket v5 server) using
    raw sockets. No external dependencies beyond the stdlib.
    """

    def __init__(self, host: str = "localhost", port: int = 4455) -> None:
        self._host = host
        self._port = port
        self._ws: Any = None  # websocket connection

    def _ensure_connected(self) -> None:
        if self._ws is not None:
            return
        import websockets.sync.client as wsc
        self._ws = wsc.connect(
            f"ws://{self._host}:{self._port}",
            open_timeout=5.0,
            close_timeout=5.0,
        )
        try:
            # Read Hello (op 0)
            hello = json.loads(self._ws.recv(timeout=5.0))
            assert hello["op"] == 0
            # Send Identify (op 1)
            self._ws.send(json.dumps({"op": 1, "d": {"rpcVersion": 1}}))
            # Read Identified (op 2)
            identified = json.loads(self._ws.recv(timeout=5.0))
            assert identified["op"] == 2
        except Exception:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            raise

    def call(self, request_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_connected()
        req_id = str(uuid.uuid4())
        msg = {
            "op": 6,
            "d": {
                "requestType": request_type,
                "requestId": req_id,
                "requestData": data or {},
            },
        }
        self._ws.send(json.dumps(msg))
        resp = json.loads(self._ws.recv())
        return resp.get("d", {}).get("responseData", {})

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


class CompatOBSWSClient:
    """obs-websocket client that normalizes v4/v5 responses to one shape."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4455,
        password: str = "",
        protocol: str = "auto",
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._protocol = protocol
        self._client: SimpleOBSWSClient | None = None
        self._v4_ws: Any = None
        self._request_counter = 0

    def _connect_v4(self) -> None:
        if self._v4_ws is not None:
            return
        import websockets.sync.client as wsc

        self._v4_ws = wsc.connect(f"ws://{self._host}:{self._port}")
        auth_info = self._v4_request("GetAuthRequired", {})
        if auth_info.get("authRequired"):
            secret = base64.b64encode(
                hashlib.sha256((self._password + auth_info["salt"]).encode("utf-8")).digest()
            ).decode("utf-8")
            auth = base64.b64encode(
                hashlib.sha256((secret + auth_info["challenge"]).encode("utf-8")).digest()
            ).decode("utf-8")
            self._v4_request("Authenticate", {"auth": auth})

    def _ensure_connected(self) -> None:
        if self._protocol == "v5":
            if self._client is None:
                self._client = SimpleOBSWSClient(host=self._host, port=self._port)
            self._client._ensure_connected()
            return
        if self._protocol == "v4":
            self._connect_v4()
            return

        try:
            self._client = SimpleOBSWSClient(host=self._host, port=self._port)
            self._client._ensure_connected()
            self._protocol = "v5"
        except Exception:
            self._client = None
            self._protocol = "v4"
            self._connect_v4()

    def _v4_request(self, request_type: str, data: dict[str, Any]) -> dict[str, Any]:
        self._request_counter += 1
        message_id = str(self._request_counter)
        payload = {
            "request-type": request_type,
            "message-id": message_id,
        }
        payload.update(data)
        self._v4_ws.send(json.dumps(payload))
        while True:
            response = json.loads(self._v4_ws.recv())
            if response.get("message-id") != message_id:
                continue
            status = response.get("status", "ok")
            if status != "ok":
                raise RuntimeError(response.get("error", f"OBS request {request_type} failed"))
            return response

    def _v4_scene_items(self, scene_name: str) -> list[dict[str, Any]]:
        scene_list = self._v4_request("GetSceneList", {})
        for scene in scene_list.get("scenes", []):
            if scene.get("name") != scene_name:
                continue
            items = []
            for index, source in enumerate(scene.get("sources", []), start=1):
                items.append(
                    {
                        "sourceName": source.get("name", ""),
                        "sceneItemId": index,
                        "sceneItemEnabled": source.get("render", True),
                        "sceneItemLocked": False,
                        "inputKind": source.get("type", "source"),
                    }
                )
            return items
        return []

    def _v4_source_name_from_item(self, scene_name: str, item_id: int) -> str:
        items = self._v4_scene_items(scene_name)
        for item in items:
            if item["sceneItemId"] == item_id:
                return str(item["sourceName"])
        raise RuntimeError(f"Scene item {item_id} not found in scene {scene_name}")

    def _call_v4(self, request_type: str, data: dict[str, Any]) -> dict[str, Any]:
        if request_type == "GetSceneList":
            response = self._v4_request("GetSceneList", {})
            scenes = response.get("scenes", [])
            return {
                "currentProgramSceneName": response.get("current-scene", ""),
                "scenes": [
                    {"sceneName": scene.get("name", ""), "sceneIndex": index}
                    for index, scene in enumerate(scenes)
                ],
            }
        if request_type == "GetCurrentProgramScene":
            response = self._v4_request("GetCurrentScene", {})
            return {"currentProgramSceneName": response.get("name", "")}
        if request_type == "GetCurrentPreviewScene":
            response = self._v4_request("GetPreviewScene", {})
            return {"currentPreviewSceneName": response.get("name", "")}
        if request_type == "GetStudioModeEnabled":
            response = self._v4_request("GetStudioModeStatus", {})
            return {"studioModeEnabled": response.get("studio-mode", False)}
        if request_type == "SetCurrentProgramScene":
            self._v4_request("SetCurrentScene", {"scene-name": data.get("sceneName", "")})
            return {}
        if request_type == "SetCurrentPreviewScene":
            self._v4_request("SetPreviewScene", {"scene-name": data.get("sceneName", "")})
            return {}
        if request_type == "TriggerStudioModeTransition":
            self._v4_request("TransitionToProgram", {})
            return {}
        if request_type == "SetStudioModeEnabled":
            if data.get("studioModeEnabled", False):
                self._v4_request("EnableStudioMode", {})
            else:
                self._v4_request("DisableStudioMode", {})
            return {}
        if request_type == "GetSceneItemList":
            return {"sceneItems": self._v4_scene_items(data.get("sceneName", ""))}
        if request_type == "SetSceneItemEnabled":
            scene_name = data.get("sceneName", "")
            source_name = self._v4_source_name_from_item(scene_name, int(data.get("sceneItemId", 0)))
            self._v4_request(
                "SetSceneItemRender",
                {
                    "scene-name": scene_name,
                    "source": source_name,
                    "render": data.get("sceneItemEnabled", True),
                },
            )
            return {}
        if request_type == "GetStreamStatus":
            response = self._v4_request("GetStreamingStatus", {})
            return {"outputActive": response.get("streaming", False), "outputDuration": 0}
        if request_type == "StartStream":
            self._v4_request("StartStreaming", {})
            return {}
        if request_type == "StopStream":
            self._v4_request("StopStreaming", {})
            return {}
        if request_type == "GetRecordStatus":
            try:
                response = self._v4_request("GetRecordingStatus", {})
                active = response.get("isRecording", response.get("recording", False))
            except Exception:
                active = False
            return {"outputActive": active, "outputDuration": 0}
        if request_type == "StartRecord":
            self._v4_request("StartRecording", {})
            return {}
        if request_type == "StopRecord":
            self._v4_request("StopRecording", {})
            return {}
        if request_type == "GetInputList":
            response = self._v4_request("GetSourcesList", {})
            return {
                "inputs": [
                    {
                        "inputName": source.get("name", ""),
                        "inputKind": source.get("typeId", source.get("type", "source")),
                    }
                    for source in response.get("sources", [])
                ]
            }
        if request_type == "GetInputMute":
            response = self._v4_request("GetMute", {"source": data.get("inputName", "")})
            return {"inputMuted": response.get("muted", False)}
        if request_type == "SetInputMute":
            self._v4_request(
                "SetMute",
                {"source": data.get("inputName", ""), "mute": data.get("inputMuted", False)},
            )
            return {}
        if request_type == "GetInputVolume":
            response = self._v4_request(
                "GetVolume",
                {"source": data.get("inputName", ""), "useDecibel": True},
            )
            volume = response.get("volume", 0.0)
            return {"inputVolumeDb": volume, "inputVolumeMul": 1.0}
        if request_type == "SetInputVolume":
            self._v4_request(
                "SetVolume",
                {
                    "source": data.get("inputName", ""),
                    "volume": data.get("inputVolumeDb", 0.0),
                    "useDecibel": "inputVolumeDb" in data,
                },
            )
            return {}
        if request_type == "GetSceneCollectionList":
            current = self._v4_request("GetCurrentSceneCollection", {})
            collections = self._v4_request("ListSceneCollections", {})
            return {
                "currentSceneCollectionName": current.get("sc-name", ""),
                "sceneCollections": collections.get("scene-collections", []),
            }
        if request_type == "SetCurrentSceneCollection":
            self._v4_request(
                "SetCurrentSceneCollection",
                {"sc-name": data.get("sceneCollectionName", "")},
            )
            return {}
        if request_type == "SetRecordDirectory":
            self._v4_request(
                "SetRecordingFolder",
                {"rec-folder": data.get("recordDirectory", "")},
            )
            return {}
        if request_type == "CreateScene":
            self._v4_request("CreateScene", {"sceneName": data.get("sceneName", "")})
            return {}
        if request_type == "CreateInput":
            self._v4_request(
                "CreateSource",
                {
                    "sceneName": data.get("sceneName", ""),
                    "sourceName": data.get("inputName", data.get("sourceName", "")),
                    "sourceKind": data.get("inputKind", "color_source"),
                    "setVisible": True,
                    "sourceSettings": data.get("inputSettings", {}),
                },
            )
            return {}
        return {}

    def call(self, request_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_connected()
        payload = data or {}
        if self._protocol == "v5":
            return self._client.call(request_type, payload)
        return self._call_v4(request_type, payload)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._v4_ws is not None:
            try:
                self._v4_ws.close()
            except Exception:
                pass
            self._v4_ws = None


class OBSAdapter(ASILAdapter):
    app_name = "OBS Studio"
    supported_action_types = ["api_call"]

    def __init__(
        self,
        ws: WSClient | None = None,
        host: str = "localhost",
        port: int = 4455,
        password: str = "",
        *,
        use_real_gui: bool | None = None,
        render_mock_ui: bool | None = None,
        obs_bin: str = "obs",
        ws_protocol: str = "auto",
    ) -> None:
        self._ws = ws
        self._host = host
        self._port = port
        self._password = password
        self._obs_bin = obs_bin
        self._ws_protocol = ws_protocol
        self._use_real_gui = (
            use_real_gui
            if use_real_gui is not None
            else os.environ.get("OBS_REAL_GUI", "").lower() in {"1", "true", "yes"}
        )
        self._render_mock_ui = (
            render_mock_ui if render_mock_ui is not None else not self._use_real_gui
        )
        if self._use_real_gui and self._port == 4455 and self._ws_protocol in {"auto", "v4"}:
            self._port = 4444
        self._obs_home = Path(tempfile.mkdtemp(prefix="asil_obs_home_")) if self._use_real_gui else None
        self._obs_proc: subprocess.Popen[str] | None = None
        self._scene_collections = ["Default", "Gaming Setup", "Podcast"]
        self._video_settings: dict[str, Any] = {
            "fpsNumerator": 30,
            "fpsDenominator": 1,
            "baseWidth": 1920,
            "baseHeight": 1080,
            "outputWidth": 1920,
            "outputHeight": 1080,
        }
        self._input_kinds = {
            "Desktop Audio": "pulse_output_capture",
            "Mic/Aux": "pulse_input_capture",
            "Webcam": "color_source",
            "Game Capture": "color_source",
        }
        self._current_scene_collection = "Default"
        self._ui_ready = False

    def get_context(self) -> dict[str, str]:
        return {"recording_path": "/tmp/obs_recording"}

    def _real_obs_extra_env(self) -> dict[str, str]:
        if self._obs_home is None:
            raise RuntimeError("OBS home is unavailable before real GUI initialization.")
        return {
            "HOME": str(self._obs_home),
            "QT_QPA_PLATFORM": "xcb",
            "LIBGL_ALWAYS_SOFTWARE": "1",
        }

    def get_gui_session_spec(self) -> GUISessionSpec | None:
        if not self._use_real_gui:
            return None
        self._prepare_real_obs_home()
        obs_bin = shutil.which(self._obs_bin)
        if obs_bin is None:
            raise RuntimeError("OBS Studio is not installed in the evaluation environment.")
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(
                obs_bin,
                "--disable-updater",
                "--disable-shutdown-check",
                "--disable-missing-files-check",
                "--multi",
            ),
            extra_env=self._real_obs_extra_env(),
            window_title_pattern=r".*OBS.*",
            window_class_pattern=r"obs|OBS",
            startup_timeout_s=60.0,
            post_launch_delay_s=1.0,
            post_launch_callback=lambda: self.observe(),
            ui_ready_probe=lambda session: self._wait_for_visible_obs_window(session),
            close_callback=self.reset_state,
            min_width=900,
            min_height=600,
        )

    def _wait_for_socket(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((self._host, self._port), timeout=1.0):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("Timed out waiting for OBS websocket to accept connections.")

    def _launch_real_obs(self) -> None:
        if not self._use_real_gui or self._obs_proc is not None:
            return
        obs_bin = shutil.which(self._obs_bin)
        if obs_bin is None:
            raise RuntimeError("OBS Studio is not installed in the evaluation environment.")
        self._prepare_real_obs_home()
        self._obs_proc = launch_gui_process(
            [
                obs_bin,
                "--disable-updater",
                "--disable-shutdown-check",
                "--disable-missing-files-check",
                "--multi",
            ],
            extra_env=self._real_obs_extra_env(),
        )
        self._ui_ready = False
        self._wait_for_socket()

    def _wait_for_visible_obs_window(
        self,
        session: Any | None,
        *,
        timeout: float = 12.0,
        poll_interval: float = 0.5,
    ) -> None:
        if session is None:
            return
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                probe_path = Path(handle.name)
            try:
                capture_complete = bool(session.capture(probe_path))
                if capture_complete and probe_path.exists():
                    self._ui_ready = True
                    return
            except Exception as exc:
                last_error = exc
            finally:
                probe_path.unlink(missing_ok=True)
            time.sleep(poll_interval)
        if last_error is not None:
            raise RuntimeError("OBS window remained visually blank after launch.") from last_error
        raise RuntimeError("OBS window did not become visually ready after launch.")

    def _prepare_real_obs_home(self) -> None:
        config_root = self._obs_home / ".config" / "obs-studio"
        (config_root / "basic" / "profiles" / "Untitled").mkdir(parents=True, exist_ok=True)
        (config_root / "basic" / "scenes").mkdir(parents=True, exist_ok=True)
        (config_root / "global.ini").write_text(
            textwrap.dedent(
                f"""\
                [General]
                Pre19Defaults=false
                Pre21Defaults=false
                Pre23Defaults=false
                Pre24.1Defaults=false
                FirstRun=false

                [Basic]
                Profile=Untitled
                ProfileDir=Untitled
                SceneCollection=Untitled
                SceneCollectionFile=Untitled

                [BasicWindow]
                gridMode=false

                [WebsocketAPI]
                ServerEnabled=true
                ServerPort={self._port}
                AuthRequired=false
                AlertsEnabled=false
                AuthSetupPrompted=true
                """
            ),
            encoding="utf-8",
        )
        (config_root / "basic" / "profiles" / "Untitled" / "basic.ini").write_text(
            "[General]\nName=Untitled\n",
            encoding="utf-8",
        )

    def _bootstrap_real_obs_state(self) -> None:
        ws = self.ws
        scene_list = ws.call("GetSceneList")
        existing_scenes = {scene["sceneName"] for scene in scene_list.get("scenes", [])}
        for scene_name in ("Main Scene", "Intermission", "BRB"):
            if scene_name not in existing_scenes:
                try:
                    ws.call("CreateScene", {"sceneName": scene_name})
                except Exception:
                    pass

        existing_inputs = {
            item.get("inputName", "")
            for item in ws.call("GetInputList").get("inputs", [])
        }
        for spec in (
            ("Main Scene", "Webcam", "color_source"),
            ("Main Scene", "Game Capture", "color_source"),
            ("Main Scene", "Desktop Audio", "pulse_output_capture"),
            ("Main Scene", "Mic/Aux", "pulse_input_capture"),
        ):
            scene_name, input_name, input_kind = spec
            if input_name in existing_inputs:
                continue
            try:
                ws.call(
                    "CreateInput",
                    {
                        "sceneName": scene_name,
                        "inputName": input_name,
                        "inputKind": input_kind,
                        "inputSettings": {},
                    },
                )
                existing_inputs.add(input_name)
            except Exception:
                continue

        try:
            ws.call("SetCurrentProgramScene", {"sceneName": "Main Scene"})
        except Exception:
            pass

    @property
    def ws(self) -> WSClient:
        if self._ws is None:
            if self._use_real_gui:
                if self._obs_proc is None:
                    try:
                        self._wait_for_socket(timeout=2.0)
                    except Exception:
                        self._launch_real_obs()
                self._ws = CompatOBSWSClient(
                    host=self._host,
                    port=self._port,
                    password=self._password,
                    protocol=self._ws_protocol,
                )
                self._bootstrap_real_obs_state()
            else:
                try:
                    import obsws_python as obsws
                    self._ws = obsws.ReqClient(host=self._host, port=self._port, password=self._password)
                except ImportError:
                    # Fallback: use lightweight WS client (works with obs-mock server)
                    self._ws = SimpleOBSWSClient(host=self._host, port=self._port)
        return self._ws

    def observe(self) -> Observation:
        scene_list = self.ws.call("GetSceneList")
        current_scene = scene_list.get("currentProgramSceneName", "")
        scenes = scene_list.get("scenes", [])
        try:
            preview_resp = self.ws.call("GetCurrentPreviewScene")
            preview_scene = preview_resp.get("currentPreviewSceneName", "")
        except Exception:
            preview_scene = ""
        try:
            studio_resp = self.ws.call("GetStudioModeEnabled")
            studio_mode = bool(studio_resp.get("studioModeEnabled", False))
        except Exception:
            studio_mode = False

        elements: list[Element] = []
        for scene in scenes:
            sname = scene["sceneName"]
            items = self.ws.call("GetSceneItemList", {"sceneName": sname})
            for item in items.get("sceneItems", []):
                elements.append(Element(
                    id=f"scene:{sname}/source:{item['sourceName']}",
                    type=item.get("inputKind", "source"),
                    label=item["sourceName"],
                    value={
                        "visible": item.get("sceneItemEnabled", True),
                        "locked": item.get("sceneItemLocked", False),
                        "scene": sname,
                    },
                    editable=True,
                    actions=["toggle_visible", "set_position", "set_size", "delete", "set_filter"],
                ))

        stream = self.ws.call("GetStreamStatus")
        record = self.ws.call("GetRecordStatus")
        try:
            video = self.ws.call("GetVideoSettings")
            if video:
                self._video_settings.update(video)
        except Exception:
            video = dict(self._video_settings)

        # Stream/Record status as elements for validation
        elements.append(Element(
            id="program_scene",
            type="scene_program",
            label=current_scene,
            value={"name": current_scene},
            editable=True,
            actions=["switch"],
        ))
        elements.append(Element(
            id="preview_scene",
            type="scene_preview",
            label=preview_scene,
            value={"name": preview_scene},
            editable=True,
            actions=["switch"],
        ))
        elements.append(Element(
            id="studio_mode",
            type="status",
            label="Studio Mode",
            value={"enabled": studio_mode},
            editable=True,
            actions=["enable", "disable"],
        ))
        elements.append(Element(
            id="stream_status",
            type="status",
            label="Stream Status",
            value={
                "active": stream.get("outputActive", False),
                "duration": stream.get("outputDuration", 0),
            },
            editable=False,
            actions=["start_stream", "stop_stream"],
        ))
        elements.append(Element(
            id="record_status",
            type="status",
            label="Record Status",
            value={
                "active": record.get("outputActive", False),
                "duration": record.get("outputDuration", 0),
            },
            editable=False,
            actions=["start_record", "stop_record"],
        ))
        elements.append(Element(
            id="video_settings",
            type="settings_group",
            label="Video Settings",
            value={
                "fps_numerator": int(video.get("fpsNumerator", 0)),
                "fps_denominator": int(video.get("fpsDenominator", 1)),
                "base_width": int(video.get("baseWidth", 0)),
                "base_height": int(video.get("baseHeight", 0)),
                "output_width": int(video.get("outputWidth", 0)),
                "output_height": int(video.get("outputHeight", 0)),
            },
            editable=True,
            actions=["set_video_settings"],
        ))

        # Input (audio) states
        try:
            inputs = self.ws.call("GetInputList")
            if not inputs.get("inputs"):
                inputs = {
                    "inputs": [
                        {"inputName": name, "inputKind": kind}
                        for name, kind in self._input_kinds.items()
                        if name in ("Desktop Audio", "Mic/Aux")
                    ]
                }
            for inp in inputs.get("inputs", []):
                name = inp.get("inputName", "")
                kind = inp.get("inputKind", "")
                try:
                    mute_resp = self.ws.call("GetInputMute", {"inputName": name})
                    vol_resp = self.ws.call("GetInputVolume", {"inputName": name})
                except Exception:
                    mute_resp = {}
                    vol_resp = {}
                elements.append(Element(
                    id=f"input:{name}",
                    type="audio_input",
                    label=name,
                    value={
                        "kind": kind,
                        "muted": mute_resp.get("inputMuted", False),
                        "volumeDb": vol_resp.get("inputVolumeDb", 0.0),
                        "volumeMul": vol_resp.get("inputVolumeMul", 1.0),
                    },
                    editable=True,
                    actions=["mute", "unmute", "set_volume"],
                ))
        except Exception:
            pass

        # Scene collection
        try:
            try:
                sc_resp = self.ws.call("GetSceneCollectionList")
                current_sc = sc_resp.get("currentSceneCollectionName", self._current_scene_collection)
                collections = sc_resp.get("sceneCollections", self._scene_collections)
            except Exception:
                current_sc = self._current_scene_collection
                collections = self._scene_collections
            elements.append(Element(
                id="scene_collection",
                type="scene_collection",
                label=current_sc,
                value={"name": current_sc, "collections": collections},
                editable=True,
                actions=["switch"],
            ))
        except Exception:
            pass

        return self._build_observation(
            source="rest_api",
            elements=elements,
            app_state={"current_view": current_scene},
            environment={
                "system": {
                    "base_width": float(video.get("baseWidth", 0)),
                    "base_height": float(video.get("baseHeight", 0)),
                }
            },
            data_summary=(
                f"OBS: {len(scenes)} scenes, {len(elements)} sources. "
                f"Streaming: {stream.get('outputActive', False)}, "
                f"Recording: {record.get('outputActive', False)}"
            ),
        )

    def execute(self, action: Action) -> Observation:
        method = action.params.get("method", "")
        args = action.params.get("args", {})
        if method == "SetVideoSettings":
            self._video_settings.update(args)
        if method == "SetCurrentSceneCollection":
            self._current_scene_collection = args.get("sceneCollectionName", self._current_scene_collection)
        if method in {"SetInputMute", "SetInputVolume"}:
            input_name = args.get("inputName", "")
            if input_name:
                self._input_kinds.setdefault(input_name, "audio_input")
        self.ws.call(method, args)
        return self.observe()

    def reset_state(self) -> None:
        """Reset OBS state between tasks."""
        if isinstance(self._ws, MockOBSWSClient):
            self._ws = MockOBSWSClient()
        elif isinstance(self._ws, SimpleOBSWSClient):
            # Ask obs-mock server to reset its state
            self._ws.call("_ASIL_ResetState")
        elif isinstance(self._ws, CompatOBSWSClient):
            self._ws.close()
            self._ws = None
            if self._obs_proc is not None:
                terminate_process(self._obs_proc)
                self._obs_proc = None
        # For real OBS connections, no-op (can't reset a live OBS instance)
        self._current_scene_collection = "Default"
        self._video_settings = {
            "fpsNumerator": 30,
            "fpsDenominator": 1,
            "baseWidth": 1920,
            "baseHeight": 1080,
            "outputWidth": 1920,
            "outputHeight": 1080,
        }
        self._ui_ready = False

    def setup_state(self, initial_state: str) -> None:
        if initial_state in ("", "default"):
            return

        if initial_state == "scene_intermission":
            self.ws.call("SetCurrentProgramScene", {"sceneName": "Intermission"})
            return

        if initial_state == "recording_active":
            self.ws.call("StartRecord", {})
            return

        if initial_state == "mic_muted":
            self.ws.call("SetInputMute", {"inputName": "Mic/Aux", "inputMuted": True})
            return

        if initial_state == "streaming_active":
            self.ws.call("StartStream", {})
            return

        raise ValueError(f"Unsupported OBS initial_state: {initial_state}")

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        if not self._render_mock_ui:
            return RenderArtifact(
                filename="",
                kind="app_window",
                backend="x11-window-capture",
                actual_page=True,
                description="Screenshot of the real OBS Studio window",
            )
        return RenderArtifact(
            filename="",
            kind="mock_ui_screenshot",
            backend="wkhtmltoimage",
            actual_page=False,
            description="Screenshot of a deterministic OBS-style page generated from websocket state",
        )

    def _render_obs_page(self, obs: Observation) -> str:
        groups: dict[str, list[Element]] = {}
        for elem in obs.interactive_elements:
            groups.setdefault(elem.type, []).append(elem)

        scene_rows = []
        for elem in groups.get("v4l2_input", []) + groups.get("source", []):
            value = elem.value if isinstance(elem.value, dict) else {}
            visible = "Visible" if value.get("visible", True) else "Hidden"
            locked = "Locked" if value.get("locked", False) else "Unlocked"
            scene_rows.append(
                f"<tr><td>{html.escape(str(elem.label))}</td>"
                f"<td><code>{html.escape(str(value.get('scene', '')))}</code></td>"
                f"<td>{visible}</td><td>{locked}</td></tr>"
            )
        if not scene_rows:
            scene_rows.append("<tr><td colspan='4'>No scene items</td></tr>")

        audio_rows = []
        for elem in groups.get("audio_input", []):
            value = elem.value if isinstance(elem.value, dict) else {}
            audio_rows.append(
                f"<tr><td>{html.escape(str(elem.label))}</td>"
                f"<td>{'Muted' if value.get('muted') else 'Live'}</td>"
                f"<td>{html.escape(str(value.get('volumeDb', 0.0)))} dB</td></tr>"
            )
        if not audio_rows:
            audio_rows.append("<tr><td colspan='3'>No audio inputs</td></tr>")

        status = {e.id: e.value for e in groups.get("status", [])}
        stream_state = status.get("stream_status", {}) if isinstance(status.get("stream_status", {}), dict) else {}
        record_state = status.get("record_status", {}) if isinstance(status.get("record_status", {}), dict) else {}
        preview_state = next(
            (
                e.value
                for e in groups.get("scene_preview", [])
                if isinstance(e.value, dict) and e.id == "preview_scene"
            ),
            {},
        )
        scene_collection = next(
            (
                e.value
                for e in groups.get("scene_collection", [])
                if isinstance(e.value, dict) and e.id == "scene_collection"
            ),
            {},
        )
        studio_state = status.get("studio_mode", {}) if isinstance(status.get("studio_mode", {}), dict) else {}
        base = obs.environment.system if hasattr(obs.environment, "system") else {}

        body = f"""
        <div style="display:grid;grid-template-columns:280px 1fr 320px;gap:18px;align-items:start;">
          <section class="panel" style="padding:18px;">
            <div class="badge">OBS Mock Workspace</div>
            <h1 style="margin:14px 0 8px;font-size:26px;">{html.escape(obs.app_state.current_view or 'Main Scene')}</h1>
            <p style="margin:0;color:#57606a;">Deterministic page render for per-step GUI comparison.</p>
            <div style="display:grid;gap:10px;margin-top:18px;">
              <div class="panel" style="padding:14px;">
                <strong>Streaming</strong>
                <div class="badge {'pill-success' if stream_state.get('active') else 'pill-danger'}" style="margin-left:8px;">
                  {'Active' if stream_state.get('active') else 'Stopped'}
                </div>
              </div>
              <div class="panel" style="padding:14px;">
                <strong>Recording</strong>
                <div class="badge {'pill-success' if record_state.get('active') else 'pill-danger'}" style="margin-left:8px;">
                  {'Active' if record_state.get('active') else 'Stopped'}
                </div>
              </div>
              <div class="panel" style="padding:14px;">
                <strong>Canvas</strong>
                <div style="margin-top:8px;color:#57606a;">{base.get('base_width', 0)} × {base.get('base_height', 0)}</div>
              </div>
              <div class="panel" style="padding:14px;">
                <strong>Studio Mode</strong>
                <div style="margin-top:8px;color:#57606a;">{'Enabled' if studio_state.get('enabled') else 'Disabled'}</div>
              </div>
              <div class="panel" style="padding:14px;">
                <strong>Scene Collection</strong>
                <div style="margin-top:8px;color:#57606a;">{html.escape(str(scene_collection.get('name', 'Default')))}</div>
              </div>
              <div class="panel" style="padding:14px;">
                <strong>Preview Scene</strong>
                <div style="margin-top:8px;color:#57606a;">{html.escape(str(preview_state.get('name', '')))}</div>
              </div>
            </div>
          </section>
          <section class="panel" style="padding:18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
              <h2 style="margin:0;font-size:20px;">Program Preview</h2>
              <span class="badge">{len(scene_rows)} scene items</span>
            </div>
            <div style="height:420px;border:1px solid #d0d7de;border-radius:12px;background:
              linear-gradient(135deg, rgba(9,105,218,0.18), transparent 40%),
              linear-gradient(180deg, #0d1117 0%, #1f2937 100%);
              position:relative;overflow:hidden;">
              <div style="position:absolute;inset:18px;border:1px dashed rgba(255,255,255,0.28);border-radius:10px;"></div>
              <div style="position:absolute;left:28px;top:28px;color:#f0f6fc;font-size:22px;font-weight:700;">{html.escape(obs.app_state.current_view or 'Scene')}</div>
              <div style="position:absolute;right:28px;bottom:28px;padding:10px 12px;border-radius:12px;background:rgba(15,23,42,0.72);color:#f0f6fc;">
                {len(groups.get('audio_input', []))} audio inputs
              </div>
            </div>
            <div class="panel" style="margin-top:16px;overflow:hidden;">
              <table>
                <thead><tr><th>Source</th><th>Scene</th><th>Visibility</th><th>Lock</th></tr></thead>
                <tbody>{''.join(scene_rows)}</tbody>
              </table>
            </div>
          </section>
          <section class="panel" style="padding:18px;overflow:hidden;">
            <h2 style="margin:0 0 12px;font-size:20px;">Audio Mixer</h2>
            <table>
              <thead><tr><th>Input</th><th>Status</th><th>Volume</th></tr></thead>
              <tbody>{''.join(audio_rows)}</tbody>
            </table>
          </section>
        </div>
        """
        return html_page("OBS Studio", body)

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        """Render current OBS state to a screenshot."""
        if not self._render_mock_ui:
            self._launch_real_obs()
            out = Path(output_path) if output_path else Path("obs_window.png")
            if not self._ui_ready:
                time.sleep(4.0)
                try:
                    send_keys_to_window("OBS", ["Escape"], timeout=20.0)
                except Exception:
                    pass
                self._ui_ready = True
            return capture_window_to_png(
                out,
                title_pattern="OBS",
                timeout=45.0,
                margin=12,
                settle_delay=2.0,
            )
        obs = self.observe()
        out = Path(output_path) if output_path else Path("obs_state.png")
        return capture_html_to_png(self._render_obs_page(obs), out)
