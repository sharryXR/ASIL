"""Minimal OBS WebSocket mock server (obs-websocket protocol v5).

Simulates the obs-websocket JSON-RPC API so that OBSAdapter and evaluation
tasks can run in CI / supercomputer environments without a real OBS instance.

State is in-memory and resets on restart. Supports all request types used
by the ASIL benchmark tasks (obs_01 – obs_20).
"""

import asyncio
import copy
import json
import uuid
import websockets

# ── Default OBS state (copied on reset) ─────────────────────────────────────

_DEFAULT_STATE = {
    "scenes": [
        {"sceneName": "Main Scene", "sceneIndex": 0},
        {"sceneName": "Intermission", "sceneIndex": 1},
        {"sceneName": "BRB", "sceneIndex": 2},
    ],
    "currentProgramSceneName": "Main Scene",
    "currentPreviewSceneName": "",
    "studioModeEnabled": False,
    "currentSceneCollection": "Default",
    "sceneCollections": ["Default", "Gaming Setup", "Podcast"],
    "streaming": False,
    "recording": False,
    "recordDirectory": "/tmp/obs_recording",
    "video": {
        "baseWidth": 1920,
        "baseHeight": 1080,
        "outputWidth": 1920,
        "outputHeight": 1080,
        "fpsNumerator": 30,
        "fpsDenominator": 1,
    },
    "inputs": {
        "Desktop Audio": {"muted": False, "volumeDb": 0.0, "volumeMul": 1.0},
        "Mic/Aux": {"muted": False, "volumeDb": 0.0, "volumeMul": 1.0},
    },
    "sceneItems": {
        "Main Scene": [
            {"sceneItemId": 1, "sourceName": "Webcam", "inputKind": "v4l2_input",
             "sceneItemEnabled": True, "sceneItemLocked": False},
            {"sceneItemId": 2, "sourceName": "Game Capture", "inputKind": "game_capture",
             "sceneItemEnabled": True, "sceneItemLocked": False},
        ],
        "Intermission": [],
        "BRB": [],
    },
}

# ── Live state (mutated by requests, reset via _ASIL_ResetState) ────────────

_state = copy.deepcopy(_DEFAULT_STATE)


def _reset_state():
    """Reset to default state (called between tasks)."""
    global _state
    _state = copy.deepcopy(_DEFAULT_STATE)


def _handle_request(req_type: str, data: dict) -> dict:
    """Dispatch a request type and return the responseData dict."""
    d = data or {}

    # ── ASIL extension: reset state between tasks ──
    if req_type == "_ASIL_ResetState":
        _reset_state()
        return {"reset": True}

    if req_type == "GetSceneList":
        return {
            "currentProgramSceneName": _state["currentProgramSceneName"],
            "currentPreviewSceneName": _state["currentPreviewSceneName"],
            "scenes": _state["scenes"],
        }

    if req_type == "GetCurrentProgramScene":
        return {"currentProgramSceneName": _state["currentProgramSceneName"]}

    if req_type == "SetCurrentProgramScene":
        name = d.get("sceneName", "")
        if any(s["sceneName"] == name for s in _state["scenes"]):
            _state["currentProgramSceneName"] = name
        return {}

    if req_type == "SetCurrentPreviewScene":
        name = d.get("sceneName", "")
        _state["currentPreviewSceneName"] = name
        return {}

    if req_type == "GetCurrentPreviewScene":
        return {"currentPreviewSceneName": _state["currentPreviewSceneName"]}

    if req_type == "TriggerStudioModeTransition":
        if _state["studioModeEnabled"] and _state["currentPreviewSceneName"]:
            _state["currentProgramSceneName"] = _state["currentPreviewSceneName"]
        return {}

    if req_type == "SetStudioModeEnabled":
        _state["studioModeEnabled"] = d.get("studioModeEnabled", False)
        return {}

    if req_type == "GetStudioModeEnabled":
        return {"studioModeEnabled": _state["studioModeEnabled"]}

    if req_type == "GetSceneItemList":
        scene = d.get("sceneName", _state["currentProgramSceneName"])
        return {"sceneItems": _state["sceneItems"].get(scene, [])}

    if req_type == "SetSceneItemEnabled":
        scene = d.get("sceneName", _state["currentProgramSceneName"])
        item_id = d.get("sceneItemId")
        enabled = d.get("sceneItemEnabled", True)
        for item in _state["sceneItems"].get(scene, []):
            if item["sceneItemId"] == item_id:
                item["sceneItemEnabled"] = enabled
        return {}

    if req_type == "GetInputList":
        return {"inputs": [
            {"inputName": name, "inputKind": "audio",
             "inputMuted": info["muted"], "inputVolumeDb": info["volumeDb"],
             "inputVolumeMul": info["volumeMul"]}
            for name, info in _state["inputs"].items()
        ]}

    if req_type == "GetInputMute":
        name = d.get("inputName", "")
        info = _state["inputs"].get(name, {})
        return {"inputMuted": info.get("muted", False)}

    if req_type == "SetInputMute":
        name = d.get("inputName", "")
        if name in _state["inputs"]:
            _state["inputs"][name]["muted"] = d.get("inputMuted", False)
        return {}

    if req_type == "GetInputVolume":
        name = d.get("inputName", "")
        info = _state["inputs"].get(name, {})
        return {
            "inputVolumeDb": info.get("volumeDb", 0.0),
            "inputVolumeMul": info.get("volumeMul", 1.0),
        }

    if req_type == "SetInputVolume":
        name = d.get("inputName", "")
        if name in _state["inputs"]:
            if "inputVolumeDb" in d:
                _state["inputs"][name]["volumeDb"] = d["inputVolumeDb"]
            if "inputVolumeMul" in d:
                _state["inputs"][name]["volumeMul"] = d["inputVolumeMul"]
        return {}

    if req_type == "StartStream":
        _state["streaming"] = True
        return {}

    if req_type == "StopStream":
        _state["streaming"] = False
        return {}

    if req_type == "GetStreamStatus":
        return {"outputActive": _state["streaming"]}

    if req_type == "StartRecord":
        _state["recording"] = True
        return {}

    if req_type == "StopRecord":
        _state["recording"] = False
        return {}

    if req_type == "GetRecordStatus":
        return {"outputActive": _state["recording"]}

    if req_type == "SetRecordDirectory":
        _state["recordDirectory"] = d.get("recordDirectory", _state["recordDirectory"])
        return {}

    if req_type == "GetVideoSettings":
        return dict(_state["video"])

    if req_type == "SetVideoSettings":
        for k in ("baseWidth", "baseHeight", "outputWidth", "outputHeight",
                   "fpsNumerator", "fpsDenominator"):
            if k in d:
                _state["video"][k] = d[k]
        return {}

    if req_type == "GetSceneCollectionList":
        return {
            "currentSceneCollectionName": _state["currentSceneCollection"],
            "sceneCollections": _state["sceneCollections"],
        }

    if req_type == "SetCurrentSceneCollection":
        target = d.get("sceneCollectionName", _state["currentSceneCollection"])
        if target in _state["sceneCollections"]:
            _state["currentSceneCollection"] = target
        return {}

    if req_type == "CreateScene":
        name = d.get("sceneName", "")
        if name and not any(s["sceneName"] == name for s in _state["scenes"]):
            _state["scenes"].append({"sceneName": name, "sceneIndex": len(_state["scenes"])})
            _state["sceneItems"][name] = []
        return {}

    if req_type == "RemoveScene":
        name = d.get("sceneName", "")
        _state["scenes"] = [s for s in _state["scenes"] if s["sceneName"] != name]
        _state["sceneItems"].pop(name, None)
        return {}

    if req_type == "CreateInput":
        scene = d.get("sceneName", _state["currentProgramSceneName"])
        source_name = d.get("inputName", d.get("sourceName", ""))
        kind = d.get("inputKind", "source")
        items = _state["sceneItems"].setdefault(scene, [])
        item_id = max((i["sceneItemId"] for i in items), default=0) + 1
        items.append({
            "sourceName": source_name, "sceneItemId": item_id,
            "sceneItemEnabled": True, "sceneItemLocked": False, "inputKind": kind,
        })
        _state["inputs"][source_name] = {"muted": False, "volumeDb": 0.0, "volumeMul": 1.0}
        return {"sceneItemId": item_id}

    if req_type == "SetSceneItemTransform":
        return {}

    # Unknown — return empty
    return {}


async def handler(websocket):
    """Handle a single WebSocket connection."""
    # obs-websocket v5 handshake: send Hello (op 0)
    await websocket.send(json.dumps({
        "op": 0,
        "d": {
            "obsWebSocketVersion": "5.0.0",
            "rpcVersion": 1,
        }
    }))

    async for message in websocket:
        msg = json.loads(message)
        op = msg.get("op")
        data = msg.get("d", {})

        if op == 1:  # Identify
            await websocket.send(json.dumps({
                "op": 2,  # Identified
                "d": {"negotiatedRpcVersion": 1}
            }))

        elif op == 6:  # Request
            req_type = data.get("requestType", "")
            req_id = data.get("requestId", str(uuid.uuid4()))
            req_data = data.get("requestData") or {}
            try:
                resp_data = _handle_request(req_type, req_data)
                status = {"result": True, "code": 100}
            except Exception as e:
                resp_data = {}
                status = {"result": False, "code": 300, "comment": str(e)}
            await websocket.send(json.dumps({
                "op": 7,  # RequestResponse
                "d": {
                    "requestType": req_type,
                    "requestId": req_id,
                    "requestStatus": status,
                    "responseData": resp_data,
                }
            }))

        elif op == 8:  # RequestBatch
            responses = []
            for req in data.get("requests", []):
                req_type = req.get("requestType", "")
                req_id = req.get("requestId", str(uuid.uuid4()))
                req_data = req.get("requestData") or {}
                try:
                    resp_data = _handle_request(req_type, req_data)
                    status = {"result": True, "code": 100}
                except Exception as e:
                    resp_data = {}
                    status = {"result": False, "code": 300, "comment": str(e)}
                responses.append({
                    "requestType": req_type,
                    "requestId": req_id,
                    "requestStatus": status,
                    "responseData": resp_data,
                })
            await websocket.send(json.dumps({
                "op": 9,  # RequestBatchResponse
                "d": {"results": responses}
            }))


async def main():
    print("[obs-mock] Starting OBS WebSocket mock on ws://0.0.0.0:4455")
    async with websockets.serve(handler, "0.0.0.0", 4455):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
