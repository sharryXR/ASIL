"""Independent raw-state validation for ASIL evaluation audits.

This module deliberately avoids ASIL observations and per-application
observation builders. It reads final files or queries backing services directly,
then evaluates the frozen task checkpoint rules with a separate implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

import requests
from lxml import etree
from PIL import Image


SUPPORTED_RULES = {
    "any_element_contains",
    "any_element_matches",
    "app_view",
    "count_elements_matching",
    "element_contains",
    "element_exists",
    "element_not_exists",
    "element_value",
    "input_volume_db",
    "no_element_matches",
    "source_visible",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _element(
    element_id: str,
    element_type: str,
    value: Any,
    *,
    label: str = "",
) -> dict[str, Any]:
    return {
        "id": str(element_id),
        "type": str(element_type),
        "label": str(label),
        "value": value,
    }


def _state(
    *,
    software: str,
    kind: str,
    evidence: Any,
    elements: list[dict[str, Any]],
    app_view: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = evidence if isinstance(evidence, bytes) else _canonical_bytes(evidence)
    return {
        "software": software,
        "elements": elements,
        "app_view": app_view,
        "raw_evidence": {"kind": kind, **(details or {})},
        "raw_sha256": _sha256(payload),
    }


def _source_path(adapter: object) -> Path:
    source = getattr(adapter, "source_path", None)
    if callable(source):
        source = source()
    if source is None:
        raise ValueError("adapter does not expose an immutable raw source path")
    return Path(source)


def _read_audacity(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    payload = path.read_bytes()
    data = json.loads(payload)
    elements = [
        _element(
            f"track:{track['id']}",
            "track",
            dict(track),
            label=str(track.get("name", track["id"])),
        )
        for track in data.get("tracks", [])
    ]
    return _state(
        software="audacity",
        kind="json_file",
        evidence=payload,
        elements=elements,
        details={"path": str(path), "bytes": len(payload)},
    )


def _read_drawio(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    payload = path.read_bytes()
    data = json.loads(payload)
    shapes = {str(shape["id"]): shape for shape in data.get("shapes", [])}
    elements = [
        _element(f"shape:{shape_id}", "shape", dict(shape), label=str(shape.get("label", shape_id)))
        for shape_id, shape in shapes.items()
    ]
    for connector in data.get("connectors", []):
        value = dict(connector)
        value["source_label"] = shapes.get(str(connector.get("source")), {}).get(
            "label", connector.get("source", "")
        )
        value["target_label"] = shapes.get(str(connector.get("target")), {}).get(
            "label", connector.get("target", "")
        )
        elements.append(
            _element(
                f"connector:{connector['id']}",
                "connector",
                value,
                label=str(connector.get("label", connector["id"])),
            )
        )
    return _state(
        software="drawio",
        kind="json_file",
        evidence=payload,
        elements=elements,
        details={"path": str(path), "bytes": len(payload)},
    )


def _read_inkscape(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    payload = path.read_bytes()
    root = etree.fromstring(payload)
    elements: list[dict[str, Any]] = []
    for node in root.iter():
        element_id = node.get("id")
        if not element_id:
            continue
        element_type = etree.QName(node).localname
        elements.append(_element(element_id, element_type, dict(node.attrib), label=element_id))
    return _state(
        software="inkscape",
        kind="svg_xml",
        evidence=payload,
        elements=elements,
        details={"path": str(path), "bytes": len(payload)},
    )


def _read_kdenlive(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    payload = path.read_bytes()
    root = etree.fromstring(payload)
    elements = [
        _element(
            f"track:{track.get('id')}",
            "track",
            dict(track.attrib),
            label=str(track.get("name", track.get("id", ""))),
        )
        for track in root.xpath(".//timeline/track")
        if track.get("id")
    ]
    return _state(
        software="kdenlive",
        kind="kdenlive_xml",
        evidence=payload,
        elements=elements,
        details={"path": str(path), "bytes": len(payload)},
    )


def _cell_output_text(cell: dict[str, Any]) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []) or []:
        for key in ("text", "data"):
            value = output.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif isinstance(value, dict):
                text = value.get("text/plain", "")
                parts.extend(text if isinstance(text, list) else [str(text)])
            elif value is not None:
                parts.append(str(value))
    return "".join(parts)


def _read_workspace(adapter: object, software: str) -> dict[str, Any]:
    root = _source_path(adapter).resolve()
    elements: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            continue
        payload = path.read_bytes()
        evidence_rows.append({"path": relative, "sha256": _sha256(payload), "bytes": len(payload)})
        text = payload.decode("utf-8", errors="replace")
        elements.append(
            _element(
                f"file:{relative}",
                "file",
                {"path": relative, "content": text},
                label=path.name,
            )
        )
        if software != "jupyterlab":
            continue
        if relative.endswith(".ipynb"):
            notebook = json.loads(text)
            for index, cell in enumerate(notebook.get("cells", [])):
                source = cell.get("source", "")
                source_text = "".join(source) if isinstance(source, list) else str(source)
                elements.append(
                    _element(
                        f"cell:{relative}:{index}",
                        "cell",
                        {
                            "source": source_text,
                            "output": _cell_output_text(cell),
                            "cell_type": cell.get("cell_type", "code"),
                        },
                        label=f"Cell {index + 1}",
                    )
                )
        else:
            elements.append(
                _element(
                    f"editor:{relative}",
                    "editor",
                    {"path": relative, "content": text},
                    label=path.name,
                )
            )
    return _state(
        software=software,
        kind="workspace_files",
        evidence=evidence_rows,
        elements=elements,
        details={"root": str(root), "file_count": len(evidence_rows)},
    )


_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODF_NS = {"office": _OFFICE_NS, "table": _TABLE_NS, "text": _TEXT_NS}


def _column_letter(index: int) -> str:
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _read_libreoffice(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    payload = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        content = archive.read("content.xml")
    root = etree.fromstring(content)
    elements: list[dict[str, Any]] = []
    for sheet in root.xpath(".//table:table", namespaces=_ODF_NS):
        sheet_name = sheet.get(f"{{{_TABLE_NS}}}name", "Sheet")
        row_index = 0
        for row in sheet.xpath("table:table-row", namespaces=_ODF_NS):
            row_repeat = int(row.get(f"{{{_TABLE_NS}}}number-rows-repeated", "1"))
            column_index = 0
            for cell in row.xpath("table:table-cell", namespaces=_ODF_NS):
                column_repeat = int(cell.get(f"{{{_TABLE_NS}}}number-columns-repeated", "1"))
                value = cell.get(f"{{{_OFFICE_NS}}}value", "")
                text = "".join(cell.xpath(".//text:p//text()", namespaces=_ODF_NS))
                resolved = value or text
                if resolved:
                    for offset in range(column_repeat):
                        cell_id = f"{sheet_name}!{_column_letter(column_index + offset)}{row_index + 1}"
                        elements.append(_element(cell_id, "cell", resolved, label=cell_id))
                column_index += column_repeat
            row_index += row_repeat
    return _state(
        software="libreoffice",
        kind="odf_zip_xml",
        evidence=payload,
        elements=elements,
        details={"path": str(path), "bytes": len(payload)},
    )


def _read_gimp(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    payload = path.read_bytes()
    with Image.open(path) as image:
        candidates = list(image.text.values()) + [
            value for value in image.info.values() if isinstance(value, str)
        ]
    state_data: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("layers"), list):
            state_data = parsed
            break
    if state_data is None:
        raise ValueError("GIMP PNG does not contain raw layer-state metadata")
    elements = [
        _element(
            str(layer["id"]),
            "layer",
            dict(layer),
            label=str(layer.get("label", layer["id"])),
        )
        for layer in state_data.get("layers", [])
    ]
    return _state(
        software="gimp",
        kind="png_text_metadata",
        evidence=payload,
        elements=elements,
        details={"path": str(path), "bytes": len(payload)},
    )


def _read_nautilus(adapter: object) -> dict[str, Any]:
    root = Path(getattr(adapter, "workspace_path", _source_path(adapter))).resolve()
    state_path = Path(getattr(adapter, "state_path"))
    state_payload = state_path.read_bytes()
    state_data = json.loads(state_payload)
    current_relative = str(state_data.get("current_dir", ""))
    current_dir = root / current_relative if current_relative else root
    elements = [
        _element(
            "location",
            "location",
            {"path": current_relative or "/", "display_name": current_dir.name},
            label="Current Location",
        )
    ]
    if current_dir.is_dir():
        for path in sorted(current_dir.iterdir(), key=lambda item: item.name.casefold()):
            relative = path.relative_to(root).as_posix()
            elements.append(
                _element(
                    f"entry:{relative}",
                    "directory_entry",
                    {
                        "name": path.name,
                        "path": relative,
                        "is_dir": path.is_dir(),
                    },
                    label=path.name,
                )
            )
    evidence = {
        "state_sha256": _sha256(state_payload),
        "entries": [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
        ],
    }
    return _state(
        software="nautilus",
        kind="filesystem_plus_state_json",
        evidence=evidence,
        elements=elements,
        details={"root": str(root), "state_path": str(state_path)},
    )


def _read_blender(adapter: object) -> dict[str, Any]:
    path = _source_path(adapter)
    blender_bin = str(getattr(adapter, "blender_bin", "blender"))
    with tempfile.TemporaryDirectory(prefix="asil-raw-blender-") as directory:
        output = Path(directory) / "scene.json"
        script = Path(directory) / "inspect.py"
        script.write_text(
            "import bpy, json\n"
            f"out = r{str(output)!r}\n"
            "rows = []\n"
            "for obj in bpy.context.scene.objects:\n"
            "    rows.append({'id': obj.name, 'type': obj.type.lower(), 'label': obj.name, "
            "'value': {'dimensions': list(obj.dimensions), 'location': list(obj.location), 'scale': list(obj.scale)}})\n"
            "with open(out, 'w') as f: json.dump({'elements': rows}, f, sort_keys=True)\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [blender_bin, "--background", str(path), "--python", str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"independent Blender inspection failed with code {completed.returncode}: "
                f"{completed.stderr[-500:]}"
            )
        direct = json.loads(output.read_text(encoding="utf-8"))
    return _state(
        software="blender",
        kind="native_background_inspection",
        evidence=direct,
        elements=list(direct.get("elements", [])),
        details={"path": str(path), "executable": Path(blender_bin).name},
    )


def _direct_gitea_get(adapter: object, endpoint: str) -> Any:
    url = f"{str(getattr(adapter, 'base_url')).rstrip('/')}{endpoint}"
    token = str(getattr(adapter, "token", ""))
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    response = requests.get(
        url,
        headers=headers,
        timeout=10,
        proxies={"http": None, "https": None},
    )
    if response.status_code == 401 and hasattr(adapter, "_basic_auth"):
        response = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            auth=getattr(adapter, "_basic_auth"),
            timeout=10,
            proxies={"http": None, "https": None},
        )
    response.raise_for_status()
    return response.json()


def _read_gitea(adapter: object) -> dict[str, Any]:
    owner = str(getattr(adapter, "owner"))
    repo = str(getattr(adapter, "repo"))
    queries = {
        "repositories": "/api/v1/repos/search?limit=50",
        "issues": f"/api/v1/repos/{owner}/{repo}/issues?type=issues&limit=50&state=all",
        "comments": f"/api/v1/repos/{owner}/{repo}/issues/comments?limit=50",
    }
    responses = {name: _direct_gitea_get(adapter, endpoint) for name, endpoint in queries.items()}
    elements: list[dict[str, Any]] = []
    repositories = responses["repositories"]
    repositories = repositories.get("data", []) if isinstance(repositories, dict) else repositories
    for repository in repositories or []:
        elements.append(
            _element(
                f"repo:{repository['full_name']}",
                "repository",
                dict(repository),
                label=str(repository.get("full_name", "")),
            )
        )
    for issue in responses["issues"] or []:
        elements.append(
            _element(
                f"issue:{issue['number']}",
                "issue",
                dict(issue),
                label=str(issue.get("title", "")),
            )
        )
    for comment in responses["comments"] or []:
        elements.append(
            _element(
                f"comment:{comment['id']}",
                "comment",
                dict(comment),
                label=str(comment.get("body", ""))[:80],
            )
        )
    app_view = str(getattr(adapter, "_current_ui_path", "")).strip("/")
    return _state(
        software="gitea",
        kind="direct_rest_json",
        evidence=responses,
        elements=elements,
        app_view=app_view,
        details={"endpoints": list(queries.values()), "response_sets": len(responses)},
    )


def _read_obs(adapter: object) -> dict[str, Any]:
    ws = getattr(adapter, "ws")
    scene_list = ws.call("GetSceneList")
    studio_mode = ws.call("GetStudioModeEnabled")
    inputs = ws.call("GetInputList")
    responses: dict[str, Any] = {
        "scene_list": scene_list,
        "studio_mode": studio_mode,
        "inputs": inputs,
        "scene_items": {},
        "input_state": {},
    }
    elements: list[dict[str, Any]] = []
    for scene in scene_list.get("scenes", []):
        scene_name = scene["sceneName"]
        items = ws.call("GetSceneItemList", {"sceneName": scene_name})
        responses["scene_items"][scene_name] = items
        for item in items.get("sceneItems", []):
            elements.append(
                _element(
                    f"scene:{scene_name}/source:{item['sourceName']}",
                    "source",
                    {
                        "scene": scene_name,
                        "visible": item.get("sceneItemEnabled", True),
                    },
                    label=str(item["sourceName"]),
                )
            )
    for input_row in inputs.get("inputs", []):
        name = input_row["inputName"]
        volume = ws.call("GetInputVolume", {"inputName": name})
        mute = ws.call("GetInputMute", {"inputName": name})
        responses["input_state"][name] = {"volume": volume, "mute": mute}
        elements.append(
            _element(
                f"input:{name}",
                "input",
                {
                    "name": name,
                    "volume_db": volume.get("inputVolumeDb", 0.0),
                    "muted": mute.get("inputMuted", False),
                },
                label=name,
            )
        )
    elements.append(
        _element(
            "studio_mode",
            "status",
            {"enabled": bool(studio_mode.get("studioModeEnabled", False))},
            label="Studio Mode",
        )
    )
    return _state(
        software="obs",
        kind="direct_websocket_json",
        evidence=responses,
        elements=elements,
        app_view=str(scene_list.get("currentProgramSceneName", "")),
        details={"request_families": 5},
    )


_READERS: dict[str, Callable[[object], dict[str, Any]]] = {
    "audacity": _read_audacity,
    "blender": _read_blender,
    "code_server": lambda adapter: _read_workspace(adapter, "code_server"),
    "drawio": _read_drawio,
    "gimp": _read_gimp,
    "gitea": _read_gitea,
    "inkscape": _read_inkscape,
    "jupyterlab": lambda adapter: _read_workspace(adapter, "jupyterlab"),
    "kdenlive": _read_kdenlive,
    "libreoffice": _read_libreoffice,
    "nautilus": _read_nautilus,
    "obs": _read_obs,
}


def read_raw_state(adapter: object, task: object) -> dict[str, Any]:
    software = str(getattr(task, "software"))
    reader = _READERS.get(software)
    if reader is None:
        raise ValueError(f"no independent raw reader for software {software!r}")
    return reader(adapter)


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _equivalent(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _equivalent(left, right) for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    return actual == expected or str(actual) == str(expected)


def _matches(element: dict[str, Any], spec: dict[str, Any]) -> bool:
    if spec.get("id") is not None and element.get("id") != spec.get("id"):
        return False
    if spec.get("type") is not None and element.get("type") != spec.get("type"):
        return False
    if spec.get("label") is not None and element.get("label") != spec.get("label"):
        return False
    if "value" in spec and not _equivalent(element.get("value"), spec["value"]):
        return False
    return True


def _rule_passes(rule: dict[str, Any], raw_state: dict[str, Any]) -> bool:
    elements = list(raw_state.get("elements", []))
    rule_name, spec = next(iter(rule.items()))
    if rule_name == "element_exists":
        return any(element.get("id") == spec for element in elements)
    if rule_name == "element_not_exists":
        return all(element.get("id") != spec for element in elements)
    if rule_name == "element_value":
        element = next((row for row in elements if row.get("id") == spec.get("id")), None)
        if element is None:
            return False
        actual = element.get("value")
        key = spec.get("key")
        if key is not None:
            if not isinstance(actual, dict) or key not in actual:
                return False
            actual = actual[key]
        return _equivalent(actual, spec.get("expected"))
    if rule_name == "element_contains":
        element = next((row for row in elements if row.get("id") == spec.get("id")), None)
        if element is None:
            return False
        actual = element.get("value")
        key = spec.get("key")
        if key is not None:
            if not isinstance(actual, dict) or key not in actual:
                return False
            actual = actual[key]
        return str(spec.get("expected", "")) in str(actual)
    if rule_name == "any_element_matches":
        return any(_matches(element, spec) for element in elements)
    if rule_name == "no_element_matches":
        return not any(_matches(element, spec) for element in elements)
    if rule_name == "any_element_contains":
        for element in elements:
            if spec.get("type") is not None and element.get("type") != spec.get("type"):
                continue
            actual = element.get("value")
            key = spec.get("key")
            if key is not None:
                if not isinstance(actual, dict) or key not in actual:
                    continue
                actual = actual[key]
            if str(spec.get("expected", "")) in str(actual):
                return True
        return False
    if rule_name == "count_elements_matching":
        count = sum(
            1
            for element in elements
            if (spec.get("type") is None or element.get("type") == spec.get("type"))
            and (spec.get("value") is None or _equivalent(element.get("value"), spec.get("value")))
        )
        return count == int(spec.get("expected", 0))
    if rule_name == "app_view":
        actual = str(raw_state.get("app_view", ""))
        expected = str(spec)
        return actual == expected or actual.endswith(expected)
    if rule_name == "source_visible":
        return any(
            element.get("label") == spec.get("source")
            and isinstance(element.get("value"), dict)
            and element["value"].get("scene") == spec.get("scene")
            and _equivalent(element["value"].get("visible"), spec.get("expected"))
            for element in elements
        )
    if rule_name == "input_volume_db":
        return any(
            element.get("label") == spec.get("name")
            and isinstance(element.get("value"), dict)
            and _equivalent(element["value"].get("volume_db"), spec.get("expected"))
            for element in elements
        )
    raise ValueError(f"unsupported rule: {rule_name}")


def evaluate_task_raw(
    task: object,
    raw_state: dict[str, Any],
    *,
    evaluator_score: float,
) -> dict[str, Any]:
    errors: list[str] = []
    path_reports: list[dict[str, Any]] = []
    evaluator = getattr(task, "evaluator", {}) or {}
    paths = evaluator.get("paths") or []
    for path in paths:
        checkpoint_reports: list[dict[str, Any]] = []
        earned = 0.0
        total = 0.0
        for checkpoint in path.get("checkpoints") or []:
            weight = float(checkpoint.get("weight", 1.0))
            total += weight
            rule = checkpoint.get("rule", checkpoint)
            rule_names = list(rule) if isinstance(rule, dict) else []
            if len(rule_names) != 1 or rule_names[0] not in SUPPORTED_RULES:
                name = rule_names[0] if rule_names else "invalid"
                errors.append(f"unsupported rule: {name}")
                passed = False
            else:
                try:
                    passed = _rule_passes(rule, raw_state)
                except Exception as exc:
                    errors.append(f"raw rule evaluation failed for {checkpoint.get('id', 'checkpoint')}: {exc}")
                    passed = False
            if passed:
                earned += weight
            checkpoint_reports.append(
                {
                    "id": checkpoint.get("id", "checkpoint"),
                    "weight": weight,
                    "rule_type": rule_names[0] if rule_names else "invalid",
                    "passed": passed,
                }
            )
        score = earned / total if total else 0.0
        path_reports.append(
            {
                "path_id": path.get("path_id", "path"),
                "score": score,
                "checkpoints": checkpoint_reports,
            }
        )
    score = max((path["score"] for path in path_reports), default=0.0)
    complete = not errors and bool(path_reports)
    agreement = complete and math.isclose(score, float(evaluator_score), abs_tol=1e-9)
    return {
        "schema_version": "1.0",
        "task_id": str(getattr(task, "id", "")),
        "software": str(getattr(task, "software", "")),
        "complete": complete,
        "score": score,
        "evaluator_score": float(evaluator_score),
        "agreement": agreement,
        "errors": errors,
        "paths": path_reports,
        "raw_evidence": raw_state.get("raw_evidence", {}),
        "raw_sha256": raw_state.get("raw_sha256", ""),
    }


def validate_raw_final_state(
    adapter: object,
    task: object,
    *,
    evaluator_score: float,
) -> dict[str, Any]:
    try:
        raw_state = read_raw_state(adapter, task)
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "task_id": str(getattr(task, "id", "")),
            "software": str(getattr(task, "software", "")),
            "complete": False,
            "score": 0.0,
            "evaluator_score": float(evaluator_score),
            "agreement": False,
            "errors": [f"raw evidence collection failed: {exc}"],
            "paths": [],
            "raw_evidence": {},
            "raw_sha256": "",
        }
    return evaluate_task_raw(task, raw_state, evaluator_score=evaluator_score)
