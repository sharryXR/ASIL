"""Evaluation helpers for ASIL tasks.

Supports:
- Legacy flat validation rules
- Multi-condition validation lists
- Path-based evaluation with checkpoints and exclusive path selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageStat


@dataclass
class CheckpointReport:
    checkpoint_id: str
    passed: bool
    required: bool = True
    weight: float = 1.0
    earned: float = 0.0
    rule: dict[str, Any] = field(default_factory=dict)
    gui_visible_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "passed": self.passed,
            "required": self.required,
            "weight": self.weight,
            "earned": self.earned,
            "rule": self.rule,
            "gui_visible_required": self.gui_visible_required,
        }


@dataclass
class PathReport:
    path_id: str
    score: float
    complete: bool
    exclusive_group: str = "default"
    checkpoints: list[CheckpointReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "score": self.score,
            "complete": self.complete,
            "exclusive_group": self.exclusive_group,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }


@dataclass
class EvaluationReport:
    score: float
    success: bool
    matched_path_id: str | None
    path_reports: list[PathReport] = field(default_factory=list)
    mode: str = "legacy_validation"
    warnings: list[str] = field(default_factory=list)
    migration_mode: str = "native"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "success": self.success,
            "matched_path_id": self.matched_path_id,
            "mode": self.mode,
            "warnings": list(self.warnings),
            "migration_mode": self.migration_mode,
            "path_reports": [path.to_dict() for path in self.path_reports],
        }


def _parse_kv_string(text: str, key: str) -> str | None:
    for pair in text.split():
        if "=" in pair:
            k, _, v = pair.partition("=")
            if k == key:
                return v
    return None


def _find_element(obs, target_id: str):
    exact = next((e for e in obs.interactive_elements if e.id == target_id), None)
    if exact is not None:
        return exact
    prefix = target_id.rsplit(":", 1)[0] + ":"
    if prefix in ("label:", "milestone:"):
        return next((e for e in obs.interactive_elements if e.id.startswith(prefix)), None)
    return None


def _observation_for_app(obs, app: str):
    app_prefix = f"{app}::"
    local_elements = []
    app_state_update: dict[str, Any] = {}
    for element in obs.interactive_elements:
        metadata = element.metadata if isinstance(element.metadata, dict) else {}
        metadata_app = metadata.get("app")
        if metadata_app != app and not element.id.startswith(app_prefix):
            continue
        local_id = metadata.get("local_id")
        if not local_id:
            local_id = element.id.removeprefix(app_prefix)
        if local_id == "app_state" and element.type == "app_state" and isinstance(element.value, dict):
            app_state_update = {
                "current_view": str(element.value.get("current_view", "") or ""),
                "active_document": str(element.value.get("active_document", "") or ""),
                "document_path": str(element.value.get("document_path", "") or ""),
            }
        local_elements.append(element.model_copy(update={"id": str(local_id)}))
    app_state = obs.app_state
    if app_state_update:
        app_state = obs.app_state.model_copy(update=app_state_update)
    return obs.model_copy(update={"interactive_elements": local_elements, "app_state": app_state})


CELL_ID_RE = re.compile(r"^(?P<sheet>.+)!(?P<column>[A-Z]+)(?P<row>\d+)$")


def _parse_cell_id(cell_id: str) -> tuple[str, str, int] | None:
    match = CELL_ID_RE.match(cell_id)
    if not match:
        return None
    return match.group("sheet"), match.group("column"), int(match.group("row"))


def _sheet_cell_map(obs, sheet: str) -> dict[tuple[str, int], Any]:
    cells: dict[tuple[str, int], Any] = {}
    for elem in obs.interactive_elements:
        parsed = _parse_cell_id(elem.id)
        if parsed is None:
            continue
        elem_sheet, column, row = parsed
        if elem_sheet == sheet:
            cells[(column, row)] = elem
    return cells


def _spreadsheet_cell(obs, sheet: str, column: str, row: int):
    return _sheet_cell_map(obs, sheet).get((column, row))


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _document_image_path(obs) -> Path | None:
    path_text = getattr(obs.app_state, "document_path", "") or getattr(obs.app_state, "active_document", "")
    if not path_text:
        return None
    path = Path(str(path_text))
    return path if path.exists() else None


def _image_region_stat(obs, spec: dict[str, Any]) -> float | None:
    image_path = _document_image_path(obs)
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        box_spec = spec.get("box") or [0, 0, image.width, image.height]
        if isinstance(box_spec, dict):
            x = int(box_spec.get("x", 0))
            y = int(box_spec.get("y", 0))
            width = int(box_spec.get("width", image.width - x))
            height = int(box_spec.get("height", image.height - y))
        else:
            x, y, width, height = [int(value) for value in box_spec]
        region = image.convert("RGBA").crop((x, y, x + width, y + height))
        metric = str(spec.get("metric", "luminance"))
        if metric == "luminance":
            return float(ImageStat.Stat(region.convert("L")).mean[0])
        if metric == "alpha":
            return float(ImageStat.Stat(region.getchannel("A")).mean[0])
        channel_map = {"red": 0, "green": 1, "blue": 2}
        if metric in channel_map:
            return float(ImageStat.Stat(region).mean[channel_map[metric]])
        return None


def _numeric_range_match(actual: float | None, spec: dict[str, Any]) -> bool:
    if actual is None:
        return False
    if "expected" in spec:
        tolerance = float(spec.get("tolerance", 1e-3))
        return abs(actual - float(spec["expected"])) <= tolerance
    if "min" in spec and actual < float(spec["min"]):
        return False
    if "max" in spec and actual > float(spec["max"]):
        return False
    return "min" in spec or "max" in spec


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _matches_date(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


def _matches_time(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


def _cell_value(obs, sheet: str, column: str, row: int) -> Any:
    elem = _spreadsheet_cell(obs, sheet, column, row)
    return None if elem is None else elem.value


def _values_match(actual: Any, expected: Any, *, tol: float = 1e-3) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= tol

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_values_match(a, e, tol=tol) for a, e in zip(actual, expected))

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key, value in expected.items():
            if key not in actual or not _values_match(actual[key], value, tol=tol):
                return False
        return True

    return str(actual) == str(expected)


def _mapping_matches(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        if key not in actual or not _values_match(actual[key], value):
            return False
    return True


def _element_matches(element, spec: dict[str, Any]) -> bool:
    if "id" in spec and element.id != spec["id"]:
        return False
    if "id_prefix" in spec and not element.id.startswith(spec["id_prefix"]):
        return False
    if "type" in spec and element.type != spec["type"]:
        return False
    if "label" in spec and str(element.label) != str(spec["label"]):
        return False
    if "label_contains" in spec and str(spec["label_contains"]) not in str(element.label):
        return False
    if "value" in spec and not _mapping_matches(element.value, spec["value"]):
        return False
    if "metadata" in spec and not _mapping_matches(element.metadata, spec["metadata"]):
        return False
    return True


def _check_single(obs, rule: dict[str, Any]) -> bool:
    if "app_rule" in rule:
        spec = rule["app_rule"]
        if not isinstance(spec, dict):
            return False
        app = str(spec.get("app") or "")
        nested_rule = spec.get("rule")
        if not app or not isinstance(nested_rule, dict):
            return False
        return _check_single(_observation_for_app(obs, app), nested_rule)

    ids = {e.id for e in obs.interactive_elements}

    if "element_exists" in rule:
        return rule["element_exists"] in ids
    if "element_not_exists" in rule:
        return rule["element_not_exists"] not in ids

    if "element_value" in rule:
        spec = rule["element_value"]
        elem = _find_element(obs, spec["id"])
        if elem is None:
            return False
        key = spec.get("key")
        expected = spec["expected"]
        if isinstance(elem.value, dict):
            actual = elem.value.get(key) if key else elem.value
        elif key and isinstance(elem.value, str):
            actual = _parse_kv_string(elem.value, key)
        else:
            actual = elem.value
        return _values_match(actual, expected)

    if "element_contains" in rule:
        spec = rule["element_contains"]
        elem = _find_element(obs, spec["id"])
        if elem is None:
            return False
        key = spec.get("key")
        expected = spec["expected"]
        if isinstance(elem.value, dict):
            actual = elem.value.get(key, "") if key else str(elem.value)
        elif key and isinstance(elem.value, str):
            actual = _parse_kv_string(elem.value, key) or ""
        else:
            actual = str(elem.value)
        return expected in str(actual)

    if "element_metadata_value" in rule:
        spec = rule["element_metadata_value"]
        elem = _find_element(obs, spec["id"])
        if elem is None or not isinstance(elem.metadata, dict):
            return False
        actual = elem.metadata.get(spec.get("key"))
        return _values_match(actual, spec["expected"])

    if "image_size" in rule:
        spec = rule["image_size"]
        image_path = _document_image_path(obs)
        if image_path is None:
            return False
        with Image.open(image_path) as image:
            return int(spec["width"]) == image.width and int(spec["height"]) == image.height

    if "image_region_stat" in rule:
        spec = rule["image_region_stat"]
        return _numeric_range_match(_image_region_stat(obs, spec), spec)

    if "any_element_contains" in rule:
        spec = rule["any_element_contains"]
        elem_type = spec.get("type")
        key = spec.get("key")
        expected = spec["expected"]
        for e in obs.interactive_elements:
            if elem_type and e.type != elem_type:
                continue
            if isinstance(e.value, dict):
                actual = e.value.get(key, "") if key else str(e.value)
            else:
                actual = str(e.value)
            if expected in str(actual):
                return True
        return False

    if "any_element_value" in rule:
        spec = rule["any_element_value"]
        elem_type = spec.get("type")
        key = spec.get("key")
        expected = spec["expected"]
        for e in obs.interactive_elements:
            if elem_type and e.type != elem_type:
                continue
            if isinstance(e.value, dict):
                actual = e.value.get(key) if key else e.value
            else:
                actual = e.value
            if _values_match(actual, expected):
                return True
        return False

    if "any_element_matches" in rule:
        spec = rule["any_element_matches"]
        return any(_element_matches(elem, spec) for elem in obs.interactive_elements)

    if "no_element_matches" in rule:
        spec = rule["no_element_matches"]
        return not any(_element_matches(elem, spec) for elem in obs.interactive_elements)

    if "count_elements_matching" in rule:
        spec = rule["count_elements_matching"]
        expected = int(spec["expected"])
        matcher = {key: value for key, value in spec.items() if key != "expected"}
        return sum(1 for elem in obs.interactive_elements if _element_matches(elem, matcher)) == expected

    if "count_equals" in rule:
        return len(obs.interactive_elements) == rule["count_equals"]
    if "count_at_least" in rule:
        return len(obs.interactive_elements) >= rule["count_at_least"]

    if "object_exists" in rule:
        name = rule["object_exists"]
        return any(e.id == name for e in obs.interactive_elements)

    if "scene_object_count" in rule:
        expected = rule["scene_object_count"]
        count = sum(1 for e in obs.interactive_elements if e.type != "settings_group")
        return count == expected

    if "material_exists" in rule:
        name = rule["material_exists"]
        for e in obs.interactive_elements:
            if isinstance(e.metadata, dict):
                for mat in e.metadata.get("materials", []):
                    if mat.get("name") == name:
                        return True
        return False

    if "object_material_value" in rule:
        spec = rule["object_material_value"]
        elem = _find_element(obs, spec["id"])
        if elem is None or not isinstance(elem.metadata, dict):
            return False
        materials = elem.metadata.get("materials", [])
        return any(mat.get("name") == spec["expected"] for mat in materials)

    if "object_material_color" in rule:
        spec = rule["object_material_color"]
        elem = _find_element(obs, spec["id"])
        if elem is None or not isinstance(elem.metadata, dict):
            return False
        materials = elem.metadata.get("materials", [])
        expected = spec["expected"]
        return any(_values_match(mat.get("color"), expected, tol=float(spec.get("tolerance", 1e-3))) for mat in materials)

    if "object_has_modifier" in rule:
        spec = rule["object_has_modifier"]
        elem = _find_element(obs, spec["id"])
        if elem is None or not isinstance(elem.metadata, dict):
            return False
        modifiers = elem.metadata.get("modifiers", [])
        expected = spec["expected"]
        return any(
            modifier.get("type") == expected or modifier.get("name") == expected
            for modifier in modifiers
        )

    if "render_setting" in rule:
        spec = rule["render_setting"]
        rs = next((e for e in obs.interactive_elements if e.id == "render_settings"), None)
        if rs is None or not isinstance(rs.value, dict):
            return False
        return _values_match(rs.value.get(spec["key"]), spec["expected"])

    if "object_type_exists" in rule:
        obj_type = rule["object_type_exists"].lower()
        return any(e.type == obj_type for e in obs.interactive_elements)

    if "animation_data_exists" in rule:
        name = rule["animation_data_exists"]
        elem = next((e for e in obs.interactive_elements if e.id == name), None)
        if elem is None or not isinstance(elem.metadata, dict):
            return False
        return elem.metadata.get("has_animation_data", False)

    if "current_scene" in rule:
        return obs.app_state.current_view == rule["current_scene"]

    if "app_view" in rule:
        return obs.app_state.current_view == rule["app_view"]

    if "source_visible" in rule:
        spec = rule["source_visible"]
        elem_id = f"scene:{spec['scene']}/source:{spec['source']}"
        elem = next((e for e in obs.interactive_elements if e.id == elem_id), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("visible") == spec["expected"]
        return False

    if "response_has_field" in rule:
        return len(obs.interactive_elements) > 0

    if "input_muted" in rule:
        spec = rule["input_muted"]
        elem = next((e for e in obs.interactive_elements if e.id == f"input:{spec['name']}"), None)
        if elem is None or not isinstance(elem.value, dict):
            return False
        return elem.value.get("muted") == spec["expected"]

    if "input_volume_db" in rule:
        spec = rule["input_volume_db"]
        elem = next((e for e in obs.interactive_elements if e.id == f"input:{spec['name']}"), None)
        if elem is None or not isinstance(elem.value, dict):
            return False
        actual = elem.value.get("volumeDb", 0.0)
        return abs(float(actual) - float(spec["expected"])) < 0.1

    if "stream_active" in rule:
        elem = next((e for e in obs.interactive_elements if e.id == "stream_status"), None)
        if elem is None or not isinstance(elem.value, dict):
            return False
        return elem.value.get("active") == rule["stream_active"]

    if "record_active" in rule:
        elem = next((e for e in obs.interactive_elements if e.id == "record_status"), None)
        if elem is None or not isinstance(elem.value, dict):
            return False
        return elem.value.get("active") == rule["record_active"]

    if "scene_collection" in rule:
        elem = next((e for e in obs.interactive_elements if e.id == "scene_collection"), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("name") == rule["scene_collection"]
        return elem.label == rule["scene_collection"]

    if "spreadsheet_headers" in rule:
        spec = rule["spreadsheet_headers"]
        sheet = spec.get("sheet", "Sheet1")
        row = int(spec.get("row", 1))
        start_column = spec.get("start_column", "A")
        expected = spec["expected"]
        start_index = sum((ord(ch) - 64) * (26 ** idx) for idx, ch in enumerate(reversed(start_column)))
        for offset, want in enumerate(expected):
            col_num = start_index + offset
            col = ""
            while col_num > 0:
                col_num, rem = divmod(col_num - 1, 26)
                col = chr(65 + rem) + col
            if not _values_match(_cell_value(obs, sheet, col, row), want):
                return False
        return True

    if "spreadsheet_complete_rows" in rule:
        spec = rule["spreadsheet_complete_rows"]
        sheet = spec.get("sheet", "Sheet1")
        columns = spec["columns"]
        start_row = int(spec["start_row"])
        expected_rows = int(spec["expected_rows"])
        cells = _sheet_cell_map(obs, sheet)
        present_rows = sorted(
            {
                row
                for (column, row), elem in cells.items()
                if row >= start_row and column in columns and not _is_blank(elem.value)
            }
        )
        expected_row_numbers = list(range(start_row, start_row + expected_rows))
        if present_rows != expected_row_numbers:
            return False
        for row in expected_row_numbers:
            for column in columns:
                elem = cells.get((column, row))
                if elem is None or _is_blank(elem.value):
                    return False
        return True

    if "spreadsheet_column_type" in rule:
        spec = rule["spreadsheet_column_type"]
        sheet = spec.get("sheet", "Sheet1")
        columns = spec.get("columns") or [spec["column"]]
        start_row = int(spec["start_row"])
        end_row = int(spec["end_row"])
        expected_type = spec["expected"]
        for row in range(start_row, end_row + 1):
            for column in columns:
                value = _cell_value(obs, sheet, column, row)
                if _is_blank(value):
                    return False
                if expected_type == "number":
                    num = _coerce_float(value)
                    if num is None:
                        return False
                    if "min" in spec and num < float(spec["min"]):
                        return False
                    if "max" in spec and num > float(spec["max"]):
                        return False
                elif expected_type == "date":
                    if not _matches_date(value):
                        return False
                elif expected_type == "time":
                    if not _matches_time(value):
                        return False
                elif expected_type == "text":
                    if _is_blank(value):
                        return False
                else:
                    return False
        return True

    if "spreadsheet_column_sequence" in rule:
        spec = rule["spreadsheet_column_sequence"]
        sheet = spec.get("sheet", "Sheet1")
        column = spec["column"]
        start_row = int(spec["start_row"])
        for offset, want in enumerate(spec["expected"]):
            if not _values_match(_cell_value(obs, sheet, column, start_row + offset), want):
                return False
        return True

    if "spreadsheet_column_unique" in rule:
        spec = rule["spreadsheet_column_unique"]
        sheet = spec.get("sheet", "Sheet1")
        column = spec["column"]
        start_row = int(spec["start_row"])
        end_row = int(spec["end_row"])
        seen: set[str] = set()
        for row in range(start_row, end_row + 1):
            value = _cell_value(obs, sheet, column, row)
            if _is_blank(value):
                return False
            key = str(value)
            if key in seen:
                return False
            seen.add(key)
        return True

    if "spreadsheet_column_allowed_values" in rule:
        spec = rule["spreadsheet_column_allowed_values"]
        sheet = spec.get("sheet", "Sheet1")
        column = spec["column"]
        start_row = int(spec["start_row"])
        end_row = int(spec["end_row"])
        allowed = {str(value) for value in spec["allowed"]}
        for row in range(start_row, end_row + 1):
            value = _cell_value(obs, sheet, column, row)
            if _is_blank(value) or str(value) not in allowed:
                return False
        return True

    if "spreadsheet_date_order" in rule:
        spec = rule["spreadsheet_date_order"]
        sheet = spec.get("sheet", "Sheet1")
        start_row = int(spec["start_row"])
        end_row = int(spec["end_row"])
        earlier_column = spec["earlier_column"]
        later_column = spec["later_column"]
        for row in range(start_row, end_row + 1):
            earlier = _cell_value(obs, sheet, earlier_column, row)
            later = _cell_value(obs, sheet, later_column, row)
            if not (_matches_date(earlier) and _matches_date(later)):
                return False
            earlier_dt = datetime.strptime(str(earlier), "%Y-%m-%d")
            later_dt = datetime.strptime(str(later), "%Y-%m-%d")
            if later_dt < earlier_dt:
                return False
        return True

    if "spreadsheet_row_numeric_relation" in rule:
        spec = rule["spreadsheet_row_numeric_relation"]
        sheet = spec.get("sheet", "Sheet1")
        start_row = int(spec["start_row"])
        end_row = int(spec["end_row"])
        target_column = spec["target_column"]
        add_columns = spec.get("add_columns", [])
        subtract_columns = spec.get("subtract_columns", [])
        tolerance = float(spec.get("tolerance", 1e-3))
        for row in range(start_row, end_row + 1):
            target = _coerce_float(_cell_value(obs, sheet, target_column, row))
            if target is None:
                return False
            adds = [_coerce_float(_cell_value(obs, sheet, column, row)) for column in add_columns]
            subs = [_coerce_float(_cell_value(obs, sheet, column, row)) for column in subtract_columns]
            if any(value is None for value in adds + subs):
                return False
            expected = sum(adds) - sum(subs)
            if abs(target - expected) > tolerance:
                return False
        return True

    return False


def legacy_validate(obs, validation: dict[str, Any]) -> bool:
    if not validation:
        return True
    if "conditions" in validation:
        return all(_check_single(obs, rule) for rule in validation["conditions"])

    for key in (
        "element_exists",
        "app_rule",
        "element_not_exists",
        "element_value",
        "element_contains",
        "element_metadata_value",
        "image_size",
        "image_region_stat",
        "any_element_contains",
        "any_element_value",
        "any_element_matches",
        "no_element_matches",
        "count_elements_matching",
        "count_equals",
        "count_at_least",
        "object_exists",
        "scene_object_count",
        "material_exists",
        "object_material_value",
        "object_material_color",
        "object_has_modifier",
        "render_setting",
        "object_type_exists",
        "animation_data_exists",
        "current_scene",
        "app_view",
        "source_visible",
        "response_has_field",
        "input_muted",
        "input_volume_db",
        "stream_active",
        "record_active",
        "scene_collection",
        "spreadsheet_headers",
        "spreadsheet_complete_rows",
        "spreadsheet_column_type",
        "spreadsheet_column_sequence",
        "spreadsheet_column_unique",
        "spreadsheet_column_allowed_values",
        "spreadsheet_date_order",
        "spreadsheet_row_numeric_relation",
    ):
        if key in validation and not _check_single(obs, {key: validation[key]}):
            return False
    return True


def _normalize_checkpoints(path: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoints = path.get("checkpoints")
    if checkpoints:
        total_weight = sum(float(item.get("weight", 0.0)) for item in checkpoints)
        default_weight = 1.0 / len(checkpoints) if checkpoints and total_weight == 0 else 0.0
        normalized = []
        for index, checkpoint in enumerate(checkpoints):
            normalized.append(
                {
                    "checkpoint_id": checkpoint.get("id", f"checkpoint_{index + 1}"),
                    "rule": checkpoint["rule"],
                    "required": checkpoint.get("required", True),
                    "weight": float(checkpoint.get("weight", default_weight)),
                    "gui_visible_required": bool(checkpoint.get("gui_visible_required", False)),
                }
            )
        return normalized

    conditions = path.get("conditions", [])
    if not conditions:
        return []
    default_weight = 1.0 / len(conditions)
    return [
        {
            "checkpoint_id": f"condition_{index + 1}",
            "rule": condition,
            "required": True,
            "weight": default_weight,
            "gui_visible_required": False,
        }
        for index, condition in enumerate(conditions)
    ]


def _gui_visibility_mapping(gui_expectations: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(gui_expectations, dict):
        return {}
    checkpoint_visibility = gui_expectations.get("checkpoint_visibility")
    return checkpoint_visibility if isinstance(checkpoint_visibility, dict) else {}


def _rule_uses_hidden_gui_only_state(rule: dict[str, Any]) -> bool:
    def _hidden_key(target_id: str | None, key: str | None) -> bool:
        if key in {"source", "target"} and str(target_id or "").startswith("connector:"):
            return True
        if key in {"fps", "proxy_enabled"}:
            return True
        if target_id == "project_settings":
            return True
        if target_id == "canvas" and key in {"width", "height"}:
            return True
        return False

    if "app_rule" in rule:
        spec = rule.get("app_rule")
        nested_rule = spec.get("rule") if isinstance(spec, dict) else None
        return _rule_uses_hidden_gui_only_state(nested_rule) if isinstance(nested_rule, dict) else False

    if "element_value" in rule or "element_contains" in rule or "element_metadata_value" in rule:
        spec = rule.get("element_value") or rule.get("element_contains") or rule.get("element_metadata_value") or {}
        return _hidden_key(str(spec.get("id", "")), spec.get("key"))

    for matcher_key in ("any_element_matches", "no_element_matches", "count_elements_matching"):
        if matcher_key not in rule:
            continue
        spec = dict(rule[matcher_key])
        target_id = str(spec.get("id", ""))
        value_spec = spec.get("value")
        if isinstance(value_spec, dict):
            if any(_hidden_key(target_id, key) for key in value_spec):
                return True
        metadata_spec = spec.get("metadata")
        if isinstance(metadata_spec, dict):
            if any(_hidden_key(target_id, key) for key in metadata_spec):
                return True

    return False


def _checkpoint_can_count_as_gui_visible(
    checkpoint_id: str,
    rule: dict[str, Any],
    gui_visible_required: bool,
    gui_expectations: dict[str, Any] | None,
) -> bool:
    if not gui_visible_required:
        return True
    if _rule_uses_hidden_gui_only_state(rule):
        return False
    if gui_expectations is None:
        return True
    checkpoint_visibility = _gui_visibility_mapping(gui_expectations)
    return checkpoint_id in checkpoint_visibility


def _evaluate_path(obs, path: dict[str, Any], *, gui_expectations: dict[str, Any] | None = None) -> PathReport:
    checkpoints: list[CheckpointReport] = []
    for item in _normalize_checkpoints(path):
        passed = _check_single(obs, item["rule"]) and _checkpoint_can_count_as_gui_visible(
            item["checkpoint_id"],
            item["rule"],
            item.get("gui_visible_required", False),
            gui_expectations,
        )
        checkpoints.append(
            CheckpointReport(
                checkpoint_id=item["checkpoint_id"],
                passed=passed,
                required=item["required"],
                weight=item["weight"],
                earned=item["weight"] if passed else 0.0,
                rule=item["rule"],
                gui_visible_required=item.get("gui_visible_required", False),
            )
        )

    complete = all(cp.passed for cp in checkpoints if cp.required)
    total_weight = sum(cp.weight for cp in checkpoints)
    score = (sum(cp.earned for cp in checkpoints) / total_weight) if total_weight > 0 else 0.0
    return PathReport(
        path_id=path.get("path_id", path.get("id", "path")),
        score=score,
        complete=complete,
        exclusive_group=path.get("exclusive_group", "default"),
        checkpoints=checkpoints,
    )


def _select_best_report(path_reports: list[PathReport], selection: str) -> PathReport | None:
    if not path_reports:
        return None

    if selection == "first_complete":
        for report in path_reports:
            if report.complete:
                return report

    return max(path_reports, key=lambda report: (report.complete, report.score))


def _select_group_winners(path_reports: list[PathReport], selection: str) -> list[PathReport]:
    grouped: dict[str, list[PathReport]] = {}
    for report in path_reports:
        group = str(report.exclusive_group or "default")
        grouped.setdefault(group, []).append(report)

    winners: list[PathReport] = []
    for reports in grouped.values():
        winner = _select_best_report(reports, selection)
        if winner is not None:
            winners.append(winner)
    return winners


def _select_path(path_reports: list[PathReport], selection: str) -> PathReport | None:
    winners = _select_group_winners(path_reports, selection)
    return _select_best_report(winners, selection)


def evaluate_observation(
    obs,
    validation: dict[str, Any] | None = None,
    evaluator: dict[str, Any] | None = None,
    gui_expectations: dict[str, Any] | None = None,
) -> EvaluationReport:
    if evaluator and evaluator.get("paths"):
        path_reports = [_evaluate_path(obs, path, gui_expectations=gui_expectations) for path in evaluator["paths"]]
        matched = _select_path(path_reports, evaluator.get("selection", "best_score"))
        return EvaluationReport(
            score=matched.score if matched else 0.0,
            success=bool(matched and matched.complete),
            matched_path_id=matched.path_id if matched else None,
            path_reports=path_reports,
            mode="paths",
            migration_mode="native",
        )

    success = legacy_validate(obs, validation or {})
    return EvaluationReport(
        score=1.0 if success else 0.0,
        success=success,
        matched_path_id="legacy_validation",
        path_reports=[],
        mode="legacy_validation",
        migration_mode="legacy",
    )


def _rules_from_validation(validation: dict[str, Any]) -> list[dict[str, Any]]:
    if not validation:
        return []
    if "conditions" in validation:
        return list(validation["conditions"])
    ordered = []
    for key in (
        "element_exists",
        "element_not_exists",
        "element_value",
        "element_contains",
        "element_metadata_value",
        "any_element_contains",
        "any_element_value",
        "any_element_matches",
        "no_element_matches",
        "count_elements_matching",
        "count_equals",
        "count_at_least",
        "object_exists",
        "scene_object_count",
        "material_exists",
        "object_material_value",
        "object_material_color",
        "object_has_modifier",
        "render_setting",
        "object_type_exists",
        "animation_data_exists",
        "current_scene",
        "app_view",
        "source_visible",
        "response_has_field",
        "input_muted",
        "input_volume_db",
        "stream_active",
        "record_active",
        "scene_collection",
        "spreadsheet_headers",
        "spreadsheet_complete_rows",
        "spreadsheet_column_type",
        "spreadsheet_column_sequence",
        "spreadsheet_column_unique",
        "spreadsheet_column_allowed_values",
        "spreadsheet_date_order",
        "spreadsheet_row_numeric_relation",
    ):
        if key in validation:
            ordered.append({key: validation[key]})
    return ordered


def _special_task_evaluator(task) -> dict[str, Any] | None:
    if task.id == "blender_06":
        return {
            "paths": [
                {
                    "path_id": "cube_red_material_applied",
                    "checkpoints": [
                        {"id": "cube_exists", "weight": 0.2, "rule": {"object_exists": "Cube"}},
                        {"id": "material_exists", "weight": 0.3, "rule": {"material_exists": "RedMaterial"}},
                        {"id": "cube_uses_material", "weight": 0.5, "rule": {"object_material_value": {"id": "Cube", "expected": "RedMaterial"}}},
                    ],
                }
            ],
            "selection": "best_score",
        }

    if task.id == "obs_20":
        return {
            "paths": [
                {
                    "path_id": "live_setup_brb",
                    "checkpoints": [
                        {"id": "program_scene", "weight": 0.3, "rule": {"current_scene": "BRB"}},
                        {"id": "stream", "weight": 0.35, "rule": {"stream_active": True}},
                        {"id": "record", "weight": 0.35, "rule": {"record_active": True}},
                    ],
                }
            ],
            "selection": "best_score",
        }

    return None


def _synthesized_task_evaluator(task) -> dict[str, Any] | None:
    special = _special_task_evaluator(task)
    if special is not None:
        return special

    if task.software not in {"blender", "obs"}:
        return None

    rules = _rules_from_validation(getattr(task, "validation", {}) or {})
    if not rules:
        return None

    weight = 1.0 / len(rules)
    checkpoints = [
        {
            "id": f"checkpoint_{index + 1}",
            "weight": weight,
            "rule": rule,
        }
        for index, rule in enumerate(rules)
    ]
    return {
        "paths": [
            {
                "path_id": f"{task.software}_primary",
                "checkpoints": checkpoints,
            }
        ],
        "selection": "best_score",
    }


def evaluate_task_result(task, obs) -> EvaluationReport:
    evaluator = getattr(task, "evaluator", {}) or {}
    validation = getattr(task, "validation", {}) or {}
    gui_expectations = getattr(task, "gui_expectations", {}) or {}
    migration_mode = "native"
    if not evaluator.get("paths"):
        synthesized = _synthesized_task_evaluator(task)
        if synthesized is not None:
            evaluator = synthesized
            migration_mode = "synthesized"
    report = evaluate_observation(obs, validation=validation, evaluator=evaluator, gui_expectations=gui_expectations)
    report.migration_mode = migration_mode if report.mode == "paths" else "legacy"
    return report
