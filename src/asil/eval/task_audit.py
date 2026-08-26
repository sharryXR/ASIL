"""Static audit helpers for migrated ASIL task definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asil.eval.evaluator import _rule_uses_hidden_gui_only_state


FULL15_SOFTWARE = {
    "inkscape",
    "libreoffice",
    "blender",
    "obs",
    "gitea",
    "gimp",
    "libreoffice_writer",
    "libreoffice_impress",
    "code_server",
    "thunderbird",
    "nautilus",
    "kdenlive",
    "audacity",
    "drawio",
    "jupyterlab",
}


@dataclass
class TaskAuditReport:
    path: Path | None
    task_id: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "task_id": self.task_id,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _rule_signature(rule: dict[str, Any]) -> str:
    if "app_rule" in rule and isinstance(rule.get("app_rule"), dict):
        spec = rule["app_rule"]
        nested = spec.get("rule")
        return json.dumps(
            {"app_rule": {"app": spec.get("app"), "rule": nested}},
            sort_keys=True,
            ensure_ascii=False,
        )
    return json.dumps(rule, sort_keys=True, ensure_ascii=False)


def _task_id(task: dict[str, Any]) -> str:
    return str(task.get("id") or task.get("task_id") or "(unknown)")


def _task_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    direct_actions = task.get("actions")
    if isinstance(direct_actions, list):
        return [action for action in direct_actions if isinstance(action, dict)]

    asil = task.get("_asil")
    if isinstance(asil, dict) and isinstance(asil.get("actions"), list):
        return [action for action in asil["actions"] if isinstance(action, dict)]

    return []


def _task_operations(task: dict[str, Any]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for action in _task_actions(task):
        params = action.get("params")
        if not isinstance(params, dict):
            continue
        ops = params.get("operations")
        if isinstance(ops, list):
            operations.extend(operation for operation in ops if isinstance(operation, dict))
    return operations


def _path_groups(paths: list[dict[str, Any]]) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    grouped: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    for path_spec in paths:
        path_id = str(path_spec.get("path_id") or "")
        group = str(path_spec.get("exclusive_group") or "default")
        checkpoints = path_spec.get("checkpoints")
        if isinstance(checkpoints, list):
            grouped.setdefault(group, []).append((path_id, checkpoints))
    return grouped


def _global_aggregate_reason(rule: dict[str, Any]) -> str | None:
    if "app_rule" in rule and isinstance(rule.get("app_rule"), dict):
        nested = rule["app_rule"].get("rule")
        if isinstance(nested, dict):
            return _global_aggregate_reason(nested)
    if "scene_object_count" in rule:
        return "scene_object_count"
    if "current_scene" in rule:
        return "current_scene"
    if "stream_active" in rule:
        return "stream_active"
    if "record_active" in rule:
        return "record_active"
    if "render_setting" in rule:
        return "render_setting"
    if "app_view" in rule:
        return "app_view"

    spec = rule.get("element_value") or rule.get("element_contains") or rule.get("element_metadata_value")
    if isinstance(spec, dict):
        target_id = str(spec.get("id") or "")
        if target_id in {"timeline_settings", "video_settings", "studio_mode", "preview_scene", "scene_collection"}:
            return target_id

    count_spec = rule.get("count_elements_matching")
    if isinstance(count_spec, dict) and "value" not in count_spec and "metadata" not in count_spec and "id" not in count_spec:
        return "count_elements_matching"

    return None


def _directory_entry_path_rule(rule: dict[str, Any]) -> tuple[str, int | None] | None:
    if "app_rule" in rule and isinstance(rule.get("app_rule"), dict):
        nested = rule["app_rule"].get("rule")
        if isinstance(nested, dict):
            return _directory_entry_path_rule(nested)
    count_spec = rule.get("count_elements_matching")
    if isinstance(count_spec, dict) and count_spec.get("type") == "directory_entry":
        value = count_spec.get("value")
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            return value["path"], count_spec.get("expected")

    no_match_spec = rule.get("no_element_matches")
    if isinstance(no_match_spec, dict) and no_match_spec.get("type") == "directory_entry":
        value = no_match_spec.get("value")
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            return value["path"], 0

    return None


def _infer_existing_paragraph_count(task: dict[str, Any]) -> int:
    software = str(task.get("software") or task.get("_asil", {}).get("software") or "")
    snapshot = str(task.get("snapshot") or "")
    if software == "libreoffice_writer" and snapshot in {"writer_default", ""}:
        return 2
    return 0


def _semantic_conflict_errors(task: dict[str, Any], checkpoints: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    operations = _task_operations(task)
    if not operations:
        return errors

    if any(op.get("action") == "copy_entry" for op in operations) and not any(op.get("action") == "move_entry" for op in operations):
        disappearing_paths = []
        for checkpoint in checkpoints:
            rule = checkpoint.get("rule")
            if not isinstance(rule, dict):
                continue
            path_and_expected = _directory_entry_path_rule(rule)
            if path_and_expected and path_and_expected[1] == 0:
                disappearing_paths.append(path_and_expected[0])
        if disappearing_paths:
            errors.append(
                "Task uses copy semantics but evaluator also requires source entries to disappear: "
                + ", ".join(sorted(disappearing_paths))
                + "."
            )

    paragraph_updates = [op for op in operations if op.get("action") == "set_paragraph_text" and isinstance(op.get("text"), str)]
    if paragraph_updates:
        added_paragraph_texts = [str(op.get("text")) for op in operations if op.get("action") == "add_paragraph" and isinstance(op.get("text"), str)]
        existing_paragraphs = _infer_existing_paragraph_count(task)
        for update in paragraph_updates:
            index = int(update.get("index", 0))
            if index <= existing_paragraphs:
                continue
            for checkpoint in checkpoints:
                rule = checkpoint.get("rule")
                if not isinstance(rule, dict):
                    continue
                spec = rule.get("any_element_matches")
                if not isinstance(spec, dict) or spec.get("type") != "paragraph":
                    continue
                value = spec.get("value")
                if isinstance(value, dict) and value.get("text_content") in added_paragraph_texts:
                    errors.append(
                        "Task updates a newly added paragraph but evaluator also requires the pre-update paragraph text to remain visible."
                    )
                    break
            if errors:
                break

    return errors


def audit_task_definition(task: dict[str, Any], *, path: Path | None = None) -> TaskAuditReport:
    task_id = _task_id(task)
    errors: list[str] = []
    warnings: list[str] = []
    asil = task.get("_asil") if isinstance(task.get("_asil"), dict) else {}
    software = str(task.get("software") or asil.get("software") or "")
    related_apps = task.get("related_apps")
    related_app_set: set[str] = set()

    if software == "multi_apps":
        if not isinstance(related_apps, list):
            errors.append("multi_apps tasks require `related_apps` as a list.")
            related_apps = []
        related_app_names = [str(app) for app in related_apps]
        related_app_set = set(related_app_names)
        if len(related_app_names) not in {2, 3}:
            errors.append("multi_apps `related_apps` must contain exactly 2 or 3 software names.")
        if len(related_app_set) != len(related_app_names):
            errors.append("multi_apps `related_apps` must not contain duplicates.")
        unknown_apps = sorted(related_app_set - FULL15_SOFTWARE)
        if unknown_apps:
            errors.append("multi_apps `related_apps` contains non-full15 software: " + ", ".join(unknown_apps))
        app_initial_states = asil.get("app_initial_states")
        if not isinstance(app_initial_states, dict):
            errors.append("multi_apps tasks require `_asil.app_initial_states`.")
            app_initial_states = {}
        missing_states = sorted(related_app_set - set(str(app) for app in app_initial_states))
        if missing_states:
            errors.append("multi_apps `_asil.app_initial_states` is missing: " + ", ".join(missing_states))
        extra_states = sorted(set(str(app) for app in app_initial_states) - related_app_set)
        if extra_states:
            errors.append("multi_apps `_asil.app_initial_states` includes apps outside related_apps: " + ", ".join(extra_states))
        primary_app = str(asil.get("primary_app") or "")
        if not primary_app:
            errors.append("multi_apps tasks require `_asil.primary_app`.")
        elif primary_app not in related_app_set:
            errors.append("multi_apps `_asil.primary_app` must be one of related_apps.")

    gui_expectations = task.get("gui_expectations")
    if not isinstance(gui_expectations, dict):
        errors.append("Missing required `gui_expectations` mapping.")
        gui_expectations = {}

    success_surface = gui_expectations.get("success_surface")
    visible_change_summary = gui_expectations.get("visible_change_summary")
    checkpoint_visibility = gui_expectations.get("checkpoint_visibility")

    if not success_surface:
        errors.append("`gui_expectations.success_surface` is required.")
    if not visible_change_summary:
        errors.append("`gui_expectations.visible_change_summary` is required.")
    if checkpoint_visibility is None:
        checkpoint_visibility = {}
    elif not isinstance(checkpoint_visibility, dict):
        errors.append("`gui_expectations.checkpoint_visibility` must be a mapping.")
        checkpoint_visibility = {}

    evaluator = task.get("evaluator")
    if not isinstance(evaluator, dict):
        errors.append("Missing required `evaluator` mapping.")
        evaluator = {}

    paths = evaluator.get("paths")
    if not isinstance(paths, list) or not paths:
        errors.append("`evaluator.paths` must be a non-empty list for migrated tasks.")
        return TaskAuditReport(path=path, task_id=task_id, ok=False, errors=errors, warnings=warnings)

    seen_path_ids: set[str] = set()
    exclusive_group_rules: dict[str, dict[str, str]] = {}

    for path_spec in paths:
        if not isinstance(path_spec, dict):
            errors.append("Each evaluator path must be a mapping.")
            continue

        path_id = str(path_spec.get("path_id") or "")
        if not path_id:
            errors.append("Every path requires a non-empty `path_id`.")
            continue
        if path_id in seen_path_ids:
            errors.append(f"Duplicate path id `{path_id}`.")
        seen_path_ids.add(path_id)

        checkpoints = path_spec.get("checkpoints")
        conditions = path_spec.get("conditions")
        if checkpoints is None and isinstance(conditions, list):
            warnings.append(f"Path `{path_id}` still uses legacy `conditions`; migrate to `checkpoints`.")
            checkpoints = [
                {"id": f"{path_id}_condition_{index+1}", "weight": 1.0 / max(len(conditions), 1), "rule": rule}
                for index, rule in enumerate(conditions)
            ]

        if not isinstance(checkpoints, list) or not checkpoints:
            errors.append(f"Path `{path_id}` must define a non-empty `checkpoints` list.")
            continue

        seen_checkpoint_ids: set[str] = set()
        rule_signatures: set[str] = set()
        exclusive_group = str(path_spec.get("exclusive_group") or "default")
        exclusive_group_rules.setdefault(exclusive_group, {})

        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                errors.append(f"Checkpoint in path `{path_id}` must be a mapping.")
                continue

            checkpoint_id = str(checkpoint.get("id") or "")
            if not checkpoint_id:
                errors.append(f"Path `{path_id}` contains a checkpoint without `id`.")
                continue
            if checkpoint_id in seen_checkpoint_ids:
                errors.append(f"Duplicate checkpoint id `{checkpoint_id}` in path `{path_id}`.")
            seen_checkpoint_ids.add(checkpoint_id)

            if "weight" not in checkpoint:
                errors.append(f"Checkpoint `{checkpoint_id}` in path `{path_id}` is missing `weight`.")
            rule = checkpoint.get("rule")
            if not isinstance(rule, dict) or not rule:
                errors.append(f"Checkpoint `{checkpoint_id}` in path `{path_id}` must define a non-empty `rule`.")
                continue

            signature = _rule_signature(rule)
            if signature in rule_signatures:
                errors.append(f"Path `{path_id}` has duplicate checkpoint rules; checkpoints should be distinct.")
            rule_signatures.add(signature)

            if checkpoint.get("gui_visible_required") and checkpoint_id not in checkpoint_visibility:
                errors.append(
                    f"Checkpoint `{checkpoint_id}` requires GUI visibility but has no "
                    "`gui_expectations.checkpoint_visibility` entry."
                )
            if (
                software == "multi_apps"
                and checkpoint.get("gui_visible_required")
                and checkpoint_id in checkpoint_visibility
            ):
                visibility = str(checkpoint_visibility.get(checkpoint_id) or "")
                if not visibility.startswith("visible_in_"):
                    errors.append(
                        f"Checkpoint `{checkpoint_id}` visibility should start with `visible_in_` for multi_apps."
                    )
                else:
                    visible_app = visibility.removeprefix("visible_in_").split(":", 1)[0]
                    if visible_app not in related_app_set and visible_app != "multi_window":
                        errors.append(
                            f"Checkpoint `{checkpoint_id}` visibility references `{visible_app}`, "
                            "which is not in related_apps."
                        )
            if checkpoint.get("gui_visible_required") and _rule_uses_hidden_gui_only_state(rule):
                errors.append(
                    f"Checkpoint `{checkpoint_id}` requires GUI visibility but evaluates hidden-only state."
                )

            previous_path = exclusive_group_rules[exclusive_group].get(signature)
            if previous_path and previous_path != path_id:
                errors.append(
                    f"Paths `{previous_path}` and `{path_id}` in exclusive_group `{exclusive_group}` "
                    "share the same checkpoint rule and can cross-score."
                )
            else:
                exclusive_group_rules[exclusive_group][signature] = path_id

        errors.extend(_semantic_conflict_errors(task, checkpoints))

    for exclusive_group, grouped_paths in _path_groups(paths).items():
        if len(grouped_paths) < 2:
            continue
        for path_id, checkpoints in grouped_paths:
            for checkpoint in checkpoints:
                rule = checkpoint.get("rule")
                if not isinstance(rule, dict):
                    continue
                reason = _global_aggregate_reason(rule)
                if reason:
                    checkpoint_id = str(checkpoint.get("id") or "(unknown)")
                    errors.append(
                        f"Path `{path_id}` in exclusive_group `{exclusive_group}` uses global aggregate checkpoint "
                        f"`{checkpoint_id}` via `{reason}`, which can cross-score across mutually exclusive paths."
                    )

    return TaskAuditReport(path=path, task_id=task_id, ok=not errors, errors=errors, warnings=warnings)


def audit_task_file(path: str | Path) -> TaskAuditReport:
    task_path = Path(path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    return audit_task_definition(task, path=task_path)


def audit_task_tree(root: str | Path) -> list[TaskAuditReport]:
    root_path = Path(root)
    if root_path.is_file():
        return [audit_task_file(root_path)]
    direct_files = sorted(root_path.glob("*.json"))
    if direct_files:
        return [audit_task_file(path) for path in direct_files]
    return [audit_task_file(path) for path in sorted(root_path.glob("*/*.json"))]
