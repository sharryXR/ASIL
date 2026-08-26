"""Composite ASIL adapter for cross-application tasks."""

from __future__ import annotations

import shutil
import html
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, AppState, Element, Environment, Meta, Navigation, Observation
from asil.rendering import RenderArtifact, capture_html_to_png, html_page


FULL15_SOFTWARE: tuple[str, ...] = (
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
)
BROWSER_SOFTWARE: frozenset[str] = frozenset({"code_server", "drawio", "gitea", "jupyterlab"})
WORKSPACE_FILE_APPS: frozenset[str] = frozenset({"code_server", "jupyterlab"})
TOKEN_RE = re.compile(r"MA-(\d{3})")

AdapterFactory = Callable[[str, Path, Any, bool], ASILAdapter]


def _default_adapter_factory(software: str, tmp: Path, sandbox=None, mock: bool = False) -> ASILAdapter:
    from asil.benchmark import _create_adapter

    return _create_adapter(software, tmp, sandbox=sandbox, mock=mock)


def _iter_workspace_file_targets(value: Any, current_app: str = ""):
    if isinstance(value, dict):
        app_rule = value.get("app_rule")
        if isinstance(app_rule, dict):
            app = str(app_rule.get("app", "") or current_app)
            yield from _iter_workspace_file_targets(app_rule.get("rule", {}), app)
            return

        for key in ("element_contains", "element_exists"):
            rule = value.get(key)
            if isinstance(rule, dict):
                element_id = str(rule.get("id", ""))
                if current_app and element_id.startswith("file:"):
                    yield current_app, element_id.removeprefix("file:")

        for child in value.values():
            yield from _iter_workspace_file_targets(child, current_app)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_workspace_file_targets(child, current_app)


class MultiAppAdapter(ASILAdapter):
    """Run 2-3 existing software adapters as one composite task surface."""

    app_name = "Multi Apps"
    software_name = "multi_apps"
    supported_action_types = ["batch"]

    def __init__(
        self,
        tmp: str | Path,
        sandbox=None,
        mock: bool = False,
        adapter_factory: AdapterFactory | None = None,
    ) -> None:
        self.tmp = Path(tmp)
        self.sandbox = sandbox
        self.mock = mock
        self.adapter_factory = adapter_factory or _default_adapter_factory
        self.active_adapters: dict[str, ASILAdapter] = {}
        self.related_apps: list[str] = []
        self.app_initial_states: dict[str, str] = {}
        self.primary_app = ""
        self._prepared_task_id = ""
        self.tmp.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: Path,
        sandbox=None,
        mock: bool = False,
    ) -> "MultiAppAdapter":
        return cls(tmp=tmp, sandbox=sandbox, mock=mock)

    def _child_tmp(self, task_id: str, app: str) -> Path:
        return self.tmp / task_id / app

    def _validate_related_apps(self, related_apps: list[str]) -> None:
        unknown = sorted(set(related_apps) - set(FULL15_SOFTWARE))
        if unknown:
            raise ValueError(f"multi_apps task references unsupported software: {', '.join(unknown)}")
        if len(related_apps) not in {2, 3}:
            raise ValueError("multi_apps tasks must reference exactly 2 or 3 related_apps.")
        if len(set(related_apps)) != len(related_apps):
            raise ValueError("multi_apps related_apps must not contain duplicates.")

    def prepare_task(self, task: Any) -> None:
        related_apps = list(getattr(task, "related_apps", None) or [])
        self._validate_related_apps(related_apps)
        primary_app = str(getattr(task, "primary_app", "") or related_apps[0])
        if primary_app not in related_apps:
            raise ValueError(f"multi_apps primary_app {primary_app!r} is not in related_apps.")

        task_id = str(getattr(task, "id", "task") or "task")
        task_root = self.tmp / task_id
        if task_root.exists():
            shutil.rmtree(task_root)
        task_root.mkdir(parents=True, exist_ok=True)

        self.active_adapters = {}
        self.related_apps = related_apps
        self.primary_app = primary_app
        raw_states = getattr(task, "app_initial_states", {}) or {}
        self.app_initial_states = {
            app: str(raw_states.get(app, "default") or "default")
            for app in related_apps
        }

        for app in related_apps:
            child_tmp = self._child_tmp(task_id, app)
            child_tmp.mkdir(parents=True, exist_ok=True)
            adapter = self.adapter_factory(app, child_tmp, self.sandbox, self.mock)
            reset = getattr(adapter, "reset_state", None)
            if callable(reset):
                reset()
            setup = getattr(adapter, "setup_state", None)
            if callable(setup):
                setup(self.app_initial_states.get(app, "default"))
            clear_shadow = getattr(adapter, "clear_gui_shadow_state", None)
            if callable(clear_shadow):
                clear_shadow()
            self.active_adapters[app] = adapter

        self._prime_workspace_targets_for_gui(task)

        self._prepared_task_id = task_id

    def _prime_workspace_targets_for_gui(self, task: Any) -> None:
        evaluator = getattr(task, "evaluator", {}) or {}
        targets_by_app: dict[str, list[str]] = {}
        for app, relative_path in _iter_workspace_file_targets(evaluator):
            if app in WORKSPACE_FILE_APPS and app in self.active_adapters:
                normalized = str(relative_path).strip("/").replace("\\", "/")
                if normalized:
                    targets_by_app.setdefault(app, []).append(normalized)

        for app, targets in targets_by_app.items():
            adapter = self.active_adapters[app]
            workspace_path = getattr(adapter, "workspace_path", None)
            if workspace_path is None:
                continue
            unique_targets = list(dict.fromkeys(targets))
            for relative_path in unique_targets:
                path = Path(workspace_path) / relative_path
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")
            set_active = getattr(adapter, "_set_active_file", None)
            if callable(set_active) and unique_targets:
                try:
                    set_active(unique_targets[0])
                except Exception:
                    pass

    def reset_state(self) -> None:
        for adapter in self.active_adapters.values():
            reset = getattr(adapter, "reset_state", None)
            if callable(reset):
                reset()
            clear_shadow = getattr(adapter, "clear_gui_shadow_state", None)
            if callable(clear_shadow):
                clear_shadow()

    def setup_state(self, initial_state: str) -> None:
        for adapter in self.active_adapters.values():
            setup = getattr(adapter, "setup_state", None)
            if callable(setup):
                setup(initial_state)

    def get_context(self) -> dict[str, str]:
        context: dict[str, str] = {}
        for app, adapter in self.active_adapters.items():
            child_context = getattr(adapter, "get_context", lambda: {})()
            for key, value in child_context.items():
                context[f"{app}.{key}"] = str(value)
        return context

    def _namespaced_element(self, app: str, element: Element) -> Element:
        metadata = dict(element.metadata or {})
        metadata.setdefault("app", app)
        metadata.setdefault("local_id", element.id)
        return element.model_copy(
            update={
                "id": f"{app}::{element.id}",
                "children": [
                    child if "::" in str(child) else f"{app}::{child}"
                    for child in element.children
                ],
                "metadata": metadata,
            }
        )

    def observe(self) -> Observation:
        elements: list[Element] = []
        summaries: list[str] = []
        active_document_parts: list[str] = []
        for app in self.related_apps:
            adapter = self.active_adapters.get(app)
            if adapter is None:
                continue
            child_obs = adapter.observe()
            elements.append(
                Element(
                    id=f"app::{app}",
                    type="app_state",
                    label=getattr(adapter, "app_name", app),
                    value={
                        "current_view": child_obs.app_state.current_view,
                        "active_document": child_obs.app_state.active_document,
                        "document_path": child_obs.app_state.document_path,
                        "element_count": len(child_obs.interactive_elements),
                    },
                    editable=False,
                    metadata={"app": app, "local_id": "app_state"},
                )
            )
            elements.extend(self._namespaced_element(app, element) for element in child_obs.interactive_elements)
            if child_obs.data_summary:
                summaries.append(f"{app}: {child_obs.data_summary}")
            if child_obs.app_state.active_document:
                active_document_parts.append(f"{app}:{child_obs.app_state.active_document}")

        return Observation(
            meta=Meta(
                app_name=self.app_name,
                app_version="",
                observation_source="multi_app_composite",
            ),
            app_state=AppState(
                current_view=self.primary_app,
                active_document=", ".join(active_document_parts),
                document_path="",
            ),
            interactive_elements=elements,
            environment=Environment(system={"app_count": float(len(self.active_adapters))}),
            navigation=Navigation(
                available_views=[],
                current_path=self.primary_app,
                breadcrumb=list(self.related_apps),
                reachable_from_here=list(self.related_apps),
            ),
            data_summary=" | ".join(summaries),
        )

    def _coerce_child_action(self, app: str, action_spec: dict[str, Any] | Action) -> Action:
        if isinstance(action_spec, Action):
            action = action_spec
        else:
            action = Action(**action_spec)
        if action.target.startswith(f"{app}::"):
            action = action.model_copy(update={"target": action.target.split("::", 1)[1]})
        return self._canonicalize_child_action(app, action)

    @staticmethod
    def _path_from_local_target(local_target: str) -> str:
        for prefix in ("file:", "editor:", "tab:", "folder:"):
            if local_target.startswith(prefix):
                return local_target[len(prefix):]
        if local_target.startswith("cell:"):
            body = local_target[len("cell:"):]
            path, _sep, maybe_index = body.rpartition(":")
            return path if maybe_index.isdigit() else body
        if local_target.startswith("notebook:"):
            return local_target[len("notebook:"):]
        return local_target

    @staticmethod
    def _cell_index_from_local_target(local_target: str) -> int | None:
        if not local_target.startswith("cell:"):
            return None
        _path, sep, maybe_index = local_target[len("cell:"):].rpartition(":")
        if sep and maybe_index.isdigit():
            return int(maybe_index)
        return None

    @staticmethod
    def _task_number_from_values(*values: Any) -> str:
        for value in values:
            match = TOKEN_RE.search(str(value))
            if match:
                return match.group(1)
        return ""

    def _canonicalize_workspace_file_action(self, action: Action) -> Action:
        params = dict(action.params or {})
        local_target = str(action.target or "")
        relative_path = str(params.get("path") or self._path_from_local_target(local_target)).strip("/").replace("\\", "/")

        if action.action_type == "open":
            return Action(action_type="navigate", target=relative_path, params={})

        operation: dict[str, Any] | None = None
        if action.action_type == "create_file":
            operation = {
                "action": "create_file",
                "path": relative_path,
                "content": str(params.get("content", params.get("text", ""))),
            }
        elif action.action_type == "set_file_text":
            operation = {
                "action": "set_file_text",
                "path": relative_path,
                "content": str(params.get("content", params.get("text", ""))),
            }
        elif action.action_type == "append_text":
            operation = {
                "action": "append_text",
                "path": relative_path,
                "text": str(params.get("text", params.get("content", ""))),
            }
        elif action.action_type == "replace_text":
            operation = {
                "action": "replace_text",
                "path": relative_path,
                "old": str(params.get("old", "")),
                "new": str(params.get("new", params.get("text", params.get("content", "")))),
            }
        elif action.action_type == "rename":
            operation = {
                "action": "rename_path",
                "path": relative_path,
                "new_path": str(
                    params.get("new_path")
                    or params.get("new_name")
                    or params.get("to")
                    or params.get("target_path")
                    or ""
                ).strip("/").replace("\\", "/"),
            }
        elif action.action_type == "delete":
            operation = {"action": "delete_path", "path": relative_path}
        elif action.action_type in {"set_cell_source", "set_cell_output", "insert_cell", "delete_cell"}:
            operation = {"action": action.action_type, "path": relative_path}
            operation.update(params)
            if "cell_index" not in operation:
                index = self._cell_index_from_local_target(local_target)
                if index is not None:
                    operation["cell_index"] = index

        if operation is None:
            return action
        if operation.get("action") == "rename_path" and not operation.get("new_path"):
            return action
        return Action(
            action_type="modify_file",
            target="workspace",
            params={"operations": [operation]},
        )

    def _canonicalize_gitea_action(self, action: Action) -> Action:
        if action.action_type != "create_issue":
            return action
        params = dict(action.params or {})
        title = str(params.get("title") or params.get("name") or "")
        body = str(params.get("body") or params.get("content") or params.get("text") or "")
        if not title:
            return action
        return Action(
            action_type="api_call",
            target="gitea_rest",
            params={
                "method": "POST",
                "endpoint": "/api/v1/repos/{{owner}}/{{repo}}/issues",
                "body": {"title": title, "body": body},
            },
        )

    def _canonicalize_gimp_action(self, action: Action) -> Action:
        if action.action_type not in {"invoke_function", "modify_file"}:
            return action
        params = dict(action.params or {})
        operations = []
        for operation in params.get("operations", []):
            if not isinstance(operation, dict):
                continue
            op = dict(operation)
            if op.get("action") == "add_text_layer":
                token = self._task_number_from_values(op.get("text"), op.get("label"), op.get("id"))
                if token:
                    op["id"] = f"gimp_multi_{token}"
                op.setdefault("label", op.get("text", "Text"))
                op.setdefault("font_size", 26)
                op.setdefault("color", "#0f766e")
            operations.append(op)
        if not operations:
            return action
        return Action(action_type="invoke_function", target="gimp", params={**params, "operations": operations})

    def _canonicalize_drawio_action(self, action: Action) -> Action:
        if action.action_type != "modify_file":
            return action
        params = dict(action.params or {})
        operations = []
        for operation in params.get("operations", []):
            if not isinstance(operation, dict):
                continue
            op = dict(operation)
            if op.get("action") == "add_shape":
                token = self._task_number_from_values(op.get("label"), op.get("id"))
                if token:
                    op["id"] = f"draw_multi_{token}"
                op.setdefault("shape_kind", "rounded")
                op.setdefault("width", 150)
                op.setdefault("height", 70)
                op.setdefault("fill", "#d1fae5")
            operations.append(op)
        if not operations:
            return action
        return Action(action_type="modify_file", target="diagram", params={**params, "operations": operations})

    def _canonicalize_inkscape_action(self, action: Action) -> Action:
        if action.action_type != "modify_file":
            return action
        params = dict(action.params or {})
        operations = []
        for operation in params.get("operations", []):
            if not isinstance(operation, dict):
                continue
            op = dict(operation)
            if op.get("action") == "add_element":
                attrs = dict(op.get("attributes", {}))
                token = self._task_number_from_values(attrs.get("text_content"), attrs.get("id"), op.get("text"))
                if token:
                    attrs["id"] = f"ink_multi_{token}"
                if op.get("text") and "text_content" not in attrs:
                    attrs["text_content"] = str(op["text"])
                attrs.setdefault("x", "120")
                attrs.setdefault("y", "130")
                attrs.setdefault("style", "font-size:24px;fill:#0f766e")
                op["attributes"] = attrs
                op.setdefault("parent_xpath", "//svg:g[@id='layer1']")
                op.setdefault("tag", "text")
            operations.append(op)
        if not operations:
            return action
        return Action(action_type="modify_file", target=action.target or "test.svg", params={**params, "operations": operations})

    def _canonicalize_audacity_action(self, action: Action) -> Action:
        if action.action_type not in {"modify_file", "set_value", "invoke_function"}:
            return action
        params = dict(action.params or {})
        operations = []
        for operation in params.get("operations", []):
            if not isinstance(operation, dict):
                continue
            op = dict(operation)
            if op.get("action") == "add_label":
                token = self._task_number_from_values(op.get("text"), op.get("label_id"), op.get("id"))
                if token:
                    op["label_id"] = f"aud_multi_{token}"
                elif "label_id" not in op and "id" in op:
                    op["label_id"] = str(op["id"])
                op.setdefault("start", 5.0)
                op.setdefault("end", 6.0)
            operations.append(op)
        if not operations:
            return action
        return Action(action_type="modify_file", target="audacity_project", params={**params, "operations": operations})

    def _canonicalize_kdenlive_action(self, action: Action) -> Action:
        if action.action_type not in {"modify_file", "invoke_function"}:
            return action
        params = dict(action.params or {})
        operations = []
        for operation in params.get("operations", []):
            if not isinstance(operation, dict):
                continue
            op = dict(operation)
            op_name = str(op.get("action", ""))
            if op_name == "add_marker":
                comment = str(op.get("comment") or op.get("text") or op.get("label") or "")
                token = self._task_number_from_values(comment, op.get("id"))
                frame = str(op.get("frame") or (90 + int(token) if token else 120))
                op = {
                    "action": "add_element",
                    "parent_xpath": "./guides",
                    "tag": "marker",
                    "attributes": {
                        "id": f"kd_multi_{token}" if token else str(op.get("id", "kd_marker")),
                        "frame": frame,
                        "comment": comment,
                        "color": str(op.get("color", "#0f766e")),
                    },
                }
            elif op_name == "add_element" and str(op.get("tag", "")) == "marker":
                attrs = dict(op.get("attributes", {}))
                token = self._task_number_from_values(attrs.get("comment"), attrs.get("id"))
                if token:
                    attrs["id"] = f"kd_multi_{token}"
                    attrs.setdefault("frame", str(90 + int(token)))
                attrs.setdefault("color", "#0f766e")
                op["attributes"] = attrs
            operations.append(op)
        if not operations:
            return action
        return Action(action_type="modify_file", target="project", params={**params, "operations": operations})

    def _canonicalize_thunderbird_action(self, action: Action) -> Action:
        params = dict(action.params or {})
        local_target = str(action.target or "")
        operations = params.get("operations")
        if not isinstance(operations, list):
            if action.action_type == "add_tag":
                operations = [{"action": "add_tag", **params}]
            elif action.action_type == "invoke_function" and "operation" in params:
                operation = {"action": params.get("operation")}
                if "message_id" in params:
                    operation["id"] = params["message_id"]
                if "value" in params:
                    operation["tag"] = params["value"]
                if "tag" in params:
                    operation["tag"] = params["tag"]
                operations = [operation]
            else:
                return action
        normalized_ops = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            op = dict(operation)
            if "id" not in op and "message_id" in op:
                op["id"] = op["message_id"]
            if "id" not in op and local_target.startswith("message:"):
                op["id"] = local_target[len("message:"):]
            if op.get("action") in {"add_tag", "remove_tag"} and "tag" not in op and "value" in op:
                op["tag"] = op["value"]
            normalized_ops.append(op)
        if not normalized_ops:
            return action
        return Action(action_type="invoke_function", target="thunderbird", params={"operations": normalized_ops})

    def _canonicalize_nautilus_action(self, action: Action) -> Action:
        params = dict(action.params or {})
        local_target = str(action.target or "")

        def target_path() -> str:
            return str(params.get("path") or self._path_from_local_target(local_target)).strip("/").replace("\\", "/")

        operations: list[dict[str, Any]] = []
        if action.action_type == "modify_file":
            for operation in params.get("operations", []):
                if not isinstance(operation, dict):
                    continue
                op_name = str(operation.get("action", ""))
                if op_name == "rename_path":
                    new_path = str(operation.get("new_path") or operation.get("target_path") or "")
                    operations.append(
                        {
                            "action": "rename_entry",
                            "path": str(operation.get("path") or target_path()).strip("/").replace("\\", "/"),
                            "new_name": Path(new_path).name,
                        }
                    )
                elif op_name == "delete_path":
                    operations.append(
                        {
                            "action": "trash_entry",
                            "path": str(operation.get("path") or target_path()).strip("/").replace("\\", "/"),
                        }
                    )
                elif op_name in {"open_directory", "go_back", "set_hidden_visibility", "set_view_mode", "search_entries", "clear_search", "add_bookmark", "rename_entry", "move_entry", "copy_entry", "trash_entry"}:
                    operations.append(dict(operation))
        elif action.action_type == "rename":
            new_path = str(
                params.get("new_path")
                or params.get("new_name")
                or params.get("to")
                or params.get("target_path")
                or ""
            )
            operations.append(
                {
                    "action": "rename_entry",
                    "path": target_path(),
                    "new_name": Path(new_path).name,
                }
            )
        elif action.action_type in {"open_directory", "go_back", "set_hidden_visibility", "set_view_mode", "search_entries", "clear_search", "add_bookmark", "rename_entry", "move_entry", "copy_entry", "trash_entry"}:
            operation = {"action": action.action_type}
            operation.update(params)
            if "path" not in operation and action.action_type not in {"go_back", "clear_search"}:
                operation["path"] = target_path()
            operations.append(operation)

        if not operations or any(op.get("action") == "rename_entry" and not op.get("new_name") for op in operations):
            return action
        return Action(
            action_type="invoke_function",
            target="nautilus",
            params={"operations": operations},
        )

    def _canonicalize_child_action(self, app: str, action: Action) -> Action:
        child = self.active_adapters.get(app)
        app_specific = {
            "gimp": self._canonicalize_gimp_action,
            "drawio": self._canonicalize_drawio_action,
            "inkscape": self._canonicalize_inkscape_action,
            "audacity": self._canonicalize_audacity_action,
            "kdenlive": self._canonicalize_kdenlive_action,
            "thunderbird": self._canonicalize_thunderbird_action,
        }.get(app)
        if app_specific is not None:
            candidate = app_specific(action)
            if child is None or child.validate_action(candidate):
                return candidate
        if child is not None and child.validate_action(action):
            return action
        if app in WORKSPACE_FILE_APPS:
            candidate = self._canonicalize_workspace_file_action(action)
            if child is None or child.validate_action(candidate):
                return candidate
        if app == "gitea":
            candidate = self._canonicalize_gitea_action(action)
            if child is None or child.validate_action(candidate):
                return candidate
        if app == "nautilus":
            candidate = self._canonicalize_nautilus_action(action)
            if child is None or child.validate_action(candidate):
                return candidate
        return action

    def _dispatch_child_action(self, app: str, action_spec: dict[str, Any] | Action) -> Observation:
        if app not in self.active_adapters:
            raise ValueError(f"No active adapter for app {app!r}.")
        child_action = self._coerce_child_action(app, action_spec)
        return self.active_adapters[app].execute(child_action)

    def execute(self, action: Action) -> Observation:
        if action.action_type == "batch" and action.target == "multi_apps":
            for item in action.params.get("actions", []):
                if not isinstance(item, dict):
                    raise ValueError("multi_apps batch actions must be mappings.")
                app = str(item.get("app") or item.get("target") or "")
                if not app:
                    raise ValueError("multi_apps batch action is missing `app`.")
                child_spec = item.get("action")
                if not isinstance(child_spec, dict):
                    child_spec = {
                        "action_type": item.get("action_type"),
                        "target": item.get("target"),
                        "params": item.get("params", {}),
                    }
                self._dispatch_child_action(app, child_spec)
            return self.observe()

        if action.target in self.active_adapters:
            child_spec = action.params.get("action")
            if not isinstance(child_spec, dict):
                raise ValueError("Single-app multi_apps actions require params.action.")
            self._dispatch_child_action(action.target, child_spec)
            return self.observe()

        if "::" in action.target:
            app, _local = action.target.split("::", 1)
            self._dispatch_child_action(app, action)
            return self.observe()

        raise ValueError(f"Unsupported multi_apps action target: {action.target!r}")

    def validate_action(self, action: Action) -> bool:
        try:
            if action.action_type == "batch" and action.target == "multi_apps":
                actions = action.params.get("actions")
                if not isinstance(actions, list):
                    return False
                for item in actions:
                    if not isinstance(item, dict):
                        return False
                    app = str(item.get("app") or item.get("target") or "")
                    if app not in self.active_adapters:
                        return False
                    child_spec = item.get("action")
                    if isinstance(child_spec, dict):
                        child_action = self._coerce_child_action(app, child_spec)
                    else:
                        child_action = self._coerce_child_action(
                            app,
                            {
                                "action_type": item.get("action_type"),
                                "target": item.get("target"),
                                "params": item.get("params", {}),
                            },
                        )
                    if not self.active_adapters[app].validate_action(child_action):
                        return False
                return True
            if action.target in self.active_adapters and isinstance(action.params.get("action"), dict):
                child_action = self._coerce_child_action(action.target, action.params["action"])
                return self.active_adapters[action.target].validate_action(child_action)
            if "::" in action.target:
                app, _local = action.target.split("::", 1)
                if app not in self.active_adapters:
                    return False
                return self.active_adapters[app].validate_action(self._coerce_child_action(app, action))
        except Exception:
            return False
        return False

    def get_gui_session_spec(self) -> GUISessionSpec | None:
        if not self.active_adapters:
            return None
        from asil.gui_agent.session import resolve_gui_session_spec

        child_specs: dict[str, GUISessionSpec] = {}
        startup_timeout = 0.0
        ordered_apps = sorted(
            self.active_adapters,
            key=lambda app: (0 if app in BROWSER_SOFTWARE else 1, self.related_apps.index(app)),
        )
        for app in ordered_apps:
            adapter = self.active_adapters[app]
            child_spec = resolve_gui_session_spec(adapter)
            child_specs[app] = child_spec
            startup_timeout += float(child_spec.startup_timeout_s or 45.0)
        return GUISessionSpec(
            surface_type="multi_window",
            window_title_pattern=r".*",
            window_class_pattern=None,
            startup_timeout_s=max(startup_timeout, 60.0),
            post_launch_delay_s=1.0,
            min_width=1200,
            min_height=700,
            child_specs=child_specs,
            primary_child=self.primary_app,
            capture_active_window=True,
        )

    def gui_eval_mode(self) -> str:
        return "custom_sync_existing"

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="state_render",
            backend="wkhtmltoimage+html",
            actual_page=False,
            description="Synthetic multi-app composite state render",
        )

    @staticmethod
    def _render_value(value: Any) -> str:
        rendered = value if isinstance(value, str) else repr(value)
        if len(rendered) > 900:
            rendered = rendered[:900] + "..."
        return html.escape(rendered)

    def _composite_html(self) -> str:
        sections: list[str] = ["<h1>Multi-App Composite</h1>"]
        for app in self.related_apps:
            adapter = self.active_adapters.get(app)
            if adapter is None:
                continue
            child_obs = adapter.observe()
            items: list[str] = []
            for element in child_obs.interactive_elements[:24]:
                label = html.escape(element.label or element.id)
                element_id = html.escape(element.id)
                element_type = html.escape(element.type)
                value = self._render_value(element.value)
                items.append(
                    "<li>"
                    f"<strong>{element_type}</strong> <code>{element_id}</code> {label}"
                    f"<pre>{value}</pre>"
                    "</li>"
                )
            active = html.escape(child_obs.app_state.active_document or "")
            summary = html.escape(child_obs.data_summary or "")
            sections.append(
                "<section>"
                f"<h2>{html.escape(app)}</h2>"
                f"<p>Active: <code>{active}</code></p>"
                f"<p>{summary}</p>"
                "<ul>"
                + "".join(items)
                + "</ul>"
                "</section>"
            )
        body = (
            "<div class='panel' style='padding:18px;'>"
            "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;'>"
            + "".join(sections)
            + "</div></div>"
        )
        return html_page("Multi-App Composite", body)

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path) if output_path else self.tmp / "multi_apps.png"
        return capture_html_to_png(self._composite_html(), output)

    def clear_gui_shadow_state(self) -> None:
        super().clear_gui_shadow_state()
        for adapter in self.active_adapters.values():
            clear_shadow = getattr(adapter, "clear_gui_shadow_state", None)
            if callable(clear_shadow):
                clear_shadow()

    def sync_from_gui(self, session: Any | None = None) -> None:
        child_sessions = getattr(session, "child_sessions", {}) if session is not None else {}
        sync_errors: dict[str, str] = {}
        for app, adapter in self.active_adapters.items():
            child_session = child_sessions.get(app) if isinstance(child_sessions, dict) else None
            sync = getattr(adapter, "sync_from_gui", None)
            if callable(sync):
                try:
                    sync(child_session)
                except Exception as exc:
                    sync_errors[app] = f"{type(exc).__name__}: {exc}"
        self.__dict__["_last_gui_sync_errors"] = sync_errors

    def persist_gui_state(self, controller: Any, spec: GUISessionSpec) -> None:
        active_before = ""
        try:
            from asil.rendering import active_window_id

            active_before = active_window_id(display=getattr(controller, "display", None))
        except Exception:
            active_before = ""
        child_specs = spec.child_specs or {}
        try:
            for app, adapter in self.active_adapters.items():
                child_spec = child_specs.get(app)
                if child_spec is None:
                    continue
                if child_spec.persist_shortcuts:
                    controller.persist(child_spec)
                persist = getattr(adapter, "persist_gui_state", None)
                if callable(persist):
                    persist(controller, child_spec)
        finally:
            if active_before and callable(getattr(controller, "activate_window_id", None)):
                try:
                    controller.activate_window_id(active_before)
                except Exception:
                    pass
