"""ASIL adapter for Nautilus — filesystem-backed file-manager benchmark tasks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote_from_bytes, unquote_to_bytes, urlsplit

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    read_clipboard_text,
    capture_window_to_png,
    click_window_relative,
    ensure_user_access,
    ensure_virtual_display,
    launch_gui_process,
    send_keys_to_window,
    terminate_process,
    type_text_to_window,
    wait_for_window,
)

_NAUTILUS_WINDOW_PATTERN = r".*"
_NAUTILUS_WINDOW_CLASS_PATTERN = r"org.gnome.Nautilus|Org.gnome.Nautilus|nautilus"
_DEFAULT_VIEW_MODE = "list"
_SEARCH_BUTTON_OFFSET = (560, 24)

_DEFAULT_FILES: dict[str, str] = {
    "Inbox/agenda.txt": "Agenda for the weekly planning sync.\n",
    "Inbox/june-notes.txt": "June notes and follow-up items.\n",
    "Projects/alpha/draft-plan.md": "# Draft Plan\n\n- Review release scope\n",
    "Projects/alpha/alpha-notes.txt": "Alpha launch notes.\n",
    "Projects/alpha/.alpha-secret.txt": "Hidden alpha details.\n",
    "Projects/beta/milestones.txt": "Beta milestones\n",
    "Archive/old-invoice.txt": "Invoice 2025-04\n",
    "Images/logo.png": "not-a-real-png-but-visible-in-listing\n",
    "Reports/budget-q2.csv": "month,amount\nApr,1200\nMay,1400\n",
    "todo.txt": "Ship the file-manager benchmark.\n",
    "meeting-notes.md": "# Meeting Notes\n\n- Finalize the rollout.\n",
    ".env": "API_BASE=https://example.test\n",
}

_DEFAULT_DIRS = [
    "Inbox",
    "Projects",
    "Projects/alpha",
    "Projects/beta",
    "Archive",
    "Images",
    "Reports",
    "Trash",
]


class NautilusAdapter(ASILAdapter):
    app_name = "Nautilus"
    supported_action_types = ["invoke_function"]

    def __init__(self, workspace_path: str | Path, state_path: str | Path) -> None:
        self.workspace_path = Path(workspace_path).resolve()
        self.state_path = Path(state_path)
        if not self.workspace_path.exists():
            self._seed_workspace()
        if not self.state_path.exists():
            self.reset_state()

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: str | Path,
        sandbox=None,
        mock: bool = False,
    ) -> "NautilusAdapter":
        del sandbox, mock
        root = Path(tmp)
        return cls(root / "nautilus-workspace", root / "nautilus_state.json")

    @property
    def source_path(self) -> Path:
        return self.workspace_path

    def clone(self, new_path: Path) -> "NautilusAdapter":
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.copytree(self.workspace_path, new_path)
        cloned_state = new_path.parent / f"{new_path.name}.state.json"
        shutil.copy2(self.state_path, cloned_state)
        return NautilusAdapter(new_path, cloned_state)

    def get_context(self) -> dict[str, str]:
        state = self._load_state()
        return {
            "workspace_path": str(self.workspace_path),
            "state_path": str(self.state_path),
            "current_dir": state["current_dir"],
        }

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real Nautilus file-manager window",
        )

    def get_gui_session_spec(self) -> GUISessionSpec:
        state = self._load_state()
        current_dir = self._current_dir_path(state)
        nautilus_bin = shutil.which("nautilus")
        if nautilus_bin is None:
            raise RuntimeError("nautilus is not installed.")

        profile_home = self.state_path.parent / "_nautilus_home"
        config_home = profile_home / ".config"
        gtk_config = config_home / "gtk-3.0"
        gtk_config.mkdir(parents=True, exist_ok=True)
        self._write_bookmarks_file(gtk_config / "bookmarks", state["bookmarks"])
        ensure_user_access(self.workspace_path, run_as_user="asilgui")
        ensure_user_access(profile_home, run_as_user="asilgui")

        def _prime_window() -> None:
            self._set_show_hidden_preference(
                profile_home=profile_home,
                config_home=config_home,
                enabled=state["show_hidden"],
            )
            if state["view_mode"] == "grid":
                self._safe_send(["ctrl+1"])
                time.sleep(0.5)
            if state["search_query"]:
                click_window_relative(
                    _NAUTILUS_WINDOW_PATTERN,
                    _SEARCH_BUTTON_OFFSET[0],
                    _SEARCH_BUTTON_OFFSET[1],
                    timeout=20.0,
                    min_width=640,
                    min_height=480,
                )
                time.sleep(0.5)
                type_text_to_window(
                    _NAUTILUS_WINDOW_PATTERN,
                    state["search_query"],
                    timeout=20.0,
                    min_width=640,
                    min_height=480,
                )
                time.sleep(1.0)

        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(nautilus_bin, "--new-window", str(current_dir)),
            window_title_pattern=_NAUTILUS_WINDOW_PATTERN,
            window_class_pattern=_NAUTILUS_WINDOW_CLASS_PATTERN,
            run_as_user="asilgui",
            startup_timeout_s=45.0,
            post_launch_delay_s=6.0,
            post_launch_callback=_prime_window,
            min_width=640,
            min_height=480,
            extra_env={
                "HOME": str(profile_home),
                "XDG_CONFIG_HOME": str(config_home),
                "XDG_CACHE_HOME": str(profile_home / ".cache"),
            },
        )

    def reset_state(self) -> None:
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self._seed_workspace()
        self._write_state(
            {
                "current_dir": "",
                "show_hidden": False,
                "view_mode": _DEFAULT_VIEW_MODE,
                "search_query": "",
                "bookmarks": [],
            }
        )

    def setup_state(self, initial_state: str) -> None:
        self.reset_state()
        state = self._load_state()
        if initial_state == "inbox":
            state["current_dir"] = "Inbox"
        elif initial_state == "projects_alpha":
            state["current_dir"] = "Projects/alpha"
        elif initial_state == "projects_beta":
            state["current_dir"] = "Projects/beta"
        elif initial_state == "archive":
            state["current_dir"] = "Archive"
        elif initial_state == "images":
            state["current_dir"] = "Images"
        elif initial_state == "hidden_root":
            state["show_hidden"] = True
        elif initial_state == "hidden_alpha":
            state["current_dir"] = "Projects/alpha"
            state["show_hidden"] = True
        elif initial_state == "search_notes":
            state["search_query"] = "notes"
        elif initial_state == "bookmarks_ready":
            state["bookmarks"] = ["Projects/alpha", "Archive"]
        self._write_state(state)

    def prepare_task(self, task: Any) -> None:
        self.setup_state(getattr(task, "initial_state", "default") or "default")
        replacements = (getattr(task, "_taskgen", {}) or {}).get("replacements") or {}
        if isinstance(replacements, dict):
            self._seed_generated_replacements(replacements)
            state = self._load_state()
            current_dir = state.get("current_dir", "")
            for old, new in replacements.items():
                old_path = self._normalize_workspace_path(str(old))
                new_path = self._normalize_workspace_path(str(new))
                if current_dir == old_path and new_path:
                    state["current_dir"] = new_path
            self._write_state(state)

        for action in getattr(task, "actions", []) or []:
            for operation in (action.get("params") or {}).get("operations", []):
                if isinstance(operation, dict):
                    self._seed_operation_target(operation)

    def validate_action(self, action: Action) -> bool:
        return (
            action.action_type in self.supported_action_types
            and action.target == "nautilus"
            and isinstance(action.params.get("operations"), list)
        )

    def sync_from_gui(self, session=None) -> None:
        state = self._load_state()
        current_dir = self._read_gui_current_dir(session)
        if current_dir is not None:
            try:
                resolved = current_dir.resolve()
                workspace_root = self.workspace_path.resolve()
                if resolved == workspace_root:
                    state["current_dir"] = ""
                elif workspace_root in resolved.parents:
                    state["current_dir"] = resolved.relative_to(workspace_root).as_posix()
            except Exception:
                pass

        try:
            state["bookmarks"] = self._read_gui_bookmarks()
        except Exception:
            pass

        self._write_state(state)

    def observe(self) -> Observation:
        state = self._load_state()
        current_dir = self._current_dir_path(state)
        visible_entries = self._visible_entries(state)
        elements: list[Element] = [
            Element(
                id="location",
                type="location",
                label="Current Location",
                value={
                    "path": state["current_dir"] or "/",
                    "display_name": current_dir.name if state["current_dir"] else "Home",
                },
                editable=False,
                actions=["open_directory", "go_back"],
            ),
            Element(
                id="view_settings",
                type="view_settings",
                label="View Settings",
                value={
                    "view_mode": state["view_mode"],
                    "show_hidden": state["show_hidden"],
                },
                editable=True,
                actions=["set_view_mode", "set_hidden_visibility"],
            ),
            Element(
                id="search_box",
                type="search",
                label="Search",
                value={"query": state["search_query"]},
                editable=True,
                actions=["search_entries", "clear_search"],
            ),
        ]

        for bookmark in state["bookmarks"]:
            elements.append(
                Element(
                    id=f"bookmark:{bookmark}",
                    type="bookmark",
                    label=Path(bookmark).name,
                    value={"path": bookmark},
                    editable=True,
                    actions=["open_directory"],
                )
            )

        for entry in visible_entries:
            relative_path = entry.relative_to(self.workspace_path).as_posix()
            elements.append(
                Element(
                    id=f"entry:{relative_path}",
                    type="directory_entry",
                    label=entry.name,
                    value={
                        "name": entry.name,
                        "path": relative_path,
                        "is_dir": entry.is_dir(),
                        "parent": entry.parent.relative_to(self.workspace_path).as_posix()
                        if entry.parent != self.workspace_path
                        else "/",
                    },
                    editable=True,
                    actions=["open_directory", "rename_entry", "move_entry", "copy_entry", "trash_entry"],
                )
            )

        for entry in sorted(self.workspace_path.rglob("*")):
            relative_path = entry.relative_to(self.workspace_path).as_posix()
            elements.append(
                Element(
                    id=f"workspace:{relative_path}",
                    type="workspace_entry",
                    label=entry.name,
                    value={
                        "name": entry.name,
                        "path": relative_path,
                        "is_dir": entry.is_dir(),
                        "parent": entry.parent.relative_to(self.workspace_path).as_posix()
                        if entry.parent != self.workspace_path
                        else "/",
                    },
                    editable=False,
                    actions=[],
                )
            )

        return self._build_observation(
            source="filesystem",
            elements=elements,
            app_state={
                "current_view": "browser",
                "active_document": current_dir.name if state["current_dir"] else "Home",
                "document_path": str(current_dir),
            },
            navigation={
                "current_path": state["current_dir"] or "/",
                "breadcrumb": [part for part in PurePosixPath(state["current_dir"]).parts if part] or ["Home"],
            },
            data_summary=(
                f"{len(visible_entries)} visible entries in {state['current_dir'] or '/'}; "
                f"search={state['search_query'] or 'off'}; hidden={state['show_hidden']}"
            ),
        )

    def execute(self, action: Action) -> Observation:
        if not self.validate_action(action):
            raise ValueError(f"Unsupported Nautilus action: {action}")

        state = self._load_state()
        for operation in action.params.get("operations", []):
            op_name = str(operation.get("action", ""))
            if op_name == "open_directory":
                target = self._resolve(operation["path"], state=state)
                if not target.is_dir():
                    raise FileNotFoundError(f"Directory does not exist: {operation['path']}")
                state["current_dir"] = target.relative_to(self.workspace_path).as_posix()
                if state["current_dir"] == ".":
                    state["current_dir"] = ""
                state["search_query"] = ""
            elif op_name == "go_back":
                current = self._current_dir_path(state)
                parent = current.parent if current != self.workspace_path else current
                state["current_dir"] = "" if parent == self.workspace_path else parent.relative_to(self.workspace_path).as_posix()
                state["search_query"] = ""
            elif op_name == "set_hidden_visibility":
                state["show_hidden"] = bool(operation["enabled"])
            elif op_name == "set_view_mode":
                mode = str(operation["mode"])
                if mode not in {"list", "grid"}:
                    raise ValueError(f"Unsupported Nautilus view mode: {mode}")
                state["view_mode"] = mode
            elif op_name == "search_entries":
                state["search_query"] = str(operation["query"]).strip()
            elif op_name == "clear_search":
                state["search_query"] = ""
            elif op_name == "add_bookmark":
                target = self._resolve(operation["path"], state=state)
                if not target.is_dir():
                    raise FileNotFoundError(f"Bookmark target must be a directory: {operation['path']}")
                rel = target.relative_to(self.workspace_path).as_posix()
                if rel not in state["bookmarks"]:
                    state["bookmarks"].append(rel)
            elif op_name == "rename_entry":
                src = self._resolve(operation["path"], state=state)
                new_name = str(operation["new_name"])
                normalized_new_name = self._normalize_workspace_path(new_name)
                if "/" in normalized_new_name:
                    dst = (self._current_dir_path(state) / normalized_new_name).resolve()
                    workspace_root = self.workspace_path.resolve()
                    if workspace_root != dst and workspace_root not in dst.parents:
                        raise ValueError(f"Path escapes workspace root: {new_name}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                else:
                    dst = src.with_name(new_name)
                src.rename(dst)
            elif op_name == "move_entry":
                src = self._resolve(operation["path"], state=state)
                destination_dir = self._resolve(operation["destination_dir"], state=state)
                if not destination_dir.is_dir():
                    raise FileNotFoundError(f"Destination directory does not exist: {operation['destination_dir']}")
                shutil.move(str(src), str(destination_dir / src.name))
            elif op_name == "copy_entry":
                src = self._resolve(operation["path"], state=state)
                destination_dir = self._resolve(operation["destination_dir"], state=state)
                if not destination_dir.is_dir():
                    raise FileNotFoundError(f"Destination directory does not exist: {operation['destination_dir']}")
                dst = destination_dir / src.name
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            elif op_name == "trash_entry":
                src = self._resolve(operation["path"], state=state)
                trash_dir = self.workspace_path / "Trash"
                trash_dir.mkdir(exist_ok=True)
                shutil.move(str(src), str(trash_dir / src.name))
            else:
                raise ValueError(f"Unknown Nautilus operation: {op_name}")

        self._write_state(state)
        return self.observe()

    def render_to_png(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        current_dir = self._current_dir_path(state)
        nautilus_bin = shutil.which("nautilus")
        if nautilus_bin is None:
            raise RuntimeError("nautilus is not installed.")

        profile_home = self.state_path.parent / "_nautilus_home"
        config_home = profile_home / ".config"
        gtk_config = config_home / "gtk-3.0"
        gtk_config.mkdir(parents=True, exist_ok=True)
        self._write_bookmarks_file(gtk_config / "bookmarks", state["bookmarks"])
        ensure_user_access(self.workspace_path, run_as_user="asilgui")
        ensure_user_access(profile_home, run_as_user="asilgui")
        self._set_show_hidden_preference(
            profile_home=profile_home,
            config_home=config_home,
            enabled=state["show_hidden"],
        )

        proc = launch_gui_process(
            [nautilus_bin, "--new-window", str(current_dir)],
            run_as_user="asilgui",
            extra_env={
                "HOME": str(profile_home),
                "XDG_CONFIG_HOME": str(config_home),
                "XDG_CACHE_HOME": str(profile_home / ".cache"),
            },
        )
        try:
            time.sleep(6.0)
            if state["view_mode"] == "grid":
                self._safe_send(["ctrl+1"])
            if state["search_query"]:
                click_window_relative(
                    _NAUTILUS_WINDOW_PATTERN,
                    _SEARCH_BUTTON_OFFSET[0],
                    _SEARCH_BUTTON_OFFSET[1],
                    timeout=20.0,
                    min_width=640,
                    min_height=480,
                )
                time.sleep(0.5)
                type_text_to_window(
                    _NAUTILUS_WINDOW_PATTERN,
                    state["search_query"],
                    timeout=20.0,
                    min_width=640,
                    min_height=480,
                )
                time.sleep(1.0)
            capture_metadata = {"capture_complete": True}
            capture_window_to_png(
                out,
                title_pattern=_NAUTILUS_WINDOW_PATTERN,
                window_class_pattern=_NAUTILUS_WINDOW_CLASS_PATTERN,
                timeout=45.0,
                margin=12,
                settle_delay=2.0,
                min_width=640,
                min_height=480,
                capture_metadata=capture_metadata,
            )
            self._last_capture_complete = bool(capture_metadata.get("capture_complete", True))
        finally:
            terminate_process(proc)
        return out

    def _seed_workspace(self) -> None:
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        for directory in _DEFAULT_DIRS:
            (self.workspace_path / directory).mkdir(parents=True, exist_ok=True)
        for relative_path, content in _DEFAULT_FILES.items():
            file_path = self.workspace_path / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _normalize_workspace_path(raw_path: str) -> str:
        text = str(raw_path).strip().replace("\\", "/")
        for prefix in ("/home/oai/share/", "/home/oai/share", "/home/user/", "/home/user"):
            if text == prefix.rstrip("/"):
                return ""
            if text.startswith(prefix):
                text = text.removeprefix(prefix).lstrip("/")
                break
        if text == "/":
            return ""
        return text.strip("/")

    def _seed_generated_replacements(self, replacements: dict[str, Any]) -> None:
        for old, new in replacements.items():
            old_rel = self._normalize_workspace_path(str(old))
            new_rel = self._normalize_workspace_path(str(new))
            if not new_rel or any(part == ".." for part in PurePosixPath(new_rel).parts):
                continue
            new_path = self.workspace_path / new_rel
            old_path = self.workspace_path / old_rel if old_rel else self.workspace_path
            if new_path.exists():
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if old_path.exists() and old_path.is_dir():
                shutil.copytree(old_path, new_path, dirs_exist_ok=True)
            elif old_path.exists() and old_path.is_file():
                shutil.copy2(old_path, new_path)
            elif Path(new_rel).suffix:
                new_path.write_text(f"Generated placeholder for {new_rel}\n", encoding="utf-8")
            else:
                new_path.mkdir(parents=True, exist_ok=True)

    def _seed_operation_target(self, operation: dict[str, Any]) -> None:
        op_name = str(operation.get("action") or "")
        state = self._load_state()
        current_dir = self._current_dir_path(state)

        def seed(rel: str, *, directory: bool) -> None:
            if not rel or any(part == ".." for part in PurePosixPath(rel).parts):
                return
            for base in (self.workspace_path, current_dir):
                path = base / rel
                if path.exists():
                    continue
                if directory:
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"Generated placeholder for {rel}\n", encoding="utf-8")

        for key in ("destination_dir",):
            if operation.get(key):
                rel = self._normalize_workspace_path(str(operation[key]))
                if rel:
                    seed(rel, directory=True)

        if op_name in {"open_directory", "add_bookmark"} and operation.get("path"):
            rel = self._normalize_workspace_path(str(operation["path"]))
            if rel:
                seed(rel, directory=True)
            return

        if op_name in {"rename_entry", "move_entry", "copy_entry", "trash_entry"} and operation.get("path"):
            rel = self._normalize_workspace_path(str(operation["path"]))
            if not rel:
                return
            new_name = str(operation.get("new_name") or "")
            if Path(rel).suffix or Path(new_name).suffix:
                seed(rel, directory=False)
            else:
                seed(rel, directory=True)

    def _load_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _current_dir_path(self, state: dict[str, Any]) -> Path:
        return self.workspace_path if not state["current_dir"] else self.workspace_path / state["current_dir"]

    def _visible_entries(self, state: dict[str, Any]) -> list[Path]:
        current_dir = self._current_dir_path(state)
        if state["search_query"]:
            query = state["search_query"].lower()
            entries = [
                path
                for path in sorted(current_dir.rglob("*"))
                if query in path.name.lower() and (state["show_hidden"] or not path.name.startswith("."))
            ]
            return [path for path in entries if path != current_dir]
        return [
            path
            for path in sorted(current_dir.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            if state["show_hidden"] or not path.name.startswith(".")
        ]

    def _resolve(self, raw_path: str, *, state: dict[str, Any]) -> Path:
        current_dir = self._current_dir_path(state)
        if str(raw_path).strip() == "/":
            return self.workspace_path.resolve()
        normalized = self._normalize_workspace_path(str(raw_path))
        workspace_root = self.workspace_path.resolve()
        if normalized in {"", "."}:
            return current_dir.resolve()
        first_part = PurePosixPath(normalized).parts[0] if PurePosixPath(normalized).parts else ""
        if first_part in {"Inbox", "Projects", "Archive", "Images", "Reports", "Trash", "Reference"}:
            candidate = (workspace_root / normalized).resolve()
            if workspace_root == candidate or workspace_root in candidate.parents:
                return candidate
        candidate = (current_dir / normalized).resolve()
        if workspace_root == candidate or workspace_root in candidate.parents:
            return candidate
        candidate = (workspace_root / normalized).resolve()
        if workspace_root == candidate or workspace_root in candidate.parents:
            return candidate
        raise ValueError(f"Path escapes workspace root: {raw_path}")

    def _write_bookmarks_file(self, path: Path, bookmarks: list[str]) -> None:
        lines = []
        for bookmark in bookmarks:
            bookmark_path = (self.workspace_path / bookmark).resolve()
            uri = "file://" + quote_from_bytes(str(bookmark_path).encode("utf-8"), safe="/:")
            lines.append(f"{uri} {bookmark_path.name}")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _set_show_hidden_preference(
        self,
        *,
        profile_home: Path,
        config_home: Path,
        enabled: bool,
    ) -> None:
        gsettings = shutil.which("gsettings")
        if gsettings is None:
            return
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(profile_home),
                "XDG_CONFIG_HOME": str(config_home),
                "XDG_CACHE_HOME": str(profile_home / ".cache"),
            }
        )
        run_as_user = ["runuser", "-u", "asilgui", "--"] if os.geteuid() == 0 and shutil.which("runuser") else []
        value = "true" if enabled else "false"
        for schema, key in (
            ("org.gnome.nautilus.preferences", "show-hidden-files"),
            ("org.gtk.Settings.FileChooser", "show-hidden"),
        ):
            subprocess.run(
                [*run_as_user, gsettings, "set", schema, key, value],
                check=False,
                capture_output=True,
                env=env,
            )

    def _safe_send(self, keys: list[str]) -> None:
        try:
            send_keys_to_window(
                _NAUTILUS_WINDOW_PATTERN,
                keys,
                timeout=20.0,
                min_width=640,
                min_height=480,
            )
        except Exception:
            pass

    def _send_keys_to_live_window(
        self,
        keys: list[str],
        *,
        preferred_window_id: str | None = None,
    ) -> str:
        xdotool = shutil.which("xdotool")
        if xdotool is None:
            raise RuntimeError("xdotool is required for Nautilus GUI state synchronization.")

        gui_env = os.environ.copy()
        gui_env.update(ensure_virtual_display())

        def send_to_active_focus(window_id: str) -> None:
            subprocess.run(
                [xdotool, "windowactivate", "--sync", window_id],
                check=True,
                capture_output=True,
                env=gui_env,
            )
            subprocess.run(
                [xdotool, "key", "--clearmodifiers", *keys],
                check=True,
                capture_output=True,
                env=gui_env,
            )

        if preferred_window_id:
            try:
                send_to_active_focus(preferred_window_id)
                return preferred_window_id
            except (OSError, subprocess.SubprocessError):
                pass

        live_window_id = wait_for_window(
            _NAUTILUS_WINDOW_PATTERN,
            timeout=20.0,
            min_width=640,
            min_height=480,
            window_class_pattern=_NAUTILUS_WINDOW_CLASS_PATTERN,
        )
        send_to_active_focus(live_window_id)
        return live_window_id

    @staticmethod
    def _decode_local_file_uri(uri: str) -> Path | None:
        parsed = urlsplit(uri.strip())
        if parsed.scheme.casefold() != "file" or parsed.netloc not in {"", "localhost"}:
            return None
        return Path(os.fsdecode(unquote_to_bytes(parsed.path)))

    @classmethod
    def _path_from_clipboard(cls, clipboard_value: str) -> Path | None:
        for raw_line in clipboard_value.replace("\r\n", "\n").split("\n"):
            line = raw_line.strip().strip("\x00")
            if not line:
                continue
            if line.casefold().startswith("file:"):
                decoded = cls._decode_local_file_uri(line)
                if decoded is not None:
                    return decoded
                continue
            candidate = Path(line)
            if candidate.is_absolute():
                return candidate
        return None

    def _read_gui_current_dir(self, session=None) -> Path | None:
        window_id = str(getattr(session, "last_capture_window_id", "") or "") or None
        try:
            window_id = self._send_keys_to_live_window(
                ["ctrl+l"],
                preferred_window_id=window_id,
            )
            time.sleep(0.25)
            window_id = self._send_keys_to_live_window(
                ["ctrl+c"],
                preferred_window_id=window_id,
            )
            time.sleep(0.15)
            clipboard_value = read_clipboard_text().strip()
            return self._path_from_clipboard(clipboard_value)
        except Exception:
            return None
        finally:
            try:
                self._send_keys_to_live_window(["Escape"], preferred_window_id=window_id)
            except Exception:
                pass

    def _read_gui_bookmarks(self) -> list[str]:
        bookmarks_file = self.state_path.parent / "_nautilus_home" / ".config" / "gtk-3.0" / "bookmarks"
        if not bookmarks_file.exists():
            return []

        entries: list[str] = []
        workspace_root = self.workspace_path.resolve()
        for line in bookmarks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            uri = line.split(" ", 1)[0]
            if not uri.startswith("file://"):
                continue
            bookmark_path = self._decode_local_file_uri(uri)
            if bookmark_path is None:
                continue
            try:
                resolved = bookmark_path.resolve()
            except Exception:
                continue
            if resolved == workspace_root:
                entries.append("/")
            elif workspace_root in resolved.parents:
                entries.append(resolved.relative_to(workspace_root).as_posix())
        return entries
