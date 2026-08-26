"""ASIL adapter for code-server — Pattern C with filesystem-backed state."""

from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import RenderArtifact, capture_html_to_png, capture_url_to_png, html_page


_DEFAULT_FILES: dict[str, str] = {
    "README.md": "# Project checklist\n\n- Review the workspace\n- Update the app message\n- Ship the slice\n",
    "src/app.py": 'def main() -> str:\n    return "Hello from ASIL"\n',
    "src/utils.py": "def format_status(name: str) -> str:\n    return f\"Status: {name}\"\n",
    "notes/backlog.md": "# Backlog\n\n- Write tests\n- Refine copy\n",
    "config/settings.json": '{\n  "theme": "light",\n  "autosave": false,\n  "tabSize": 4\n}\n',
}


class CodeServerAdapter(ASILAdapter):
    app_name = "code-server"
    supported_action_types = ["modify_file", "navigate"]

    def __init__(self, workspace_path: str | Path, base_url: str = "", active_file: str = "README.md") -> None:
        self.workspace_path = Path(workspace_path)
        self.base_url = base_url.rstrip("/")
        self._default_active_file = active_file
        self._open_tabs: list[str] = [active_file]
        self._active_file = active_file
        self.clear_gui_shadow_state()

    def gui_eval_mode(self) -> str:
        return "live_shadow_required"

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: Path,
        sandbox=None,
        mock: bool = False,
    ) -> "CodeServerAdapter":
        shared_root = os.environ.get("CODE_SERVER_WORKSPACE_ROOT", "").strip()
        workspace = Path(shared_root) / "code-server-workspace" if shared_root else tmp / "code-server-workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        for relative_path, content in _DEFAULT_FILES.items():
            file_path = workspace / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        base_url = os.environ.get("CODE_SERVER_URL", "").strip()
        adapter = cls(workspace_path=workspace, base_url=base_url, active_file="README.md")
        adapter._make_shared_workspace_gui_writable()
        return adapter

    @property
    def source_path(self) -> Path:
        return self.workspace_path

    def _make_shared_workspace_gui_writable(self) -> None:
        shared_root = os.environ.get("CODE_SERVER_WORKSPACE_ROOT", "").strip()
        if not shared_root or not self.workspace_path.exists():
            return
        try:
            self.workspace_path.resolve().relative_to(Path(shared_root).resolve())
        except ValueError:
            return
        for path in (self.workspace_path, *self.workspace_path.rglob("*")):
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode | (0o333 if path.is_dir() else 0o222))

    def clone(self, new_path: Path) -> "CodeServerAdapter":
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.copytree(self.workspace_path, new_path)
        cloned = CodeServerAdapter(new_path, base_url=self.base_url, active_file=self._active_file)
        cloned._open_tabs = list(self._open_tabs)
        return cloned

    def reset_state(self) -> None:
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        for relative_path, content in _DEFAULT_FILES.items():
            file_path = self.workspace_path / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        self._open_tabs = [self._default_active_file]
        self._active_file = self._default_active_file
        self.clear_gui_shadow_state()
        self._make_shared_workspace_gui_writable()

    def prepare_task(self, task: Any) -> None:
        """Reset and seed task-specific files referenced by generated tasks."""
        self.reset_state()
        self.setup_state(getattr(task, "initial_state", "default") or "default")
        seed_by_path: dict[str, str] = {}
        for action in getattr(task, "actions", []) or []:
            for operation in (action.get("params") or {}).get("operations", []):
                if not isinstance(operation, dict) or operation.get("action") != "replace_text":
                    continue
                path = str(operation.get("path") or "").strip("/").replace("\\", "/")
                if path:
                    seed_by_path[path] = str(operation.get("old", ""))
        for action in getattr(task, "actions", []) or []:
            for operation in (action.get("params") or {}).get("operations", []):
                if not isinstance(operation, dict):
                    continue
                op_name = str(operation.get("action", ""))
                path = str(operation.get("path") or "").strip("/").replace("\\", "/")
                if not path:
                    continue
                if op_name in {"create_file", "set_file_text"}:
                    continue
                file_path = self.workspace_path / path
                if file_path.exists():
                    continue
                file_path.parent.mkdir(parents=True, exist_ok=True)
                seed = ""
                if path in seed_by_path:
                    seed = seed_by_path[path]
                elif op_name == "replace_text":
                    seed = str(operation.get("old", ""))
                elif op_name == "append_text":
                    seed = "# Task workspace seed\n"
                file_path.write_text(seed, encoding="utf-8")
        self._make_shared_workspace_gui_writable()

    def setup_state(self, initial_state: str) -> None:
        state_to_file = {
            "default": "README.md",
            "src_focus": "src/app.py",
            "notes_focus": "notes/backlog.md",
            "config_focus": "config/settings.json",
        }
        selected = state_to_file.get(initial_state, self._default_active_file)
        self._set_active_file(selected)
        self.clear_gui_shadow_state()

    def get_context(self) -> dict[str, str]:
        return {
            "workspace_path": str(self.workspace_path),
            "active_file": self._active_file,
        }

    def _set_active_file(self, relative_path: str) -> None:
        normalized = relative_path.strip("/").replace("\\", "/")
        file_path = self.workspace_path / normalized
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Workspace file does not exist: {normalized}")
        self._active_file = normalized
        if normalized not in self._open_tabs:
            self._open_tabs.append(normalized)

    def _read_text(self, relative_path: str) -> str:
        return (self.workspace_path / relative_path).read_text(encoding="utf-8")

    def _write_text(self, relative_path: str, content: str) -> None:
        file_path = self.workspace_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    def _iter_workspace_elements(self) -> list[Element]:
        shadow = self._get_gui_shadow_state() or {}
        text_files = shadow.get("text_files", {})
        elements: list[Element] = []
        for path in sorted(self.workspace_path.rglob("*")):
            if path.name.startswith("."):
                continue
            relative = path.relative_to(self.workspace_path).as_posix()
            if path.is_dir():
                elements.append(
                    Element(
                        id=f"folder:{relative}",
                        type="folder",
                        label=path.name,
                        value={"path": relative},
                        editable=False,
                        actions=["open"],
                    )
                )
                continue

            content = str(text_files.get(relative, path.read_text(encoding="utf-8")))
            elements.append(
                Element(
                    id=f"file:{relative}",
                    type="file",
                    label=path.name,
                    value={
                        "path": relative,
                        "extension": path.suffix,
                        "line_count": len(content.splitlines()),
                        "content": content,
                    },
                    editable=True,
                    actions=["open", "replace_text", "append_text", "rename", "delete"],
                )
            )
        return elements

    def observe(self) -> Observation:
        shadow = self._get_gui_shadow_state() or {}
        open_tabs = list(shadow.get("open_tabs", self._open_tabs))
        active_file = str(shadow.get("active_file", self._active_file))
        text_files = shadow.get("text_files", {})
        elements = self._iter_workspace_elements()
        for tab_path in open_tabs:
            elements.append(
                Element(
                    id=f"tab:{tab_path}",
                    type="tab",
                    label=Path(tab_path).name,
                    value={"path": tab_path, "active": tab_path == active_file},
                    editable=False,
                    actions=["activate", "close"],
                )
            )

        active_content = str(text_files.get(active_file, self._read_text(active_file)))
        elements.append(
            Element(
                id=f"editor:{active_file}",
                type="editor",
                label=active_file,
                value={
                    "path": active_file,
                    "content": active_content,
                    "line_count": len(active_content.splitlines()),
                },
                editable=True,
                actions=["replace_text", "append_text", "set_file_text"],
            )
        )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "editor",
                "active_document": Path(active_file).name,
                "document_path": str(self.workspace_path / active_file),
            },
            navigation={
                "current_path": active_file,
                "breadcrumb": active_file.split("/"),
            },
            data_summary=(
                f"Workspace with {len([e for e in elements if e.type == 'file'])} files; "
                f"active file is {active_file}"
            ),
        )

    def sync_from_gui(self, session=None) -> None:
        if session is None or getattr(session, "browser_page", None) is None:
            return
        page = session.browser_page
        try:
            payload = page.evaluate(
                """
                () => {
                  const models = [];
                  try {
                    const monacoModels = globalThis.monaco?.editor?.getModels?.() || [];
                    for (const model of monacoModels) {
                      const uri = model.uri || {};
                      const rawPath = uri.path || uri.fsPath || '';
                      models.push({
                        path: String(rawPath || ''),
                        value: String(model.getValue()),
                      });
                    }
                  } catch (error) {
                  }
                  const tabs = Array.from(document.querySelectorAll('.tabs-container .tab')).map((tab) => ({
                    label: (tab.getAttribute('aria-label') || tab.textContent || '').trim(),
                    active: tab.classList.contains('active') || tab.getAttribute('aria-selected') === 'true',
                  })).filter((tab) => tab.label);
                  return { tabs, models };
                }
                """
            )
        except Exception:
            fallback_files: dict[str, str] = {}
            if self._active_file:
                active_path = self.workspace_path / self._active_file
                if active_path.exists() and active_path.is_file():
                    fallback_files[self._active_file] = self._read_text(self._active_file)
            self._set_gui_shadow_state(
                {
                    "open_tabs": list(self._open_tabs),
                    "active_file": self._active_file,
                    "text_files": fallback_files,
                }
            )
            return
        if not isinstance(payload, dict):
            return
        resolved_models: dict[str, str] = {}
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            relative = self._resolve_model_path(str(item.get("path", "")))
            if relative:
                resolved_models[relative] = str(item.get("value", ""))
        resolved_tabs = []
        active_file = self._active_file
        for item in payload.get("tabs", []):
            if not isinstance(item, dict):
                continue
            relative = self._resolve_visible_path(str(item.get("label", "")))
            if not relative:
                continue
            if relative not in resolved_tabs:
                resolved_tabs.append(relative)
            if item.get("active"):
                active_file = relative
        self._set_gui_shadow_state(
            {
                "open_tabs": resolved_tabs or list(self._open_tabs),
                "active_file": active_file,
                "text_files": resolved_models,
            }
        )

    def _resolve_model_path(self, raw_path: str) -> str:
        text = raw_path.strip()
        if not text:
            return ""
        for candidate in (
            text,
            text.removeprefix(str(self.workspace_path)),
            text.removeprefix("/"),
            Path(text).name,
        ):
            normalized = candidate.strip("/").replace("\\", "/")
            if normalized and (self.workspace_path / normalized).exists():
                return normalized
        by_name = [tab for tab in self._open_tabs if Path(tab).name == Path(text).name]
        if len(by_name) == 1:
            return by_name[0]
        return ""

    def _resolve_visible_path(self, label: str) -> str:
        normalized = label.replace("•", "").strip()
        if not normalized:
            return ""
        if "(" in normalized and normalized.endswith(")"):
            normalized = normalized.split("(", 1)[0].strip()
        if normalized in self._open_tabs:
            return normalized
        by_name = [tab for tab in self._open_tabs if Path(tab).name == normalized]
        if len(by_name) == 1:
            return by_name[0]
        filesystem_matches = [
            path.relative_to(self.workspace_path).as_posix()
            for path in self.workspace_path.rglob("*")
            if path.is_file() and path.name == normalized
        ]
        if len(filesystem_matches) == 1:
            return filesystem_matches[0]
        return ""

    def execute(self, action: Action) -> Observation:
        operations = action.params.get("operations", [])
        if action.action_type == "navigate" and action.target:
            operations = [{"action": "open_file", "path": action.target}]

        for operation in operations:
            op_name = operation.get("action", "")
            relative_path = str(operation.get("path", self._active_file)).strip("/").replace("\\", "/")

            if op_name == "open_file":
                self._set_active_file(relative_path)
            elif op_name == "create_file":
                self._write_text(relative_path, str(operation.get("content", "")))
            elif op_name == "set_file_text":
                self._write_text(relative_path, str(operation.get("content", "")))
            elif op_name == "append_text":
                existing = self._read_text(relative_path) if (self.workspace_path / relative_path).exists() else ""
                self._write_text(relative_path, existing + str(operation.get("text", "")))
            elif op_name == "replace_text":
                old = str(operation["old"])
                new = str(operation["new"])
                content = self._read_text(relative_path)
                count = operation.get("count")
                if count is None:
                    updated = content.replace(old, new)
                else:
                    updated = content.replace(old, new, int(count))
                self._write_text(relative_path, updated)
            elif op_name == "rename_path":
                new_path = str(operation["new_path"]).strip("/").replace("\\", "/")
                src = self.workspace_path / relative_path
                dst = self.workspace_path / new_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                self._open_tabs = [new_path if tab == relative_path else tab for tab in self._open_tabs]
                if self._active_file == relative_path:
                    self._active_file = new_path
            elif op_name == "delete_path":
                target = self.workspace_path / relative_path
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                self._open_tabs = [tab for tab in self._open_tabs if tab != relative_path]
                if self._active_file == relative_path:
                    fallback = self._open_tabs[-1] if self._open_tabs else self._default_active_file
                    self._set_active_file(fallback)
            else:
                raise ValueError(f"Unsupported code-server operation: {op_name}")

        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def get_gui_session_spec(self) -> GUISessionSpec | None:
        if not self.base_url:
            return None
        return GUISessionSpec(
            surface_type="browser",
            browser_url="about:blank",
            browser_navigation_mode="current_page",
            window_title_pattern=r".*",
            window_class_pattern=r"chromium|Chromium|chrome|Google-chrome",
            startup_timeout_s=120.0,
            min_width=1000,
            min_height=700,
            post_launch_delay_s=3.0,
            post_launch_callback=self._prime_browser_session,
            backend_ready_probe=self._probe_backend_ready,
            ui_ready_probe=self._probe_ui_ready,
        )

    def _probe_backend_ready(self) -> None:
        from asil.gui_agent.session import GUISessionStartupError

        try:
            response = requests.get(self.base_url, timeout=10)
        except requests.RequestException as exc:
            raise GUISessionStartupError("backend_unready", f"code-server backend is not reachable at {self.base_url}.") from exc
        if response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"code-server backend returned HTTP {response.status_code} for {self.base_url}.",
            )
        target_url = self._workspace_url()
        try:
            target_response = requests.get(target_url, timeout=10)
        except requests.RequestException as exc:
            raise GUISessionStartupError(
                "backend_unready",
                f"code-server target page is not reachable at {target_url}.",
            ) from exc
        if target_response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"code-server target page returned HTTP {target_response.status_code} for {target_url}.",
            )

    def _prime_browser_session(self, session) -> None:
        from asil.gui_agent.session import navigate_browser_target

        page = session.browser_page
        target_url = self._workspace_url()
        if not str(page.url).startswith(target_url):
            navigate_browser_target(session, target_url, timeout_ms=60_000)
            page = session.browser_page
        try:
            page.wait_for_load_state("domcontentloaded", timeout=5_000)
        except Exception:
            pass
        self._wait_for_workbench_ready(session, timeout_ms=120_000)
        self._dismiss_startup_prompts(session.browser_page)
        self._open_active_file_in_browser(session.browser_page)
        self._hide_secondary_side_bar(session.browser_page)
        self._wait_for_workbench_ready(session, timeout_ms=30_000)

    def _dismiss_startup_prompts(self, page) -> None:
        for selector in (
            "text=Yes, I trust the authors",
            "text=Trust",
            "text=I understand",
        ):
            locator = page.locator(selector)
            try:
                if locator.count() > 0:
                    locator.first.click(timeout=1_000)
                    page.wait_for_timeout(500)
            except Exception:
                continue

    def _active_file_is_visible(self, page) -> bool:
        active_name = Path(self._active_file).name
        try:
            state = page.evaluate(
                """
                (activeName) => {
                  const tabs = Array.from(document.querySelectorAll('.tabs-container .tab, .tab'))
                    .map((tab) => [
                      tab.getAttribute('aria-label') || '',
                      tab.textContent || '',
                    ].join(' '));
                  const editorText = Array.from(document.querySelectorAll('.monaco-editor, .editor-instance'))
                    .map((node) => node.textContent || '')
                    .join('\\n');
                  return {
                    hasTab: tabs.some((text) => text.includes(activeName)),
                    hasEditor: Boolean(document.querySelector('.monaco-editor, .editor-instance')),
                    editorText,
                  };
                }
                """,
                active_name,
            )
        except Exception:
            return False
        if not isinstance(state, dict):
            return False
        return bool(state.get("hasTab") or (state.get("hasEditor") and active_name in str(state.get("editorText", ""))))

    def _open_active_file_in_browser(self, page) -> None:
        keyboard = getattr(page, "keyboard", None)
        if keyboard is None:
            return
        active_file = str(self._active_file)
        active_name = Path(active_file).name
        for _attempt in range(2):
            if self._active_file_is_visible(page):
                return
            try:
                page.locator("body").click(timeout=1_000)
            except Exception:
                pass
            try:
                keyboard.press("Control+P")
                page.wait_for_timeout(300)
                keyboard.type(active_file, delay=0)
                page.wait_for_timeout(300)
                keyboard.press("Enter")
                page.wait_for_timeout(1_000)
            except Exception:
                continue
            if self._active_file_is_visible(page):
                return
        try:
            page.locator(f'.tabs-container .tab[aria-label*="{active_name}"]').first.wait_for(
                state="visible",
                timeout=2_000,
            )
        except Exception:
            pass

    def _hide_secondary_side_bar(self, page) -> None:
        for selector in (
            '[aria-label*="Hide Secondary Side Bar"]',
            '[title*="Hide Secondary Side Bar"]',
            '.auxiliarybar .action-label.codicon-close',
            '.auxiliarybar .codicon-close',
        ):
            try:
                locator = page.locator(selector)
                if locator.count() > 0:
                    locator.first.click(timeout=1_000)
                    page.wait_for_timeout(300)
                    return
            except Exception:
                continue

    def _workbench_ready_state(self, page) -> dict:
        payload = page.evaluate(
            """
            () => {
              const body = document.body;
              const bodyText = (body?.innerText || '').trim();
              const workbench = document.querySelector('.monaco-workbench');
              const explorer = document.querySelector(
                '.explorer-folders-view, .part.sidebar, .pane-composite-part, .monaco-list-row'
              );
              const editor = document.querySelector(
                '.editor-group-container, .monaco-editor, .editor-instance'
              );
              const buttons = Array.from(document.querySelectorAll('button')).map((button) =>
                (button.innerText || button.textContent || '').trim()
              );
              return {
                ready: Boolean(workbench && (explorer || editor)),
                bodyText,
                title: document.title || '',
                href: location.href || '',
                elementCount: document.querySelectorAll('*').length,
                buttons,
              };
            }
            """
        )
        return payload if isinstance(payload, dict) else {}

    def _workbench_ready_script(self) -> str:
        return """
        () => {
          const workbench = document.querySelector('.monaco-workbench');
          const explorer = document.querySelector(
            '.explorer-folders-view, .part.sidebar, .pane-composite-part, .monaco-list-row'
          );
          const editor = document.querySelector(
            '.editor-group-container, .monaco-editor, .editor-instance'
          );
          return Boolean(workbench && (explorer || editor));
        }
        """

    def _wait_for_workbench_ready(self, session, *, timeout_ms: int) -> None:
        from asil.gui_agent.session import (
            GUISessionStartupError,
            _browser_page_failure_category,
            browser_page_snapshot,
            record_browser_snapshot,
        )

        page = session.browser_page
        if page is None:
            raise GUISessionStartupError("browser_crashed", "code-server browser page is not available.")

        deadline = time.monotonic() + max(timeout_ms, 1_000) / 1000
        last_state: dict = {}
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if hasattr(page, "is_closed") and page.is_closed():
                    raise GUISessionStartupError("browser_crashed", "code-server browser page closed.")
            except GUISessionStartupError:
                raise
            except Exception:
                pass

            failure_category = _browser_page_failure_category(page)
            if failure_category == "browser_crashed":
                raise GUISessionStartupError(
                    "browser_crashed",
                    "code-server page failed before it became ready.",
                )

            try:
                wait_for_function = getattr(page, "wait_for_function", None)
                if callable(wait_for_function):
                    wait_for_function(
                        self._workbench_ready_script(),
                        timeout=min(5_000, max(250, int((deadline - time.monotonic()) * 1000))),
                    )
                    return
                last_state = self._workbench_ready_state(page)
                if bool(last_state.get("ready")):
                    return
            except Exception as exc:
                last_error = exc

            try:
                self._dismiss_startup_prompts(page)
            except Exception:
                pass

            remaining_ms = max(100, int((deadline - time.monotonic()) * 1000))
            wait_ms = min(1_000, remaining_ms)
            try:
                page.wait_for_timeout(wait_ms)
            except Exception:
                time.sleep(wait_ms / 1000)

        failure_category = _browser_page_failure_category(page)
        diagnostics = getattr(session, "startup_diagnostics", None)
        record_browser_snapshot(diagnostics, page, label="code_server_timeout")
        if failure_category == "blank_shell":
            raise GUISessionStartupError(
                "blank_shell",
                "code-server page stayed blank before it became ready.",
            )
        if failure_category == "browser_crashed":
            raise GUISessionStartupError(
                "browser_crashed",
                "code-server page failed before it became ready.",
            )
        snapshot = browser_page_snapshot(page)
        details = {
            "href": last_state.get("href", snapshot.get("url", str(getattr(page, "url", "")))),
            "title": last_state.get("title", snapshot.get("title", "")),
            "body_len": len(str(last_state.get("bodyText", ""))) if last_state else snapshot.get("body_len"),
            "element_count": last_state.get("elementCount", snapshot.get("element_count")),
        }
        if last_error is not None:
            details["last_error"] = str(last_error)
        raise GUISessionStartupError(
            "window_timeout",
            f"code-server UI did not become ready within {timeout_ms}ms: {details}",
        )

    def _probe_ui_ready(self, session) -> None:
        self._wait_for_workbench_ready(session, timeout_ms=30_000)

    def _workspace_url(self) -> str:
        return (
            f"{self.base_url}/?folder={quote(str(self.workspace_path), safe='')}"
            f"&file={quote(str(self.workspace_path / self._active_file), safe='')}"
        )

    def describe_rendering(self) -> RenderArtifact:
        if self.base_url:
            return RenderArtifact(
                filename="",
                kind="web_page_capture",
                backend="playwright+chromium",
                actual_page=True,
                description="Live code-server webpage capture",
            )
        return RenderArtifact(
            filename="",
            kind="state_render",
            backend="wkhtmltoimage+html",
            actual_page=False,
            description="Synthetic code-server workspace state render",
        )

    def _workspace_html(self) -> str:
        tree_items = []
        for path in sorted(self.workspace_path.rglob("*")):
            if path.name.startswith("."):
                continue
            relative = path.relative_to(self.workspace_path).as_posix()
            prefix = "dir" if path.is_dir() else "file"
            tree_items.append(f"<li><strong>{prefix}</strong> {relative}</li>")

        editor_content = self._read_text(self._active_file)
        body = (
            "<h1>code-server Workspace</h1>"
            f"<p>Active file: <code>{self._active_file}</code></p>"
            "<div class='panel' style='display:grid;grid-template-columns:320px 1fr;gap:18px;padding:18px;'>"
            "<section><h2>Explorer</h2><ul>"
            + "".join(tree_items)
            + "</ul></section>"
            f"<section><h2>Editor</h2><pre>{editor_content}</pre></section>"
            "</div>"
        )
        return html_page("code-server Workspace", body)

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path) if output_path else self.workspace_path / "code-server.png"
        if self.base_url:
            active_path = Path(self._active_file)
            workspace_url = (
                f"{self.base_url}/?folder={quote(str(self.workspace_path), safe='')}"
                f"&file={quote(str(self.workspace_path / self._active_file), safe='')}"
            )
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    return capture_url_to_png(
                        workspace_url,
                        output,
                        backend="playwright",
                        initial_wait_ms=5_000,
                        timeout_ms=60_000,
                        optional_click_selectors=[
                            "text=Yes, I trust the authors",
                            "text=I understand",
                        ],
                        keyboard_steps=[
                            {"press": "Control+P"},
                            {"wait_ms": 500},
                            {"type": self._active_file},
                            {"wait_ms": 500},
                            {"press": "Enter"},
                            {"wait_ms": 1_000},
                        ],
                        wait_for_selectors_after_actions=[
                            ".monaco-workbench",
                            ".explorer-folders-view .monaco-list-row",
                            ".editor-group-container",
                            f'.tabs-container .tab[aria-label*="{active_path.name}"]',
                        ],
                    )
                except Exception as exc:
                    last_exc = exc
                    if attempt == 0:
                        time.sleep(10)
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
        return capture_html_to_png(self._workspace_html(), output)
