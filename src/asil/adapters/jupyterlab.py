"""ASIL adapter for JupyterLab with notebook-first workspace state."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
import uuid

import requests

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import RenderArtifact, capture_html_to_png, capture_url_to_png, html_page


def _notebook_cell(
    cell_type: str,
    source: str,
    *,
    output: str = "",
    execution_count: int | None = None,
) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = (
            [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output,
                }
            ]
            if output
            else []
        )
    return cell


def _empty_notebook() -> dict[str, Any]:
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "title": "analysis.ipynb",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _cell_source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _cell_output_text(cell: dict[str, Any]) -> str:
    outputs = cell.get("outputs", [])
    fragments: list[str] = []
    for output in outputs:
        if "text" in output:
            text = output["text"]
            fragments.append("".join(text) if isinstance(text, list) else str(text))
        elif "data" in output and "text/plain" in output["data"]:
            text = output["data"]["text/plain"]
            fragments.append("".join(text) if isinstance(text, list) else str(text))
    return "".join(fragments)


def _set_cell_output_text(cell: dict[str, Any], text: str, execution_count: int | None) -> None:
    cell["execution_count"] = execution_count
    if text:
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": text}]
    else:
        cell["outputs"] = []


_DEFAULT_FILES: dict[str, Any] = {
    "README.md": "# Analysis Workspace\n\n- Review the notebook\n- Update the summary\n- Share findings\n",
    "notebooks/analysis.ipynb": {
        **_empty_notebook(),
        "cells": [
            _notebook_cell("markdown", "# Weekly Analysis\nInspect the KPI snapshot before publishing."),
            _notebook_cell("code", "total = 6 * 7", output="pending\n", execution_count=1),
            _notebook_cell("markdown", "## Summary\nThe dashboard is ready for review."),
        ],
    },
    "notebooks/summary.md": "# Summary\n\nThe notebook output is ready to share.\n",
    "src/helpers.py": 'def format_total(total: int) -> str:\n    return f"Total={total}"\n',
    "data/metrics.csv": "metric,value\nrevenue,42\ncost,18\n",
}


class JupyterLabAdapter(ASILAdapter):
    app_name = "JupyterLab"
    supported_action_types = ["modify_file", "navigate"]

    def __init__(self, workspace_path: str | Path, base_url: str = "", active_file: str = "notebooks/analysis.ipynb") -> None:
        self.workspace_path = Path(workspace_path)
        self.base_url = base_url.rstrip("/")
        self._default_active_file = active_file
        self._lab_workspace_id = f"asil-{uuid.uuid4().hex[:12]}"
        self._open_tabs: list[str] = [active_file]
        self._active_file = active_file
        self._active_cell_index_by_notebook: dict[str, int] = {active_file: 0}
        self.clear_gui_shadow_state()

    def gui_eval_mode(self) -> str:
        return "live_shadow_required"

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: Path,
        sandbox=None,
        mock: bool = False,
    ) -> "JupyterLabAdapter":
        shared_root = os.environ.get("JUPYTERLAB_WORKSPACE_ROOT", "").strip()
        workspace = Path(shared_root) / "jupyterlab-workspace" if shared_root else tmp / "jupyterlab-workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        for relative_path, content in _DEFAULT_FILES.items():
            file_path = workspace / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.suffix == ".ipynb":
                file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
            else:
                file_path.write_text(str(content), encoding="utf-8")

        base_url = os.environ.get("JUPYTERLAB_URL", "").strip()
        return cls(workspace_path=workspace, base_url=base_url, active_file="notebooks/analysis.ipynb")

    @property
    def source_path(self) -> Path:
        return self.workspace_path

    def clone(self, new_path: Path) -> "JupyterLabAdapter":
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.copytree(self.workspace_path, new_path)
        cloned = JupyterLabAdapter(new_path, base_url=self.base_url, active_file=self._active_file)
        cloned._open_tabs = list(self._open_tabs)
        cloned._active_cell_index_by_notebook = dict(self._active_cell_index_by_notebook)
        return cloned

    def reset_state(self) -> None:
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        for relative_path, content in _DEFAULT_FILES.items():
            file_path = self.workspace_path / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.suffix == ".ipynb":
                file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
            else:
                file_path.write_text(str(content), encoding="utf-8")
        self._open_tabs = [self._default_active_file]
        self._active_file = self._default_active_file
        self._active_cell_index_by_notebook = {self._default_active_file: 0}
        self.clear_gui_shadow_state()

    def prepare_task(self, task: Any) -> None:
        """Reset and seed task-specific files referenced by generated tasks."""
        self.reset_state()
        self.setup_state(getattr(task, "initial_state", "default") or "default")
        notebook_min_cells: dict[str, int] = {}
        seed_by_path: dict[str, str] = {}
        for action in getattr(task, "actions", []) or []:
            for operation in (action.get("params") or {}).get("operations", []):
                if not isinstance(operation, dict):
                    continue
                path = self._normalize_path(str(operation.get("path") or ""))
                if path.endswith(".ipynb") and "cell_index" in operation:
                    notebook_min_cells[path] = max(
                        notebook_min_cells.get(path, 0),
                        int(operation.get("cell_index", 0) or 0) + 1,
                    )
                if path and operation.get("action") == "replace_text":
                    seed_by_path[path] = str(operation.get("old", ""))
        for action in getattr(task, "actions", []) or []:
            for operation in (action.get("params") or {}).get("operations", []):
                if not isinstance(operation, dict):
                    continue
                op_name = str(operation.get("action", ""))
                path = self._normalize_path(str(operation.get("path") or ""))
                if not path:
                    continue
                if op_name in {"create_file", "set_file_text"}:
                    continue
                file_path = self.workspace_path / path
                if file_path.exists():
                    continue
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if path.endswith(".ipynb"):
                    notebook = _empty_notebook()
                    cells = notebook.setdefault("cells", [])
                    min_cells = max(1, notebook_min_cells.get(path, 1))
                    while len(cells) < min_cells:
                        cells.append(_notebook_cell("code", ""))
                    self._write_notebook(path, notebook)
                    continue
                seed = seed_by_path.get(path, str(operation.get("old", "")) if op_name == "replace_text" else "# Task workspace seed\n")
                file_path.write_text(seed, encoding="utf-8")

    def setup_state(self, initial_state: str) -> None:
        state_to_file = {
            "default": "notebooks/analysis.ipynb",
            "notebook_focus": "notebooks/analysis.ipynb",
            "summary_focus": "notebooks/summary.md",
            "src_focus": "src/helpers.py",
            "readme_focus": "README.md",
        }
        selected = state_to_file.get(initial_state, self._default_active_file)
        self._set_active_file(selected)
        self.clear_gui_shadow_state()

    def get_context(self) -> dict[str, str]:
        return {
            "workspace_path": str(self.workspace_path),
            "active_file": self._active_file,
        }

    def get_gui_session_spec(self) -> GUISessionSpec | None:
        if not self.base_url:
            return None
        return GUISessionSpec(
            surface_type="browser",
            browser_url="about:blank",
            browser_navigation_mode="current_page",
            window_title_pattern=r".*",
            window_class_pattern=r"chromium|Chromium|chrome|Google-chrome",
            min_width=1000,
            min_height=700,
            startup_timeout_s=120.0,
            post_launch_delay_s=5.0,
            post_launch_callback=self._prime_browser_session,
            backend_ready_probe=self._probe_backend_ready,
            ui_ready_probe=self._probe_ui_ready,
        )

    def _filesystem_shadow(self) -> dict[str, Any]:
        visible_files = [
            path.relative_to(self.workspace_path).as_posix()
            for path in sorted(self.workspace_path.rglob("*"))
            if path.is_file() and not path.name.startswith(".")
        ]
        shadow: dict[str, Any] = {
            "open_tabs": list(self._open_tabs),
            "active_file": self._active_file,
            "text_files": {},
            "notebooks": {},
            "visible_files": visible_files,
        }
        if self._active_file and not self._active_file.endswith(".ipynb"):
            active_path = self.workspace_path / self._active_file
            if active_path.exists():
                shadow["text_files"][self._active_file] = active_path.read_text(encoding="utf-8")
        return shadow

    def _probe_backend_ready(self) -> None:
        from asil.gui_agent.session import GUISessionStartupError

        url = f"{self.base_url}/lab"
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException as exc:
            raise GUISessionStartupError("backend_unready", f"JupyterLab backend is not reachable at {url}.") from exc
        if response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"JupyterLab backend returned HTTP {response.status_code} for {url}.",
            )
        target_url = self._target_lab_url()
        try:
            target_response = requests.get(target_url, timeout=10)
        except requests.RequestException as exc:
            raise GUISessionStartupError(
                "backend_unready",
                f"JupyterLab target page is not reachable at {target_url}.",
            ) from exc
        if target_response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"JupyterLab target page returned HTTP {target_response.status_code} for {target_url}.",
            )

    def _prime_browser_session(self, session) -> None:
        body_timeout_ms = 60_000
        page = session.browser_page
        target_url = self._target_lab_url()
        if not str(page.url).startswith(target_url):
            from asil.gui_agent.session import navigate_browser_target

            navigate_browser_target(session, target_url, timeout_ms=60_000)
            page = session.browser_page
        page.wait_for_load_state("domcontentloaded", timeout=body_timeout_ms)
        self._wait_for_lab_ready(session, timeout_ms=120_000)
        self._dismiss_startup_overlays(page)
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        if self._active_file.endswith(".ipynb"):
            self._ensure_active_notebook_visible(page)
        else:
            self._ensure_active_text_file_visible(page)

    def _probe_ui_ready(self, session) -> None:
        from asil.gui_agent.session import _assert_browser_page_ready

        if not self._active_file.endswith(".ipynb"):
            return
        self._wait_for_lab_ready(session, timeout_ms=120_000)
        ready_script = (
            "() => !!document.querySelector('.jp-NotebookPanel:not(.lm-mod-hidden)')"
        )

        _assert_browser_page_ready(
            session,
            required_selectors=(),
            ready_script=ready_script,
            app_name="JupyterLab",
            timeout_ms=30_000,
        )

    def _lab_ready_script(self) -> str:
        return (
            "() => Boolean("
            "document.querySelector('.jp-LabShell') && "
            "document.querySelector('.jp-FileBrowser .jp-DirListing-content')"
            ")"
        )

    def _wait_for_lab_ready(self, session, *, timeout_ms: int) -> None:
        from asil.gui_agent.session import (
            GUISessionStartupError,
            _browser_page_failure_category,
            record_browser_snapshot,
        )

        page = session.browser_page
        if page is None:
            raise GUISessionStartupError("browser_crashed", "JupyterLab browser page is not available.")

        deadline = time.monotonic() + max(timeout_ms, 1_000) / 1000
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            failure_category = _browser_page_failure_category(page)
            if failure_category == "browser_crashed":
                raise GUISessionStartupError(
                    "browser_crashed",
                    "JupyterLab page failed before its shell became ready.",
                )
            try:
                page.wait_for_function(
                    self._lab_ready_script(),
                    timeout=min(5_000, max(250, int((deadline - time.monotonic()) * 1000))),
                )
                return
            except Exception as exc:
                last_error = exc
            self._dismiss_startup_overlays(page)
            remaining_ms = max(100, int((deadline - time.monotonic()) * 1000))
            wait_ms = min(1_000, remaining_ms)
            try:
                page.wait_for_timeout(wait_ms)
            except Exception:
                time.sleep(wait_ms / 1000)

        diagnostics = getattr(session, "startup_diagnostics", None)
        record_browser_snapshot(diagnostics, page, label="jupyterlab_timeout")
        failure_category = _browser_page_failure_category(page)
        if failure_category == "browser_crashed":
            raise GUISessionStartupError(
                "browser_crashed",
                "JupyterLab page failed before its shell became ready.",
            )
        if failure_category == "blank_shell":
            raise GUISessionStartupError(
                "blank_shell",
                "JupyterLab page stayed blank before its shell became ready.",
            )
        suffix = f" Last error: {last_error}" if last_error is not None else ""
        raise GUISessionStartupError(
            "window_timeout",
            f"JupyterLab shell did not become ready within {timeout_ms}ms.{suffix}",
        )

    def _target_lab_url(self) -> str:
        workspace_base = f"{self.base_url}/lab/workspaces/{quote(self._lab_workspace_id)}"
        if self._active_file.endswith(".ipynb"):
            return f"{workspace_base}/tree/{quote(self._active_file)}"
        return f"{workspace_base}/tree/{quote(self._active_file)}"

    def _open_target_file_in_browser(self, page) -> None:
        target_parts = list(Path(self._active_file).parts)
        if not target_parts:
            return

        if self._activate_dirlisting_item(page, target_parts[-1], open_item=True):
            return

        for folder in target_parts[:-1]:
            if not self._activate_dirlisting_item(page, folder, open_item=True):
                break
            self._dismiss_startup_overlays(page)
            time.sleep(0.2)

        self._activate_dirlisting_item(page, target_parts[-1], open_item=True)

    def _activate_dirlisting_item(self, page, label: str, *, open_item: bool = False) -> bool:
        try:
            file_item = page.locator(".jp-DirListing-item", has_text=label).first
            file_item.wait_for(state="visible", timeout=5_000)
        except Exception:
            return False

        self._dismiss_startup_overlays(page)
        if not open_item:
            try:
                file_item.click(timeout=1_000)
                return True
            except Exception:
                return False

        try:
            file_item.dblclick(timeout=10_000)
            return True
        except Exception:
            try:
                file_item.click(timeout=1_000)
                keyboard = getattr(page, "keyboard", None)
                if keyboard is not None:
                    keyboard.press("Enter")
                return True
            except Exception:
                return False

    def _ensure_active_notebook_visible(self, page) -> None:
        try:
            page.wait_for_selector(".jp-NotebookPanel:not(.lm-mod-hidden)", timeout=5_000)
            return
        except Exception:
            pass

        self._dismiss_startup_overlays(page)
        self._open_target_file_in_browser(page)
        try:
            page.wait_for_selector(".jp-NotebookPanel:not(.lm-mod-hidden)", timeout=15_000)
        except Exception:
            return

    def _ensure_active_text_file_visible(self, page) -> None:
        self._open_target_file_in_browser(page)
        try:
            page.wait_for_selector(".jp-FileEditor:not(.lm-mod-hidden)", timeout=10_000)
        except Exception:
            # Startup should remain gated on an interactive Lab shell; opening the
            # current text file is only a best-effort convenience for GUI sync.
            return

    def _dismiss_startup_overlays(self, page) -> None:
        for selector in (
            "button:has-text('No')",
            "button:has-text('Close')",
            "button:has-text('Dismiss')",
            "button[aria-label='Close']",
        ):
            try:
                page.locator(selector).first.click(timeout=1_000)
                time.sleep(0.2)
            except Exception:
                continue

    def _normalize_path(self, relative_path: str) -> str:
        return relative_path.strip("/").replace("\\", "/")

    def _set_active_file(self, relative_path: str) -> None:
        normalized = self._normalize_path(relative_path)
        file_path = self.workspace_path / normalized
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Workspace file does not exist: {normalized}")
        self._active_file = normalized
        if normalized not in self._open_tabs:
            self._open_tabs.append(normalized)
        if normalized.endswith(".ipynb"):
            self._active_cell_index_by_notebook.setdefault(normalized, 0)

    def _read_text(self, relative_path: str) -> str:
        return (self.workspace_path / relative_path).read_text(encoding="utf-8")

    def _write_text(self, relative_path: str, content: str) -> None:
        file_path = self.workspace_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    def _read_notebook(self, relative_path: str) -> dict[str, Any]:
        notebook = json.loads(self._read_text(relative_path))
        notebook.setdefault("nbformat", 4)
        notebook.setdefault("nbformat_minor", 5)
        notebook.setdefault("cells", [])
        notebook.setdefault("metadata", {})
        notebook["metadata"].setdefault(
            "kernelspec",
            {"display_name": "Python 3", "language": "python", "name": "python3"},
        )
        notebook["metadata"].setdefault("language_info", {"name": "python", "version": "3.11"})
        return notebook

    def _write_notebook(self, relative_path: str, notebook: dict[str, Any]) -> None:
        self._write_text(relative_path, json.dumps(notebook, indent=2))

    def _iter_workspace_elements(self) -> list[Element]:
        shadow = self._get_gui_shadow_state() or {}
        shadow_text_files = shadow.get("text_files", {})
        shadow_notebooks = shadow.get("notebooks", {})
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

            if path.suffix == ".ipynb":
                notebook_shadow = shadow_notebooks.get(relative)
                if isinstance(notebook_shadow, dict) and isinstance(notebook_shadow.get("cells"), list):
                    notebook = {
                        **self._read_notebook(relative),
                        "cells": [
                            {
                                "cell_type": str(cell.get("cell_type", "code")),
                                "source": str(cell.get("source", "")),
                                "outputs": (
                                    [{"output_type": "stream", "name": "stdout", "text": str(cell.get("output", ""))}]
                                    if str(cell.get("output", ""))
                                    else []
                                ),
                                "execution_count": None,
                                "metadata": {},
                            }
                            for cell in notebook_shadow.get("cells", [])
                        ],
                    }
                else:
                    notebook = self._read_notebook(relative)
                elements.append(
                    Element(
                        id=f"file:{relative}",
                        type="file",
                        label=path.name,
                        value={
                            "path": relative,
                            "extension": path.suffix,
                            "cell_count": len(notebook.get("cells", [])),
                            "kind": "notebook",
                        },
                        editable=True,
                        actions=["open", "set_cell_source", "set_cell_output", "insert_cell", "delete_cell", "rename", "delete"],
                    )
                )
                continue

            content = str(shadow_text_files.get(relative, path.read_text(encoding="utf-8")))
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
                    actions=["open", "replace_text", "append_text", "set_file_text", "rename", "delete"],
                )
            )
        return elements

    def observe(self) -> Observation:
        shadow = self._get_gui_shadow_state() or {}
        open_tabs = list(shadow.get("open_tabs", self._open_tabs))
        active_file = str(shadow.get("active_file", self._active_file))
        shadow_notebooks = shadow.get("notebooks", {})
        shadow_text_files = shadow.get("text_files", {})
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

        for tab_path in open_tabs:
            if not tab_path.endswith(".ipynb"):
                continue
            notebook_shadow = shadow_notebooks.get(tab_path)
            if isinstance(notebook_shadow, dict) and isinstance(notebook_shadow.get("cells"), list):
                notebook = {
                    "cells": [
                        {
                            "cell_type": str(cell.get("cell_type", "code")),
                            "source": str(cell.get("source", "")),
                            "outputs": (
                                [{"output_type": "stream", "name": "stdout", "text": str(cell.get("output", ""))}]
                                if str(cell.get("output", ""))
                                else []
                            ),
                            "execution_count": None,
                            "metadata": {},
                        }
                        for cell in notebook_shadow.get("cells", [])
                    ],
                    "metadata": {"kernelspec": {"name": "python3"}},
                }
                active_cell_index = int(notebook_shadow.get("active_cell_index", 0) or 0)
            else:
                notebook = self._read_notebook(tab_path)
                active_cell_index = self._active_cell_index_by_notebook.get(tab_path, 0)
            cell_ids: list[str] = []
            for index, cell in enumerate(notebook.get("cells", [])):
                cell_id = f"cell:{tab_path}:{index}"
                cell_ids.append(cell_id)
                elements.append(
                    Element(
                        id=cell_id,
                        type="cell",
                        label=f"Cell {index + 1}",
                        value={
                            "cell_type": cell.get("cell_type", "code"),
                            "source": _cell_source_text(cell),
                            "output": _cell_output_text(cell),
                        },
                        editable=True,
                        actions=["set_cell_source", "set_cell_output", "insert_cell", "delete_cell"],
                        metadata={
                            "execution_count": cell.get("execution_count"),
                            "active": tab_path == active_file and index == active_cell_index,
                            "tab_path": tab_path,
                        },
                    )
                )
            elements.append(
                Element(
                    id=f"notebook:{tab_path}",
                    type="notebook",
                    label=Path(tab_path).name,
                    value={
                        "path": tab_path,
                        "cell_count": len(notebook.get("cells", [])),
                        "active_cell_index": active_cell_index,
                        "kernel": notebook.get("metadata", {}).get("kernelspec", {}).get("name", "python3"),
                    },
                    editable=True,
                    actions=["set_cell_source", "set_cell_output", "insert_cell", "delete_cell"],
                    children=cell_ids,
                    metadata={"active": tab_path == active_file},
                )
            )

        current_view = "editor"
        if active_file.endswith(".ipynb"):
            current_view = "notebook"
        else:
            editor_content = str(shadow_text_files.get(active_file, self._read_text(active_file)))
            elements.append(
                Element(
                    id=f"editor:{active_file}",
                    type="editor",
                    label=active_file,
                    value={
                        "path": active_file,
                        "content": editor_content,
                        "line_count": len(editor_content.splitlines()),
                    },
                    editable=True,
                    actions=["replace_text", "append_text", "set_file_text"],
                )
            )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": current_view,
                "active_document": Path(active_file).name,
                "document_path": str(self.workspace_path / active_file),
            },
            navigation={
                "current_path": active_file,
                "breadcrumb": active_file.split("/"),
            },
            data_summary=(
                f"JupyterLab workspace with {len([e for e in elements if e.type == 'file'])} files; "
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
                  const textOf = (root) => {
                    if (!root) return '';
                    const lines = Array.from(root.querySelectorAll('.cm-line'));
                    if (lines.length) {
                      return lines.map((line) => line.textContent || '').join('\\n');
                    }
                    return (root.textContent || '').trim();
                  };
                  const tabs = Array.from(document.querySelectorAll('.lm-TabBar-tab')).map((tab) => ({
                    label: (tab.textContent || '').trim(),
                    active: tab.classList.contains('lm-mod-current') || tab.getAttribute('aria-selected') === 'true',
                  })).filter((tab) => tab.label);
                  const activeNotebook = document.querySelector('.jp-NotebookPanel:not(.lm-mod-hidden)');
                  const notebookCells = activeNotebook ? Array.from(activeNotebook.querySelectorAll('.jp-Cell')).map((cell) => {
                    const cellType = cell.classList.contains('jp-MarkdownCell') ? 'markdown' : 'code';
                    const sourceRoot = cell.querySelector('.cm-content') || cell.querySelector('.jp-InputArea-editor');
                    const outputRoot = cell.querySelector('.jp-OutputArea-output') || cell.querySelector('.jp-RenderedText');
                    return {
                      cell_type: cellType,
                      source: textOf(sourceRoot),
                      output: textOf(outputRoot),
                      active: cell.classList.contains('jp-mod-active'),
                    };
                  }) : [];
                  const activeEditor = document.querySelector('.jp-FileEditor:not(.lm-mod-hidden), .jp-FileEditor.jp-mod-current');
                  const fileItems = Array.from(document.querySelectorAll('.jp-DirListing-item')).map((item) => (item.textContent || '').trim()).filter(Boolean);
                  return {
                    tabs,
                    file_items: fileItems,
                    text_editor_content: textOf(activeEditor?.querySelector('.cm-content') || activeEditor),
                    notebook_cells: notebookCells,
                  };
                }
                """
            )
        except Exception:
            self._set_gui_shadow_state(self._filesystem_shadow())
            return
        if not isinstance(payload, dict):
            return
        open_tabs = self._shadow_tabs_from_labels(payload.get("tabs", []))
        active_file = next((tab for tab in open_tabs if tab and any(
            item.get("active") for item in payload.get("tabs", []) if self._resolve_visible_path(item.get("label", "")) == tab
        )), None)
        active_file = active_file or self._active_file
        text_editor_content = payload.get("text_editor_content")
        if not self._active_file.endswith(".ipynb") and isinstance(text_editor_content, str) and text_editor_content:
            active_file = self._active_file
            if active_file not in open_tabs:
                open_tabs.append(active_file)
        shadow: dict[str, Any] = {
            "open_tabs": open_tabs or list(self._open_tabs),
            "active_file": active_file,
            "text_files": {},
            "notebooks": {},
            "visible_files": [str(item) for item in payload.get("file_items", []) if isinstance(item, str)],
        }
        if active_file.endswith(".ipynb") and isinstance(payload.get("notebook_cells"), list):
            cells = []
            active_cell_index = 0
            for index, cell in enumerate(payload.get("notebook_cells", [])):
                if not isinstance(cell, dict):
                    continue
                if cell.get("active"):
                    active_cell_index = index
                cells.append(
                    {
                        "cell_type": str(cell.get("cell_type", "code")),
                        "source": str(cell.get("source", "")),
                        "output": str(cell.get("output", "")),
                    }
                )
            shadow["notebooks"][active_file] = {
                "cells": cells,
                "active_cell_index": active_cell_index,
            }
        elif isinstance(text_editor_content, str) and text_editor_content:
            shadow["text_files"][active_file] = str(text_editor_content)
        self._set_gui_shadow_state(shadow)

    def _shadow_tabs_from_labels(self, raw_tabs: list[Any]) -> list[str]:
        resolved: list[str] = []
        for item in raw_tabs:
            label = item.get("label", "") if isinstance(item, dict) else str(item)
            candidate = self._resolve_visible_path(label)
            if candidate and candidate not in resolved:
                resolved.append(candidate)
        return resolved

    def _resolve_visible_path(self, label: str) -> str:
        normalized = label.strip()
        if not normalized:
            return ""
        candidates = []
        for path in sorted(self.workspace_path.rglob("*")):
            if path.is_file() and path.name == normalized:
                candidates.append(path.relative_to(self.workspace_path).as_posix())
        if len(candidates) == 1:
            return candidates[0]
        if normalized in self._open_tabs:
            return normalized
        matching_open_tab = next((tab for tab in self._open_tabs if Path(tab).name == normalized), "")
        if matching_open_tab:
            return matching_open_tab
        return normalized if (self.workspace_path / normalized).exists() else ""

    def _ensure_notebook_cell(self, relative_path: str, cell_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        notebook = self._read_notebook(relative_path)
        cells = notebook.setdefault("cells", [])
        if not (0 <= cell_index < len(cells)):
            raise IndexError(f"Notebook cell index out of range: {cell_index}")
        return notebook, cells

    def execute(self, action: Action) -> Observation:
        operations = action.params.get("operations", [])
        if action.action_type == "navigate" and action.target:
            operations = [{"action": "open_file", "path": action.target}]

        for operation in operations:
            op_name = operation.get("action", "")
            relative_path = self._normalize_path(str(operation.get("path", self._active_file)))

            if op_name == "open_file":
                self._set_active_file(relative_path)
            elif op_name == "create_file":
                content = operation.get("content", "")
                if relative_path.endswith(".ipynb"):
                    if (self.workspace_path / relative_path).exists() and not isinstance(content, dict):
                        continue
                    notebook = content if isinstance(content, dict) else _empty_notebook()
                    self._write_notebook(relative_path, notebook)
                else:
                    self._write_text(relative_path, str(content))
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
            elif op_name == "set_cell_source":
                cell_index = int(operation["cell_index"])
                notebook, cells = self._ensure_notebook_cell(relative_path, cell_index)
                cells[cell_index]["source"] = str(operation.get("source", ""))
                self._active_cell_index_by_notebook[relative_path] = cell_index
                self._write_notebook(relative_path, notebook)
            elif op_name == "set_cell_output":
                cell_index = int(operation["cell_index"])
                notebook, cells = self._ensure_notebook_cell(relative_path, cell_index)
                _set_cell_output_text(
                    cells[cell_index],
                    str(operation.get("output", "")),
                    operation.get("execution_count"),
                )
                self._active_cell_index_by_notebook[relative_path] = cell_index
                self._write_notebook(relative_path, notebook)
            elif op_name == "insert_cell":
                notebook = self._read_notebook(relative_path)
                cells = notebook.setdefault("cells", [])
                position = int(operation.get("position", len(cells)))
                if position < 0:
                    position = 0
                if position > len(cells):
                    position = len(cells)
                cells.insert(
                    position,
                    _notebook_cell(
                        str(operation.get("cell_type", "code")),
                        str(operation.get("source", "")),
                        output=str(operation.get("output", "")),
                        execution_count=operation.get("execution_count"),
                    ),
                )
                self._active_cell_index_by_notebook[relative_path] = position
                self._write_notebook(relative_path, notebook)
            elif op_name == "delete_cell":
                cell_index = int(operation["cell_index"])
                notebook, cells = self._ensure_notebook_cell(relative_path, cell_index)
                cells.pop(cell_index)
                if not cells:
                    cells.append(_notebook_cell("code", ""))
                self._active_cell_index_by_notebook[relative_path] = min(cell_index, len(cells) - 1)
                self._write_notebook(relative_path, notebook)
            elif op_name == "rename_path":
                new_path = self._normalize_path(str(operation["new_path"]))
                src = self.workspace_path / relative_path
                dst = self.workspace_path / new_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                self._open_tabs = [new_path if tab == relative_path else tab for tab in self._open_tabs]
                if relative_path in self._active_cell_index_by_notebook:
                    self._active_cell_index_by_notebook[new_path] = self._active_cell_index_by_notebook.pop(relative_path)
                if self._active_file == relative_path:
                    self._active_file = new_path
            elif op_name == "delete_path":
                target = self.workspace_path / relative_path
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                self._open_tabs = [tab for tab in self._open_tabs if tab != relative_path]
                self._active_cell_index_by_notebook.pop(relative_path, None)
                if self._active_file == relative_path:
                    fallback = self._open_tabs[-1] if self._open_tabs else self._default_active_file
                    self._set_active_file(fallback)
            else:
                raise ValueError(f"Unsupported JupyterLab operation: {op_name}")

        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        if self.base_url:
            return RenderArtifact(
                filename="",
                kind="web_page_capture",
                backend="playwright+chromium",
                actual_page=True,
                description="Live JupyterLab webpage capture",
            )
        return RenderArtifact(
            filename="",
            kind="state_render",
            backend="wkhtmltoimage+html",
            actual_page=False,
            description="Synthetic JupyterLab workspace state render",
        )

    def _workspace_html(self) -> str:
        tree_items = []
        for path in sorted(self.workspace_path.rglob("*")):
            if path.name.startswith("."):
                continue
            relative = path.relative_to(self.workspace_path).as_posix()
            prefix = "dir" if path.is_dir() else "file"
            tree_items.append(f"<li><strong>{prefix}</strong> {relative}</li>")

        if self._active_file.endswith(".ipynb"):
            notebook = self._read_notebook(self._active_file)
            preview = "".join(
                f"<article class='panel' style='padding:12px;margin-bottom:12px;'><strong>{cell.get('cell_type', 'code')}</strong>"
                f"<pre>{_cell_source_text(cell)}</pre><div><em>Output:</em> {_cell_output_text(cell)}</div></article>"
                for cell in notebook.get("cells", [])
            )
            detail = f"<section><h2>Notebook preview</h2>{preview}</section>"
        else:
            editor_content = self._read_text(self._active_file)
            detail = f"<section><h2>Editor</h2><pre>{editor_content}</pre></section>"

        body = (
            "<h1>JupyterLab Workspace</h1>"
            f"<p>Active file: <code>{self._active_file}</code></p>"
            "<div class='panel' style='display:grid;grid-template-columns:320px 1fr;gap:18px;padding:18px;'>"
            "<section><h2>File Browser</h2><ul>"
            + "".join(tree_items)
            + "</ul></section>"
            + detail
            + "</div>"
        )
        return html_page("JupyterLab Workspace", body)

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path) if output_path else self.workspace_path / "jupyterlab.png"
        if self.base_url:
            lab_url = self._target_lab_url()
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    return capture_url_to_png(
                        lab_url,
                        output,
                        backend="playwright",
                        full_page=False,
                        initial_wait_ms=20_000,
                        timeout_ms=90_000,
                        wait_for_selectors=[
                            ".jp-LabShell",
                            ".jp-FileBrowser .jp-DirListing-content",
                        ],
                        ready_script=(
                            "() => !!document.querySelector("
                            "'.jp-NotebookPanel:not(.lm-mod-hidden), "
                            ".jp-FileEditor:not(.lm-mod-hidden), "
                            ".jp-MainAreaWidget:not(.lm-mod-hidden)'"
                            ")"
                        ),
                    )
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(8)
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
        return capture_html_to_png(self._workspace_html(), output)
