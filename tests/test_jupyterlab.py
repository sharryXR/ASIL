import json
import os
from pathlib import Path

from asil.eval.evaluator import evaluate_observation
from asil.gui_agent.session import GUISessionStartupError
from asil.protocol import Action


class _FakePage:
    def __init__(self, url: str) -> None:
        self.url = url
        self.goto_calls: list[str] = []
        self.wait_for_selector_calls: list[str] = []
        self.wait_for_load_state_calls: list[str] = []
        self.wait_for_function_calls: list[tuple[str, int]] = []
        self.wait_for_timeout_calls: list[int] = []
        self.dir_listing_opens: list[str] = []
        self.keyboard_presses: list[str] = []
        self.visible_file_items: set[str] = {"summary.md", "helpers.py"}
        self.keyboard = type(
            "Keyboard",
            (),
            {"press": lambda inner_self, key: self.keyboard_presses.append(key)},
        )()
        self.selector_counts: dict[str, int] = {
            ".jp-LabShell": 1,
            ".jp-FileBrowser .jp-DirListing-content": 1,
            "body": 1,
        }

    def content(self) -> str:
        return "<html><body>JupyterLab ready</body></html>"

    def wait_for_load_state(self, state: str, timeout=None) -> None:
        self.wait_for_load_state_calls.append(state)

    def wait_for_selector(self, selector: str, timeout=None) -> None:
        self.wait_for_selector_calls.append(selector)

    def wait_for_function(self, script: str, timeout: int = 0) -> None:
        self.wait_for_function_calls.append((script, timeout))
        return None

    def wait_for_timeout(self, timeout_ms: int) -> None:
        self.wait_for_timeout_calls.append(timeout_ms)

    def goto(self, url: str, wait_until=None, timeout=None) -> None:
        self.goto_calls.append(url)
        self.url = url

    def locator(self, selector: str, has_text: str | None = None):
        owner = self

        class Locator:
            @property
            def first(self):
                return self

            def count(self) -> int:
                return owner.selector_counts.get(selector, 0)

            def inner_text(self, timeout: int = 0) -> str:
                return "JupyterLab ready"

            def wait_for(self, state: str = "visible", timeout: int = 0) -> None:
                del state, timeout
                if has_text and has_text not in owner.visible_file_items:
                    raise AssertionError(f"Missing file item: {has_text}")

            def dblclick(self, timeout: int = 0) -> None:
                del timeout
                if has_text:
                    owner.dir_listing_opens.append(has_text)

            def click(self, timeout: int = 0) -> None:
                del timeout
                if has_text:
                    owner.dir_listing_opens.append(f"click:{has_text}")

        return Locator()


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "jupyterlab"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def test_observe_returns_file_browser_tabs_notebook_and_editor(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)

    obs = adapter.observe()

    assert obs.meta.app_name == "JupyterLab"
    assert obs.meta.observation_source == "file_parse"
    elements = {element.id: element for element in obs.interactive_elements}
    assert "folder:notebooks" in elements
    assert "file:notebooks/analysis.ipynb" in elements
    assert "file:src/helpers.py" in elements
    assert "tab:notebooks/analysis.ipynb" in elements
    assert "notebook:notebooks/analysis.ipynb" in elements
    assert "cell:notebooks/analysis.ipynb:0" in elements
    assert "editor:notebooks/summary.md" not in elements
    assert obs.app_state.active_document == "analysis.ipynb"


def test_seeded_notebook_is_valid_nbformat_document(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    notebook = json.loads((adapter.workspace_path / "notebooks" / "analysis.ipynb").read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "source" in notebook["cells"][0]
    assert "outputs" in notebook["cells"][1]


def test_execute_updates_notebook_outputs_and_creates_files(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    action = Action(
        action_type="modify_file",
        target="workspace",
        params={
            "operations": [
                {"action": "open_file", "path": "notebooks/analysis.ipynb"},
                {"action": "set_cell_source", "path": "notebooks/analysis.ipynb", "cell_index": 1, "source": "total = 21 * 2"},
                {
                    "action": "set_cell_output",
                    "path": "notebooks/analysis.ipynb",
                    "cell_index": 1,
                    "output": "42",
                    "execution_count": 5,
                },
                {"action": "create_file", "path": "notes/findings.md", "content": "# Findings\n- Result is 42\n"},
                {"action": "open_file", "path": "notes/findings.md"},
                {"action": "append_text", "path": "notes/findings.md", "text": "- Ready to share\n"},
            ]
        },
    )

    obs = adapter.execute(action)

    elements = {element.id: element for element in obs.interactive_elements}
    assert "file:notes/findings.md" in elements
    assert "editor:notes/findings.md" in elements
    assert "Ready to share" in elements["editor:notes/findings.md"].value["content"]
    notebook = elements["notebook:notebooks/analysis.ipynb"]
    assert notebook.value["cell_count"] >= 2
    assert notebook.value["active_cell_index"] == 1
    assert elements["cell:notebooks/analysis.ipynb:1"].value["output"] == "42"
    assert elements["cell:notebooks/analysis.ipynb:1"].metadata["execution_count"] == 5


def test_execute_supports_notebook_structure_changes(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    action = Action(
        action_type="modify_file",
        target="workspace",
        params={
            "operations": [
                {"action": "open_file", "path": "notebooks/analysis.ipynb"},
                {"action": "insert_cell", "path": "notebooks/analysis.ipynb", "cell_type": "markdown", "source": "## Next Steps", "position": 2},
                {"action": "delete_cell", "path": "notebooks/analysis.ipynb", "cell_index": 0},
                {"action": "rename_path", "path": "notebooks/summary.md", "new_path": "notebooks/notes.md"},
                {"action": "open_file", "path": "notebooks/notes.md"},
            ]
        },
    )

    obs = adapter.execute(action)
    elements = {element.id: element for element in obs.interactive_elements}

    assert "file:notebooks/notes.md" in elements
    assert "file:notebooks/summary.md" not in elements
    assert "editor:notebooks/notes.md" in elements
    assert elements["cell:notebooks/analysis.ipynb:1"].value["source"] == "## Next Steps"


def test_rendering_prefers_real_webpage_capture_when_base_url_is_available(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    captured = {}

    def fake_capture(url, output_path, **kwargs):
        captured["url"] = url
        captured["output_path"] = Path(output_path)
        captured.update(kwargs)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.jupyterlab.capture_url_to_png", fake_capture)

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "jupyterlab.png")

    assert artifact.actual_page is True
    assert artifact.kind == "web_page_capture"
    assert artifact.backend == "playwright+chromium"
    assert captured["url"] == adapter._target_lab_url()
    assert captured["wait_for_selectors"] == [
        ".jp-LabShell",
        ".jp-FileBrowser .jp-DirListing-content",
    ]
    assert "jp-NotebookPanel:not(.lm-mod-hidden)" in captured["ready_script"]
    assert "jp-FileEditor:not(.lm-mod-hidden)" in captured["ready_script"]
    assert captured["backend"] == "playwright"
    assert captured["full_page"] is False
    assert out == tmp_path / "jupyterlab.png"


def test_rendering_retries_live_webpage_capture_after_transient_crash(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    attempts = {"count": 0}

    def flaky_capture(url, output_path, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Page.wait_for_selector: Target crashed")
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.jupyterlab.capture_url_to_png", flaky_capture)
    monkeypatch.setattr("asil.adapters.jupyterlab.time.sleep", lambda _seconds: None)

    out = adapter.render_to_png(tmp_path / "jupyterlab.png")

    assert attempts["count"] == 2
    assert out == tmp_path / "jupyterlab.png"


def test_from_evaluation_context_prefers_shared_workspace_root_when_configured(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    shared_root = tmp_path / "shared-workspaces"
    monkeypatch.setenv("JUPYTERLAB_WORKSPACE_ROOT", str(shared_root))

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path / "ignored", mock=True)

    assert adapter.workspace_path == shared_root / "jupyterlab-workspace"
    assert (adapter.workspace_path / "notebooks" / "analysis.ipynb").exists()


def test_rendering_fallback_is_honest_when_no_live_page_is_configured(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = ""

    captured = {}

    def fake_capture(html_content, output_path, **kwargs):
        captured["html_content"] = html_content
        captured["output_path"] = Path(output_path)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.jupyterlab.capture_html_to_png", fake_capture)

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "jupyterlab-fallback.png")

    assert artifact.actual_page is False
    assert artifact.kind == "state_render"
    assert "JupyterLab Workspace" in captured["html_content"]
    assert "Notebook preview" in captured["html_content"]
    assert out == tmp_path / "jupyterlab-fallback.png"


def test_observe_prefers_live_shadow_for_text_editor_and_notebook_cells(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter._set_gui_shadow_state(
        {
            "open_tabs": ["notebooks/analysis.ipynb", "notebooks/summary.md"],
            "active_file": "notebooks/summary.md",
            "text_files": {
                "notebooks/summary.md": "# Summary\n\nUpdated in GUI only.\n",
            },
            "notebooks": {
                "notebooks/analysis.ipynb": {
                    "active_cell_index": 1,
                    "cells": [
                        {"cell_type": "markdown", "source": "# Weekly Analysis", "output": ""},
                        {"cell_type": "code", "source": "total = 84", "output": "84"},
                    ],
                }
            },
        }
    )

    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert elements["editor:notebooks/summary.md"].value["content"] == "# Summary\n\nUpdated in GUI only.\n"
    assert elements["cell:notebooks/analysis.ipynb:1"].value["source"] == "total = 84"
    assert elements["cell:notebooks/analysis.ipynb:1"].value["output"] == "84"
    assert elements["tab:notebooks/summary.md"].value["active"] is True
    assert obs.app_state.active_document == "summary.md"


def test_sync_from_gui_prefers_canonical_active_text_file_when_editor_is_visible(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.setup_state("src_focus")

    class Page:
        def evaluate(self, _script):
            return {
                "tabs": [
                    {"label": "analysis.ipynb", "active": True},
                ],
                "file_items": ["helpers.py"],
                "text_editor_content": 'def format_total(total: int) -> str:\n    return f"Total: {total}"\n',
                "notebook_cells": [],
            }

    session = type("Session", (), {"browser_page": Page()})()
    adapter.sync_from_gui(session)

    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert elements["editor:src/helpers.py"].value["content"].endswith('return f"Total: {total}"\n')
    assert elements["tab:src/helpers.py"].value["active"] is True
    assert obs.app_state.active_document == "helpers.py"


def test_validate_action_accepts_modify_file_and_navigate(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)

    assert adapter.validate_action(Action(action_type="modify_file", target="workspace", params={}))
    assert adapter.validate_action(Action(action_type="navigate", target="notebooks/analysis.ipynb", params={}))
    assert not adapter.validate_action(Action(action_type="api_call", target="jupyterlab", params={}))


def test_gui_session_spec_exposes_explicit_browser_readiness_probes(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "browser"
    assert spec.browser_url == "about:blank"
    assert spec.startup_timeout_s == 120.0
    assert spec.post_launch_callback is not None
    assert spec.backend_ready_probe is not None
    assert spec.ui_ready_probe is not None


def test_prime_browser_session_navigates_from_lab_shell_to_target_notebook(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    class LauncherPage(_FakePage):
        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.wait_for_selector_calls.append(selector)
            if selector == ".jp-NotebookPanel:not(.lm-mod-hidden)":
                raise AssertionError("notebook panel not visible yet")

    page = LauncherPage("http://127.0.0.1:8888/lab")
    page.visible_file_items.add("analysis.ipynb")
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert page.wait_for_selector_calls.count("body") == 0
    assert page.wait_for_function_calls
    assert ".jp-LabShell" in page.wait_for_function_calls[0][0]
    assert page.goto_calls == [adapter._target_lab_url()]
    assert "analysis.ipynb" in page.dir_listing_opens


def test_text_initial_states_navigate_to_target_file(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    adapter.setup_state("summary_focus")
    assert adapter._target_lab_url().endswith(
        "/lab/workspaces/" + adapter._lab_workspace_id + "/tree/notebooks/summary.md"
    )

    adapter.setup_state("src_focus")
    assert adapter._target_lab_url().endswith(
        "/lab/workspaces/" + adapter._lab_workspace_id + "/tree/src/helpers.py"
    )


def test_prime_browser_session_uses_bounded_lab_ready_polling(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    page = _FakePage("http://127.0.0.1:8888/lab")
    session = type("Session", (), {"browser_page": page})()

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", lambda *args, **kwargs: None)

    adapter._prime_browser_session(session)

    assert ".jp-LabShell" not in page.wait_for_selector_calls
    assert page.wait_for_function_calls
    script, timeout = page.wait_for_function_calls[0]
    assert ".jp-LabShell" in script
    assert ".jp-FileBrowser .jp-DirListing-content" in script
    assert timeout <= 5_000


def test_prime_browser_session_best_effort_opens_text_file_for_gui_visibility(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    adapter.setup_state("summary_focus")
    page = _FakePage("http://127.0.0.1:8888/lab/tree/notebooks")
    session = type("Session", (), {"browser_page": page})()

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", lambda *args, **kwargs: None)

    adapter._prime_browser_session(session)

    assert "summary.md" in page.dir_listing_opens


def test_prime_browser_session_best_effort_opens_notebook_when_launcher_is_visible(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    class LauncherPage(_FakePage):
        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.wait_for_selector_calls.append(selector)
            if selector == ".jp-NotebookPanel:not(.lm-mod-hidden)":
                raise AssertionError("notebook panel not visible yet")

    page = LauncherPage("http://127.0.0.1:8888/lab")
    page.visible_file_items.add("analysis.ipynb")
    session = type("Session", (), {"browser_page": page})()

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", lambda *args, **kwargs: None)

    adapter._prime_browser_session(session)

    assert "analysis.ipynb" in page.dir_listing_opens


def test_prime_browser_session_traverses_file_browser_folders_for_notebook(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    class NestedLauncherPage(_FakePage):
        def __init__(self, url: str) -> None:
            super().__init__(url)
            self.visible_file_items = {"notebooks"}

        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.wait_for_selector_calls.append(selector)
            if selector == ".jp-NotebookPanel:not(.lm-mod-hidden)":
                raise AssertionError("notebook panel not visible yet")

        def locator(self, selector: str, has_text: str | None = None):
            owner = self
            parent = super().locator(selector, has_text=has_text)

            class Locator(parent.__class__):
                def dblclick(self, timeout: int = 0) -> None:
                    del timeout
                    if has_text:
                        owner.dir_listing_opens.append(has_text)
                        if has_text == "notebooks":
                            owner.visible_file_items.add("analysis.ipynb")

            return Locator()

    page = NestedLauncherPage("http://127.0.0.1:8888/lab")
    session = type("Session", (), {"browser_page": page})()

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", lambda *args, **kwargs: None)

    adapter._prime_browser_session(session)

    assert page.dir_listing_opens[:2] == ["notebooks", "analysis.ipynb"]


def test_prime_browser_session_does_not_fail_if_text_file_opening_remains_unavailable(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    adapter.setup_state("summary_focus")
    page = _FakePage("http://127.0.0.1:8888/lab/tree/notebooks")
    page.visible_file_items.clear()
    session = type("Session", (), {"browser_page": page})()

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", lambda *args, **kwargs: None)

    adapter._prime_browser_session(session)

    assert page.dir_listing_opens in ([], ["click:summary.md"])


def test_prime_browser_session_raises_on_browser_crash(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    class CrashPage:
        url = adapter._target_lab_url()

        def content(self) -> str:
            return "<html><body>Aw, Snap!</body></html>"

        def wait_for_load_state(self, state: str, timeout=None) -> None:
            return None

        def wait_for_selector(self, selector: str, timeout=None) -> None:
            raise AssertionError("crashed page never exposes selectors")

        def locator(self, selector: str):
            class Locator:
                def count(self) -> int:
                    return 1 if selector == "body" else 0

                def inner_text(self, timeout: int = 0) -> str:
                    return "Aw, Snap!"

            return Locator()

    session = type("Session", (), {"browser_page": CrashPage()})()

    try:
        adapter._prime_browser_session(session)
        assert False, "Expected browser crash to abort JupyterLab startup"
    except GUISessionStartupError as exc:
        assert exc.category == "browser_crashed"


def test_prime_browser_session_allows_short_blank_warmup_before_shell_appears(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"

    class WarmupPage(_FakePage):
        def __init__(self, url: str) -> None:
            super().__init__(url)
            self.selector_counts[".jp-LabShell"] = 0
            self.selector_counts[".jp-FileBrowser .jp-DirListing-content"] = 0
            self._warm = True

        def content(self) -> str:
            if self._warm:
                return "<html><body></body></html>"
            return "<html><body>JupyterLab ready</body></html>"

        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.wait_for_selector_calls.append(selector)
            if selector == ".jp-LabShell":
                self._warm = False
                self.selector_counts[".jp-LabShell"] = 1
                self.selector_counts[".jp-FileBrowser .jp-DirListing-content"] = 1

        def wait_for_function(self, script: str, timeout: int = 0) -> None:
            self.wait_for_function_calls.append((script, timeout))
            if self._warm:
                self._warm = False
                raise TimeoutError("Lab shell is still warming up")

    page = WarmupPage("http://127.0.0.1:8888/lab")
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert len(page.wait_for_function_calls) >= 2
    assert page.wait_for_timeout_calls


def test_wait_for_lab_ready_records_diagnostics_on_blank_timeout(tmp_path: Path, monkeypatch):
    from asil.adapters import jupyterlab as jupyterlab_module
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)

    class BlankPage:
        url = "http://127.0.0.1:8888/lab"

        def __init__(self) -> None:
            self.wait_for_function_calls: list[int] = []

        def content(self) -> str:
            return "<html><body></body></html>"

        def wait_for_function(self, script: str, timeout: int = 0) -> None:
            del script
            self.wait_for_function_calls.append(timeout)
            raise TimeoutError("still blank")

        def wait_for_timeout(self, timeout_ms: int) -> None:
            del timeout_ms

        def locator(self, selector: str):
            class Locator:
                def count(self) -> int:
                    return 1 if selector == "body" else 0

                def inner_text(self, timeout: int = 0) -> str:
                    del timeout
                    return ""

            return Locator()

    ticks = iter([0.0, 0.1, 0.2, 1.2])
    monkeypatch.setattr(jupyterlab_module.time, "monotonic", lambda: next(ticks, 1.3))
    session = type(
        "Session",
        (),
        {
            "browser_page": BlankPage(),
            "startup_diagnostics": {"phases": [], "snapshots": []},
        },
    )()

    try:
        adapter._wait_for_lab_ready(session, timeout_ms=1)
        assert False, "Expected blank shell timeout"
    except GUISessionStartupError as exc:
        assert exc.category == "blank_shell"

    assert session.browser_page.wait_for_function_calls
    assert session.browser_page.wait_for_function_calls[0] <= 5_000
    assert session.startup_diagnostics["snapshots"][-1]["label"] == "jupyterlab_timeout"


def test_probe_ui_ready_accepts_lab_shell_for_text_focus(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    adapter.setup_state("src_focus")

    captured: dict[str, object] = {}

    def fail_assert(*_args, **_kwargs):
        raise AssertionError("Text-file readiness should not require a visible editor widget")

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", fail_assert)
    monkeypatch.setattr(
        adapter,
        "_wait_for_lab_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Text-file ui_ready_probe should rely on post-launch Lab readiness")
        ),
    )

    adapter._probe_ui_ready(type("Session", (), {})())


def test_probe_ui_ready_requires_visible_notebook_panel_for_notebook_focus(tmp_path: Path, monkeypatch):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    adapter = JupyterLabAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8888"
    adapter.setup_state("notebook_focus")

    captured: dict[str, object] = {}

    def fake_assert(session, *, required_selectors=(), ready_script=None, app_name, timeout_ms=45_000):
        del session, app_name, timeout_ms
        captured["required_selectors"] = required_selectors
        captured["ready_script"] = ready_script

    monkeypatch.setattr("asil.gui_agent.session._assert_browser_page_ready", fake_assert)
    monkeypatch.setattr(adapter, "_wait_for_lab_ready", lambda session, *, timeout_ms: captured.setdefault("lab_timeout", timeout_ms))

    adapter._probe_ui_ready(type("Session", (), {})())

    assert captured["lab_timeout"] == 120_000
    assert captured["required_selectors"] == ()
    assert ".jp-NotebookPanel:not(.lm-mod-hidden)" in str(captured["ready_script"])
    assert ".jp-MainAreaWidget:not(.lm-mod-hidden)" not in str(captured["ready_script"])


def test_jupyterlab_example_set_contains_20_tasks():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "jupyterlab"
    tasks = sorted(path for path in root.glob("jupyterlab_*.json") if path.stem.removeprefix("jupyterlab_").isdigit())

    assert len(tasks) == 20
    ids = [json.loads(task.read_text(encoding="utf-8"))["id"] for task in tasks]
    assert ids == [f"jupyterlab_{idx:02d}" for idx in range(1, 21)]


def test_representative_jupyterlab_tasks_evaluate_successfully(tmp_path: Path):
    from asil.adapters.jupyterlab import JupyterLabAdapter

    for task_id in ("jupyterlab_01", "jupyterlab_10", "jupyterlab_20"):
        task = _task(task_id)
        adapter = JupyterLabAdapter.from_evaluation_context(tmp_path / task_id, mock=True)
        adapter.setup_state(task["_asil"].get("initial_state", "default"))

        observation = adapter.observe()
        for action_data in task["_asil"]["actions"]:
            action = Action(**action_data)
            observation = adapter.execute(action)

        report = evaluate_observation(
            observation,
            validation=task["_asil"].get("validation"),
            evaluator=task.get("evaluator"),
        )
        assert report.success, task_id
