import json
import stat
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

from asil.eval.evaluator import evaluate_observation
from asil.gui_agent.session import GUISessionStartupError
from asil.protocol import Action


class _FakeLocator:
    def __init__(self, count: int = 0) -> None:
        self._count = count
        self.first = self

    def count(self) -> int:
        return self._count

    def click(self, timeout=None) -> None:
        return None


class _FakePage:
    def __init__(self, url: str) -> None:
        self.url = url
        self.goto_calls: list[str] = []
        self.wait_for_selector_calls: list[str] = []
        self.wait_for_load_state_calls: list[str] = []
        self.evaluate_calls: list[str] = []

    def content(self) -> str:
        return "<html><body>code-server ready</body></html>"

    def wait_for_load_state(self, state: str, timeout=None) -> None:
        self.wait_for_load_state_calls.append(state)

    def wait_for_selector(self, selector: str, timeout=None) -> None:
        self.wait_for_selector_calls.append(selector)

    def evaluate(self, script: str):
        self.evaluate_calls.append(script)
        return {
            "ready": True,
            "bodyText": "code-server ready",
            "title": "code-server",
            "href": self.url,
            "elementCount": 5,
        }

    def is_closed(self) -> bool:
        return False

    def goto(self, url: str, wait_until=None, timeout=None) -> None:
        self.goto_calls.append(url)
        self.url = url

    def locator(self, selector: str) -> _FakeLocator:
        if selector == "body":
            locator = _FakeLocator(1)
            locator.inner_text = lambda timeout=0: "code-server ready"
            return locator
        return _FakeLocator()

    def wait_for_timeout(self, timeout_ms: int) -> None:
        return None


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "code_server"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def test_observe_returns_file_tree_tabs_and_editor(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)

    obs = adapter.observe()

    assert obs.meta.app_name == "code-server"
    assert obs.meta.observation_source == "file_parse"
    elements = {element.id: element for element in obs.interactive_elements}
    assert "folder:src" in elements
    assert "file:README.md" in elements
    assert "file:src/app.py" in elements
    assert "tab:README.md" in elements
    assert "editor:README.md" in elements
    assert "Project checklist" in elements["editor:README.md"].value["content"]
    assert obs.app_state.active_document == "README.md"


def test_execute_updates_editor_and_file_tree_state(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    action = Action(
        action_type="modify_file",
        target="workspace",
        params={
            "operations": [
                {"action": "open_file", "path": "src/app.py"},
                {"action": "replace_text", "path": "src/app.py", "old": "Hello from ASIL", "new": "Hello from code-server"},
                {"action": "create_file", "path": "notes/today.md", "content": "# Today\nShip the slice.\n"},
            ]
        },
    )

    obs = adapter.execute(action)

    elements = {element.id: element for element in obs.interactive_elements}
    assert "tab:src/app.py" in elements
    assert "editor:src/app.py" in elements
    assert "Hello from code-server" in elements["editor:src/app.py"].value["content"]
    assert "file:notes/today.md" in elements
    assert (adapter.workspace_path / "notes" / "today.md").read_text(encoding="utf-8").startswith("# Today")


def test_rendering_prefers_real_webpage_capture_when_base_url_is_available(tmp_path: Path, monkeypatch):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"

    captured = {}

    def fake_capture(url, output_path, **kwargs):
        captured["url"] = url
        captured["output_path"] = Path(output_path)
        captured.update(kwargs)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.code_server.capture_url_to_png", fake_capture)

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "code-server.png")

    assert artifact.actual_page is True
    assert artifact.kind == "web_page_capture"
    assert artifact.backend == "playwright+chromium"
    assert captured["url"].startswith(adapter.base_url)
    assert "folder=" in captured["url"]
    assert "file=" in captured["url"]
    assert str(adapter.workspace_path) in unquote(captured["url"])
    assert str(adapter.workspace_path / "README.md") in unquote(captured["url"])
    assert captured["initial_wait_ms"] == 5000
    assert captured["timeout_ms"] == 60000
    assert "wait_for_selectors" not in captured
    assert captured["optional_click_selectors"] == [
        "text=Yes, I trust the authors",
        "text=I understand",
    ]
    assert captured["keyboard_steps"] == [
        {"press": "Control+P"},
        {"wait_ms": 500},
        {"type": "README.md"},
        {"wait_ms": 500},
        {"press": "Enter"},
        {"wait_ms": 1_000},
    ]
    assert captured["wait_for_selectors_after_actions"] == [
        ".monaco-workbench",
        ".explorer-folders-view .monaco-list-row",
        ".editor-group-container",
        '.tabs-container .tab[aria-label*="README.md"]',
    ]
    assert captured["backend"] == "playwright"
    assert out == tmp_path / "code-server.png"


def test_rendering_expands_parent_directories_before_opening_nested_file(tmp_path: Path, monkeypatch):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"
    adapter._active_file = "src/app.py"

    captured = {}

    def fake_capture(url, output_path, **kwargs):
        captured.update(kwargs)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.code_server.capture_url_to_png", fake_capture)

    adapter.render_to_png(tmp_path / "nested.png")

    assert captured["optional_click_selectors"] == [
        "text=Yes, I trust the authors",
        "text=I understand",
    ]
    assert captured["keyboard_steps"] == [
        {"press": "Control+P"},
        {"wait_ms": 500},
        {"type": "src/app.py"},
        {"wait_ms": 500},
        {"press": "Enter"},
        {"wait_ms": 1_000},
    ]
    assert captured["timeout_ms"] == 60000


def test_from_evaluation_context_prefers_shared_workspace_root_when_configured(tmp_path: Path, monkeypatch):
    from asil.adapters.code_server import CodeServerAdapter

    shared_root = tmp_path / "shared-workspaces"
    monkeypatch.setenv("CODE_SERVER_WORKSPACE_ROOT", str(shared_root))

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path / "ignored", mock=True)

    assert adapter.workspace_path == shared_root / "code-server-workspace"
    assert (adapter.workspace_path / "README.md").exists()


def test_prepare_task_makes_shared_workspace_writable_by_gui_service(tmp_path: Path, monkeypatch):
    from asil.adapters.code_server import CodeServerAdapter

    shared_root = tmp_path / "shared-workspaces"
    monkeypatch.setenv("CODE_SERVER_WORKSPACE_ROOT", str(shared_root))
    adapter = CodeServerAdapter.from_evaluation_context(tmp_path / "ignored", mock=True)
    task = SimpleNamespace(
        initial_state="default",
        actions=[
            {
                "params": {
                    "operations": [
                        {"action": "append_text", "path": "new/plan.md", "text": "- Ship\n"},
                    ]
                }
            }
        ],
    )

    adapter.prepare_task(task)

    for path in (
        adapter.workspace_path,
        adapter.workspace_path / "README.md",
        adapter.workspace_path / "new",
        adapter.workspace_path / "new" / "plan.md",
    ):
        assert path.stat().st_mode & stat.S_IWOTH


def test_rendering_fallback_is_honest_when_no_live_page_is_configured(tmp_path: Path, monkeypatch):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = ""

    captured = {}

    def fake_capture(html_content, output_path, **kwargs):
        captured["html_content"] = html_content
        captured["output_path"] = Path(output_path)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.code_server.capture_html_to_png", fake_capture)

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "code-server-fallback.png")

    assert artifact.actual_page is False
    assert artifact.kind == "state_render"
    assert "code-server Workspace" in captured["html_content"]
    assert out == tmp_path / "code-server-fallback.png"


def test_observe_prefers_live_shadow_for_active_editor_and_file_contents(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter._set_gui_shadow_state(
        {
            "open_tabs": ["README.md", "src/app.py"],
            "active_file": "src/app.py",
            "text_files": {
                "src/app.py": 'def main() -> str:\n    return "Hello from GUI"\n',
            },
        }
    )

    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert elements["file:src/app.py"].value["content"] == 'def main() -> str:\n    return "Hello from GUI"\n'
    assert elements["editor:src/app.py"].value["content"] == 'def main() -> str:\n    return "Hello from GUI"\n'
    assert elements["tab:src/app.py"].value["active"] is True
    assert obs.app_state.active_document == "app.py"


def test_validate_action_accepts_modify_file_and_navigate(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)

    assert adapter.validate_action(Action(action_type="modify_file", target="workspace", params={}))
    assert adapter.validate_action(Action(action_type="navigate", target="src/app.py", params={}))
    assert not adapter.validate_action(Action(action_type="api_call", target="code_server", params={}))


def test_gui_session_spec_exposes_explicit_browser_readiness_probes(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "browser"
    assert spec.browser_url == "about:blank"
    assert spec.browser_navigation_mode == "current_page"
    assert spec.startup_timeout_s == 120.0
    assert spec.post_launch_callback is not None
    assert spec.backend_ready_probe is not None
    assert spec.ui_ready_probe is not None


def test_prime_browser_session_navigates_from_base_shell_to_workspace(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"
    page = _FakePage(adapter.base_url)
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert page.evaluate_calls
    assert page.goto_calls == [adapter._workspace_url()]


def test_prime_browser_session_raises_on_blank_shell(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"

    class CrashPage:
        url = adapter._workspace_url()

        def content(self) -> str:
            return "<html><body>Aw, Snap!</body></html>"

        def wait_for_load_state(self, state: str, timeout=None) -> None:
            return None

        def locator(self, selector: str):
            class Locator:
                def count(self) -> int:
                    return 1 if selector == "body" else 0

                @property
                def first(self):
                    return self

                def click(self, timeout=None) -> None:
                    return None

                def inner_text(self, timeout: int = 0) -> str:
                    return "Aw, Snap!"

            return Locator()

    class FakeSession:
        def __init__(self):
            self.browser_page = CrashPage()

    try:
        adapter._prime_browser_session(FakeSession())
        assert False, "Expected blank/crash shell to abort code-server startup"
    except GUISessionStartupError as exc:
        assert exc.category == "browser_crashed"


def test_prime_browser_session_allows_short_blank_warmup_before_workbench_appears(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"

    class WarmupPage(_FakePage):
        def __init__(self, url: str) -> None:
            super().__init__(url)
            self._warm = True

        def content(self) -> str:
            if self._warm:
                return "<html><body></body></html>"
            return "<html><body>code-server ready</body></html>"

        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.wait_for_selector_calls.append(selector)
            if selector == ".monaco-workbench":
                self._warm = False

        def locator(self, selector: str):
            if selector == "body":
                class Locator:
                    def count(self) -> int:
                        return 1

                    @property
                    def first(self):
                        return self

                    def click(self, timeout=None) -> None:
                        return None

                    def inner_text(self, timeout: int = 0) -> str:
                        return "" if self_outer._warm else "code-server ready"

                self_outer = self
                return Locator()
            return super().locator(selector)

    page = WarmupPage(adapter.base_url)
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert page.evaluate_calls


def test_prime_browser_session_polls_dom_readiness_without_long_selector_wait(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8080"

    class PollingPage(_FakePage):
        def __init__(self, url: str) -> None:
            super().__init__(url)
            self.poll_count = 0
            self.selector_timeouts: list[int | None] = []
            self.function_timeouts: list[int | None] = []

        def wait_for_load_state(self, state: str, timeout=None) -> None:
            self.wait_for_load_state_calls.append(state)
            assert timeout is None or timeout <= 5_000

        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.selector_timeouts.append(timeout)
            raise AssertionError("code-server startup should not depend on a long selector wait")

        def evaluate(self, script: str):
            self.evaluate_calls.append(script)
            self.poll_count += 1
            return {
                "ready": self.poll_count >= 2,
                "bodyText": "Loading workbench" if self.poll_count < 2 else "Explorer README.md",
                "title": "code-server",
                "href": self.url,
                "elementCount": 10,
            }

        def wait_for_function(self, script: str, timeout: int = 0) -> None:
            del script
            self.function_timeouts.append(timeout)
            self.poll_count += 1
            if self.poll_count < 2:
                raise TimeoutError("Workbench is still starting")

    page = PollingPage(adapter.base_url)
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert len(page.function_timeouts) >= 2
    assert all(timeout is not None and timeout <= 5_000 for timeout in page.function_timeouts)
    assert page.evaluate_calls == []
    assert page.selector_timeouts == []


def test_wait_for_workbench_ready_records_diagnostics_on_timeout(tmp_path: Path, monkeypatch):
    from asil.adapters import code_server as code_server_module
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)

    class BlankPage:
        url = "http://127.0.0.1:8080"

        def __init__(self) -> None:
            self.wait_for_function_calls: list[int] = []

        def content(self) -> str:
            return "<html><body></body></html>"

        def wait_for_function(self, script: str, timeout: int = 0) -> None:
            del script
            self.wait_for_function_calls.append(timeout)
            raise TimeoutError("workbench still loading")

        def wait_for_timeout(self, timeout_ms: int) -> None:
            del timeout_ms

        def is_closed(self) -> bool:
            return False

        def locator(self, selector: str) -> _FakeLocator:
            if selector == "body":
                locator = _FakeLocator(1)
                locator.inner_text = lambda timeout=0: ""
                return locator
            return _FakeLocator()

    ticks = iter([0.0, 0.1, 0.2, 1.2])
    monkeypatch.setattr(code_server_module.time, "monotonic", lambda: next(ticks, 1.3))
    session = type(
        "Session",
        (),
        {
            "browser_page": BlankPage(),
            "startup_diagnostics": {"phases": [], "snapshots": []},
        },
    )()

    try:
        adapter._wait_for_workbench_ready(session, timeout_ms=1)
        assert False, "Expected blank workbench timeout"
    except GUISessionStartupError as exc:
        assert exc.category == "blank_shell"

    assert session.browser_page.wait_for_function_calls
    assert session.browser_page.wait_for_function_calls[0] <= 5_000
    assert session.startup_diagnostics["snapshots"][-1]["label"] == "code_server_timeout"


def test_sync_from_gui_falls_back_to_filesystem_shadow_when_page_crashes(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    adapter = CodeServerAdapter.from_evaluation_context(tmp_path, mock=True)

    class CrashedPage:
        def evaluate(self, script: str):
            del script
            raise RuntimeError("Page.evaluate: Target crashed")

    session = type("Session", (), {"browser_page": CrashedPage()})()

    adapter.sync_from_gui(session)
    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert elements["editor:README.md"].value["content"].startswith("# Project checklist")


def test_code_server_example_set_contains_20_tasks():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "code_server"
    tasks = sorted(path for path in root.glob("code_server_*.json") if path.stem.removeprefix("code_server_").isdigit())

    assert len(tasks) == 20
    ids = [json.loads(task.read_text(encoding="utf-8"))["id"] for task in tasks]
    assert ids == [f"code_server_{idx:02d}" for idx in range(1, 21)]


def test_representative_code_server_tasks_evaluate_successfully(tmp_path: Path):
    from asil.adapters.code_server import CodeServerAdapter

    for task_id in ("code_server_01", "code_server_11", "code_server_20"):
        task = _task(task_id)
        adapter = CodeServerAdapter.from_evaluation_context(tmp_path / task_id, mock=True)
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
