import json
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import quote

from asil.eval.evaluator import evaluate_observation
from asil.gui_agent.session import GUISessionStartupError
from asil.protocol import Action


class _FakePage:
    def __init__(self, url: str) -> None:
        self.url = url
        self.goto_calls: list[str] = []
        self.wait_for_selector_calls: list[str] = []
        self.wait_for_load_state_calls: list[str] = []

    def content(self) -> str:
        return "<html><body>draw.io ready</body></html>"

    def wait_for_load_state(self, state: str, timeout=None) -> None:
        self.wait_for_load_state_calls.append(state)

    def wait_for_selector(self, selector: str, timeout=None) -> None:
        self.wait_for_selector_calls.append(selector)

    def goto(self, url: str, wait_until=None, timeout=None) -> None:
        self.goto_calls.append(url)
        self.url = url

    def locator(self, selector: str):
        class Locator:
            def count(self) -> int:
                return 1 if selector == "body" else 0

            def inner_text(self, timeout: int = 0) -> str:
                return "draw.io ready"

        return Locator()


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "drawio"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def test_observe_returns_canvas_shapes_and_connectors(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    obs = adapter.observe()

    assert obs.meta.app_name == "draw.io"
    elements = {element.id: element for element in obs.interactive_elements}
    assert "canvas" in elements
    assert "shape:start" in elements
    assert "shape:review" in elements
    assert "connector:flow_1" in elements


def test_execute_updates_diagram_state(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    action = Action(
        action_type="modify_file",
        target="diagram",
        params={
            "operations": [
                {
                    "action": "add_shape",
                    "id": "deploy",
                    "label": "Deploy",
                    "shape_kind": "rectangle",
                    "x": 500,
                    "y": 160,
                    "width": 150,
                    "height": 70,
                    "fill": "#d1fae5",
                },
                {
                    "action": "add_connector",
                    "id": "flow_3",
                    "source": "review",
                    "target": "deploy",
                    "label": "approved",
                },
            ]
        },
    )

    obs = adapter.execute(action)
    elements = {element.id: element for element in obs.interactive_elements}

    assert elements["shape:deploy"].value["label"] == "Deploy"
    assert elements["connector:flow_3"].value["target"] == "deploy"


def test_rendering_prefers_real_webpage_capture_when_base_url_is_available(tmp_path: Path, monkeypatch):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"

    captured = {}

    def fake_capture(url, output_path, **kwargs):
        captured["url"] = url
        captured["output_path"] = Path(output_path)
        captured.update(kwargs)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.drawio.capture_url_to_png", fake_capture)

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "drawio.png")

    assert artifact.actual_page is True
    assert artifact.kind == "web_page_capture"
    assert artifact.backend == "playwright+chromium"
    assert captured["url"].startswith("http://127.0.0.1:8081/?")
    assert "#R" in captured["url"]
    raw_xml = unquote(captured["url"].split("#R", 1)[1])
    assert "<mxfile" in raw_xml
    assert "Review" in raw_xml
    assert captured["wait_for_selectors"] == [
        ".geMenubarContainer",
        ".geToolbarContainer",
        ".geDiagramContainer",
    ]
    assert captured["backend"] == "playwright"
    assert captured["full_page"] is False
    assert out == tmp_path / "drawio.png"


def test_rendering_retries_live_webpage_capture_after_transient_crash(tmp_path: Path, monkeypatch):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"

    attempts = {"count": 0}

    def flaky_capture(url, output_path, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Navigation failed because page crashed!")
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.drawio.capture_url_to_png", flaky_capture)
    monkeypatch.setattr("asil.adapters.drawio.time.sleep", lambda _seconds: None)

    out = adapter.render_to_png(tmp_path / "drawio.png")

    assert attempts["count"] == 2
    assert out == tmp_path / "drawio.png"


def test_write_state_emits_real_drawio_companion_file(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)

    xml_path = adapter.diagram_path.with_suffix("")
    assert xml_path.name.endswith(".drawio")
    assert xml_path.exists()

    xml_content = xml_path.read_text(encoding="utf-8")
    assert "<mxfile" in xml_content
    assert "Start" in xml_content
    assert "Review" in xml_content


def test_rendering_fallback_is_honest_when_no_live_page_is_configured(tmp_path: Path, monkeypatch):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = ""

    captured = {}

    def fake_capture(html_content, output_path, **kwargs):
        captured["html_content"] = html_content
        captured["output_path"] = Path(output_path)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("asil.adapters.drawio.capture_html_to_png", fake_capture)

    artifact = adapter.describe_rendering()
    out = adapter.render_to_png(tmp_path / "drawio-fallback.png")

    assert artifact.actual_page is False
    assert artifact.kind == "state_render"
    assert "draw.io Diagram" in captured["html_content"]
    assert out == tmp_path / "drawio-fallback.png"


def test_validate_action_accepts_modify_file(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)

    assert adapter.validate_action(Action(action_type="modify_file", target="diagram", params={}))
    assert not adapter.validate_action(Action(action_type="api_call", target="drawio", params={}))


def test_gui_session_spec_exposes_explicit_browser_readiness_probes(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "browser"
    assert spec.browser_url == adapter._live_editor_url()
    assert spec.browser_navigation_mode == "current_page"
    assert spec.post_launch_delay_s == 5.0
    assert spec.post_launch_callback is None
    assert spec.backend_ready_probe is not None
    assert spec.ui_ready_probe is None


def test_prime_browser_session_navigates_from_shell_to_live_editor_url(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"
    page = _FakePage(adapter._editor_shell_url())
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert ".geMenubarContainer" in page.wait_for_selector_calls
    assert page.goto_calls == [adapter._live_editor_url()]


def test_prime_browser_session_raises_on_browser_crash(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"

    class CrashPage:
        url = adapter._live_editor_url()

        def content(self) -> str:
            return "<html><body>Aw, Snap!</body></html>"

        def wait_for_load_state(self, state: str, timeout=None) -> None:
            return None

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
        assert False, "Expected browser crash to abort draw.io startup"
    except GUISessionStartupError as exc:
        assert exc.category == "browser_crashed"


def test_prime_browser_session_allows_short_blank_warmup_before_editor_shell_appears(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"

    class WarmupPage(_FakePage):
        def __init__(self, url: str) -> None:
            super().__init__(url)
            self._warm = True

        def content(self) -> str:
            if self._warm:
                return "<html><body></body></html>"
            return "<html><body>draw.io ready</body></html>"

        def wait_for_selector(self, selector: str, timeout=None) -> None:
            self.wait_for_selector_calls.append(selector)
            if selector == ".geMenubarContainer":
                self._warm = False

        def locator(self, selector: str):
            if selector == "body":
                class Locator:
                    def count(self) -> int:
                        return 1

                    def inner_text(self, timeout: int = 0) -> str:
                        return "" if self_outer._warm else "draw.io ready"

                self_outer = self
                return Locator()
            return super().locator(selector)

    page = WarmupPage(adapter._editor_shell_url())
    session = type("Session", (), {"browser_page": page})()

    adapter._prime_browser_session(session)

    assert ".geMenubarContainer" in page.wait_for_selector_calls


def test_sync_from_gui_prefers_live_editor_hash_state(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"
    live_state = {
        "canvas": {"width": 960, "height": 640, "background": "#f8fafc"},
        "shapes": [
            {
                "id": "review",
                "label": "Review",
                "shape_kind": "rectangle",
                "x": 300,
                "y": 140,
                "width": 150,
                "height": 80,
                "fill": "#fef3c7",
                "stroke": "#d97706",
            },
            {
                "id": "deploy",
                "label": "Deploy",
                "shape_kind": "rectangle",
                "x": 520,
                "y": 140,
                "width": 150,
                "height": 80,
                "fill": "#d1fae5",
                "stroke": "#059669",
            },
        ],
        "connectors": [
            {
                "id": "flow_2",
                "source": "review",
                "target": "deploy",
                "label": "approved",
                "stroke": "#64748b",
            }
        ],
    }
    live_xml = adapter._state_to_drawio_xml(live_state)
    live_url = f"{adapter.base_url}/?offline=1&stealth=1#R{quote(live_xml, safe='')}"
    session = type("Session", (), {"browser_page": type("Page", (), {"url": live_url})()})()

    adapter.sync_from_gui(session)
    obs = adapter.observe()
    elements = {element.id: element for element in obs.interactive_elements}

    assert elements["shape:deploy"].value["label"] == "Deploy"
    connector = next(element for element in obs.interactive_elements if element.id == "connector:flow_2")
    assert connector.value["target_label"] == "Deploy"


def test_sync_from_gui_reads_edited_graph_model_instead_of_stale_url_hash(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"
    stale_url = adapter._live_editor_url()
    edited_state = adapter._read_state()
    edited_state["shapes"].append(
        {
            "id": "deploy",
            "label": "Deploy",
            "shape_kind": "rectangle",
            "x": 520,
            "y": 140,
            "width": 150,
            "height": 80,
            "fill": "#d1fae5",
            "stroke": "#059669",
        }
    )
    edited_xml = adapter._state_to_drawio_xml(edited_state)

    class EditedPage:
        url = stale_url

        def __init__(self) -> None:
            self.evaluate_calls: list[str] = []

        def evaluate(self, expression: str) -> str:
            self.evaluate_calls.append(expression)
            return edited_xml

    page = EditedPage()
    session = type("Session", (), {"browser_page": page})()

    adapter.sync_from_gui(session)
    elements = {element.id: element for element in adapter.observe().interactive_elements}

    assert page.evaluate_calls
    assert elements["shape:deploy"].value["label"] == "Deploy"


def test_sync_from_gui_falls_back_to_url_hash_when_live_model_read_fails(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    adapter = DrawioAdapter.from_evaluation_context(tmp_path, mock=True)
    adapter.base_url = "http://127.0.0.1:8081"
    url_state = adapter._read_state()
    url_state["shapes"][1]["label"] = "Reviewed in URL"
    url_xml = adapter._state_to_drawio_xml(url_state)
    live_url = f"{adapter.base_url}/?offline=1&stealth=1#R{quote(url_xml, safe='')}"

    class FailingPage:
        url = live_url

        def evaluate(self, expression: str) -> str:
            raise RuntimeError("editor context disappeared")

    session = type("Session", (), {"browser_page": FailingPage()})()

    adapter.sync_from_gui(session)
    elements = {element.id: element for element in adapter.observe().interactive_elements}

    assert elements["shape:review"].value["label"] == "Reviewed in URL"


def test_drawio_example_set_contains_20_tasks():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "drawio"
    tasks = sorted(path for path in root.glob("drawio_*.json") if path.stem.removeprefix("drawio_").isdigit())

    assert len(tasks) == 20
    ids = [json.loads(task.read_text(encoding="utf-8"))["id"] for task in tasks]
    assert ids == [f"drawio_{idx:02d}" for idx in range(1, 21)]


def test_representative_drawio_tasks_evaluate_successfully(tmp_path: Path):
    from asil.adapters.drawio import DrawioAdapter

    for task_id in ("drawio_01", "drawio_11", "drawio_20"):
        task = _task(task_id)
        adapter = DrawioAdapter.from_evaluation_context(tmp_path / task_id, mock=True)
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
