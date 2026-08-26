from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from asil.adapter import GUISessionSpec
from asil.adapters.inkscape import InkscapeAdapter
from asil.benchmark import BenchmarkConfig, _result_root_for_config
from asil.protocol import Observation


def _fake_observation() -> Observation:
    from asil.protocol import AppState, Environment, Meta, Navigation

    return Observation(
        meta=Meta(app_name="Fake", app_version="", observation_source="unit"),
        app_state=AppState(current_view="view", active_document="doc", document_path="/tmp/doc"),
        interactive_elements=[],
        environment=Environment(),
        navigation=Navigation(current_path="/", breadcrumb=["/"]),
        data_summary="",
    )


def _install_fake_playwright(monkeypatch, sync_playwright_factory) -> None:
    playwright_module = types.ModuleType("playwright")
    sync_api_module = types.ModuleType("playwright.sync_api")
    sync_api_module.sync_playwright = sync_playwright_factory
    playwright_module.sync_api = sync_api_module
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api_module)


def test_result_root_for_single_gui_run_uses_gui_agent_tree(tmp_path: Path):
    config = BenchmarkConfig(
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        task_index="test_full15.json",
        participant="gui",
        run_mode="single",
        provider="openai",
        model="gpt-5.4",
    )

    result_root = _result_root_for_config(config)

    assert result_root == tmp_path / "results" / "pyautogui" / "screenshot" / "gpt-5.4"


def test_result_root_for_comparison_uses_comparison_tree(tmp_path: Path):
    config = BenchmarkConfig(
        output_dir=tmp_path / "results",
        output_json=tmp_path / "results.json",
        task_index="test_full15.json",
        participant="asil",
        run_mode="comparison",
        asil_execution="agentic",
        provider="openai",
        model="gpt-5.4",
    )

    result_root = _result_root_for_config(config)

    assert result_root == tmp_path / "results" / "comparison" / "agentic__gpt-5.4"


def test_parse_gui_response_handles_json_action():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        'Thought: click the highlighted button\\nAction: {"action_type":"CLICK","x":120,"y":45}'
    )

    assert output.thought == "click the highlighted button"
    assert output.action == GUIAction(action_type="CLICK", payload={"x": 120, "y": 45})


def test_parse_gui_response_handles_activate_app_json_action():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        'Thought: switch to the notebook window\nAction: {"action_type":"ACTIVATE_APP","app":"jupyterlab"}'
    )

    assert output.action == GUIAction(action_type="ACTIVATE_APP", payload={"app": "jupyterlab"})


def test_parse_gui_response_handles_pyautogui_click():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        "Thought: click the button\nAction: import pyautogui\npyautogui.click(120, 45)"
    )

    assert output.action == GUIAction(action_type="CLICK", payload={"x": 120, "y": 45})


def test_parse_gui_response_handles_pyautogui_hotkey():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        "Thought: save changes\nAction: pyautogui.hotkey('ctrl', 's')"
    )

    assert output.action == GUIAction(action_type="HOTKEY", payload={"keys": ["ctrl", "s"]})


def test_parse_gui_response_handles_special_tokens():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response("Thought: wait for UI update\nAction: WAIT")

    assert output.action == GUIAction(action_type="WAIT", payload={})


def test_extract_openai_output_handles_typed_reasoning_items():
    from asil.gui_agent.llm import _extract_openai_output

    class FakeOutputText:
        type = "output_text"

        def __init__(self, text: str):
            self.text = text

    class FakeSummaryPart:
        def __init__(self, text: str):
            self.text = text

    class FakeMessageItem:
        type = "message"

        def __init__(self, text: str):
            self.content = [FakeOutputText(text)]

    class FakeReasoningItem:
        type = "reasoning"

        def __init__(self, text: str):
            self.summary = [FakeSummaryPart(text)]

    class FakeResponse:
        def __init__(self):
            self.output = [
                FakeReasoningItem("Need to click Save before finishing."),
                FakeMessageItem("Thought: save the document\nAction: DONE"),
            ]

    output = _extract_openai_output(FakeResponse(), "gpt-5.4")

    assert output.text == "Thought: save the document\nAction: DONE"
    assert output.reasoning_summary == "Need to click Save before finishing."
    assert output.provider == "openai"
    assert output.model == "gpt-5.4"


def test_extract_openai_output_falls_back_to_output_text():
    from asil.gui_agent.llm import _extract_openai_output

    class FakeResponse:
        output = []
        output_text = "Thought: stop here\nAction: DONE"

    output = _extract_openai_output(FakeResponse(), "gpt-5.4")

    assert output.text == "Thought: stop here\nAction: DONE"
    assert output.reasoning_summary == ""


def test_openai_gui_llm_uses_configured_request_timeout(monkeypatch):
    from asil.gui_agent.llm import create_gui_llm_fn

    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured["request"] = kwargs

            class FakeResponse:
                output = []
                output_text = "Thought: visible enough\nAction: DONE"

            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = FakeResponses()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("ASIL_GUI_LLM_TIMEOUT_S", "42")
    monkeypatch.setenv("ASIL_GUI_AGENT_BACKEND", "legacy")

    llm_fn = create_gui_llm_fn(provider="openai", model="gpt-5.4", api_key="test-key")
    output = llm_fn("prompt", b"png")

    assert output.text == "Thought: visible enough\nAction: DONE"
    assert captured["client"]["timeout"] == 42.0


def test_openai_gpt54_gui_llm_uses_osworld_computer_backend(monkeypatch):
    from asil.gui_agent.llm import create_gui_llm_fn
    from asil.gui_agent.parser import GUIAction

    captured: dict[str, object] = {"requests": []}

    class FakeResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            captured["requests"].append(kwargs)
            self.calls += 1

            class FakeResponse:
                id = f"resp_{self.calls}"

            if self.calls == 1:
                FakeResponse.output = [
                    {
                        "type": "computer_call",
                        "call_id": "call_1",
                        "action": {"type": "click", "x": 12, "y": 34, "button": "left"},
                        "pending_safety_checks": [],
                    }
                ]
            else:
                FakeResponse.output = [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Thought: done\nAction: DONE"}],
                    }
                ]
            return FakeResponse()

    fake_responses = FakeResponses()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = fake_responses

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("ASIL_GUI_AGENT_BACKEND", raising=False)
    monkeypatch.setenv("ASIL_GUI_REASONING_EFFORT", "high")

    llm_fn = create_gui_llm_fn(provider="openai", model="gpt-5.4", api_key="test-key")
    first = llm_fn("prompt", b"png1")
    second = llm_fn("prompt", b"png2")

    assert first.provider == "openai-osworld-gpt54"
    assert captured["client"]["max_retries"] == 0
    assert first.metadata == {
        "backend": "osworld_gpt54",
        "computer_tool": "computer",
        "reasoning_effort": "high",
    }
    assert first.actions == (GUIAction("CLICK", {"x": 12, "y": 34, "button": "left"}),)
    assert second.text == "Thought: done\nAction: DONE"
    requests = captured["requests"]
    assert requests[0]["tools"] == [{"type": "computer"}]
    assert requests[0]["reasoning"]["effort"] == "high"
    assert "previous_response_id" not in requests[1]
    assert requests[0]["input"][0]["content"][0]["type"] == "input_image"
    assert requests[0]["input"][0]["content"][1]["type"] == "input_text"

    second_input = requests[1]["input"]
    assert len(second_input) == 3
    assert second_input[0]["role"] == "user"
    assert second_input[1]["type"] == "computer_call"
    assert second_input[1]["call_id"] == "call_1"
    assert second_input[2]["type"] == "computer_call_output"
    assert second_input[2]["call_id"] == "call_1"
    assert second_input[2]["output"]["type"] == "input_image"
    assert second_input[2]["output"]["image_url"].startswith("data:image/png;base64,")


def test_osworld_gpt54_reset_clears_task_conversation_state():
    from asil.gui_agent.osworld_gpt54 import OSWorldGPT54ComputerLLM

    llm_fn = OSWorldGPT54ComputerLLM(api_key="test-key")
    llm_fn.cua_messages = [{"role": "user", "content": "previous task"}]
    llm_fn.pending_call_id = "previous-call"
    llm_fn.pending_safety_checks = [{"id": "previous-check"}]

    llm_fn.reset()

    assert llm_fn.cua_messages == []
    assert llm_fn.pending_call_id == ""
    assert llm_fn.pending_safety_checks == []


def test_osworld_gpt54_reset_rejects_late_response_from_previous_task(monkeypatch):
    from threading import Event, Thread

    from asil.gui_agent.osworld_gpt54 import OSWorldGPT54ComputerLLM

    llm_fn = OSWorldGPT54ComputerLLM(api_key="test-key")
    old_request_started = Event()
    release_old_request = Event()

    def fake_create_response(request_input):
        serialized = json.dumps(request_input, default=str)
        if "old task" in serialized:
            old_request_started.set()
            assert release_old_request.wait(timeout=2.0)
            label = "old task response"
        else:
            assert "new task" in serialized
            label = "new task response"
        return types.SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": label}],
                }
            ]
        )

    monkeypatch.setattr(llm_fn, "_create_response", fake_create_response)
    old_result = {}

    def run_old_request():
        old_result["output"] = llm_fn("old task", b"old-png")

    old_thread = Thread(target=run_old_request)
    old_thread.start()
    assert old_request_started.wait(timeout=2.0)

    llm_fn.reset()
    new_output = llm_fn("new task", b"new-png")
    release_old_request.set()
    old_thread.join(timeout=2.0)

    assert old_thread.is_alive() is False
    assert new_output.text == "new task response"
    assert old_result["output"].text == "old task response"
    serialized_state = json.dumps(llm_fn.cua_messages, default=str)
    assert "new task" in serialized_state
    assert "new task response" in serialized_state
    assert "old task" not in serialized_state


def test_legacy_openai_gui_backend_can_be_forced(monkeypatch):
    from asil.gui_agent.llm import create_gui_llm_fn

    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured["request"] = kwargs

            class FakeResponse:
                output = []
                output_text = "Thought: visible enough\nAction: DONE"

            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = FakeResponses()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("ASIL_GUI_AGENT_BACKEND", "legacy")

    llm_fn = create_gui_llm_fn(provider="openai", model="gpt-5.4", api_key="test-key")
    output = llm_fn("prompt", b"png")

    assert output.text == "Thought: visible enough\nAction: DONE"
    assert "tools" not in captured["request"]


def test_parse_gui_response_maps_sleep_to_wait():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response("Thought: let the dialog settle\nAction: import time\ntime.sleep(2)")

    assert output.action == GUIAction(action_type="WAIT", payload={"seconds": 2.0})


def test_parse_gui_response_ignores_thought_lines_inside_action_block():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        "Thought: understand the current notebook state\n"
        "Action:\n"
        "Thought: the code cell is focused and should be replaced\n"
        "pyautogui.hotkey('ctrl', 'a')"
    )

    assert output.action == GUIAction(action_type="HOTKEY", payload={"keys": ["ctrl", "a"]})


def test_parse_gui_response_recovers_pyautogui_call_from_noisy_action_text():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        "Thought: the rectangle needs to become blue\n"
        "Action: I should click the blue swatch next, so use pyautogui.click(449, 922) now."
    )

    assert output.action == GUIAction(action_type="CLICK", payload={"x": 449, "y": 922})


def test_parse_gui_response_handles_multiline_pyautogui_write_text():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        'Thought: replace the markdown file contents\n'
        'Action: pyautogui.write("# Summary\n\nThe KPI notebook is ready for review.\n")'
    )

    assert output.action == GUIAction(
        action_type="TYPING",
        payload={"text": "# Summary\n\nThe KPI notebook is ready for review.\n"},
    )


def test_parse_gui_response_handles_multiline_pyautogui_write_embedded_in_prose():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        'Thought: replace the markdown file contents\n'
        'Action: The editor is ready, so use pyautogui.write("# Summary\n\nUpdated line\n") now.'
    )

    assert output.action == GUIAction(
        action_type="TYPING",
        payload={"text": "# Summary\n\nUpdated line\n"},
    )


def test_parse_gui_response_handles_loose_json_typing_with_raw_newlines():
    from asil.gui_agent.parser import GUIAction, parse_gui_response

    output = parse_gui_response(
        'Thought: replace the markdown file contents\n'
        'Action: {"action_type":"TYPING","text":"# Summary\n\nUpdated line\n"}'
    )

    assert output.action == GUIAction(
        action_type="TYPING",
        payload={"text": "# Summary\n\nUpdated line\n"},
    )


def test_gui_prompt_warns_against_premature_done_and_retries_failed_clicks():
    from asil.gui_agent.prompts import GUI_SYSTEM_PROMPT

    assert "Do not stop early" in GUI_SYSTEM_PROMPT
    assert "If a click or key press does not appear to change the visible document" in GUI_SYSTEM_PROMPT
    assert "Prefer actions that directly change the document or saved application state" in GUI_SYSTEM_PROMPT
    assert '{"action_type":"ACTIVATE_APP","app":"code_server"}' in GUI_SYSTEM_PROMPT


def test_gui_prompt_adds_jupyterlab_specific_tips_and_json_typing_guidance():
    from asil.gui_agent.prompts import GUI_SYSTEM_PROMPT, build_gui_user_prompt

    prompt = build_gui_user_prompt(
        instruction='Open notebooks/summary.md and replace it with "# Summary\\n\\nUpdated\\n".',
        window_width=1280,
        window_height=900,
        history=[],
        software="jupyterlab",
    )

    assert '{"action_type":"TYPING","text":"..."}' in GUI_SYSTEM_PROMPT
    assert 'use ONLY the JSON TYPING form: {"action_type":"TYPING","text":"..."}' in GUI_SYSTEM_PROMPT
    assert "JupyterLab tips:" in prompt
    assert "Esc to enter command mode" in prompt
    assert "If renaming a file and the extension is already present" in prompt
    assert "do not describe the typing action in prose" in prompt


def test_gui_prompt_adds_multi_app_switching_guidance():
    from asil.gui_agent.prompts import build_gui_user_prompt

    prompt = build_gui_user_prompt(
        instruction="Update the code, notebook, and issue.",
        window_width=1200,
        window_height=800,
        history=[],
        software="multi_apps",
        related_apps=["code_server", "jupyterlab", "gitea"],
    )

    assert "Multi-app tips:" in prompt
    assert "code_server, jupyterlab, gitea" in prompt
    assert '{"action_type":"ACTIVATE_APP","app":"<app_name>"}' in prompt
    assert "Alt+Tab" in prompt
    assert "current working window" in prompt
    assert "code-server: if the target file is open" in prompt
    assert "avoid right-click context menus" in prompt
    assert "JupyterLab: use the file browser" in prompt
    assert "Gitea: use the visible repository web UI" in prompt


def test_resolve_gui_session_spec_prefers_adapter_hook(sample_svg: Path):
    from asil.gui_agent.session import resolve_gui_session_spec

    adapter = InkscapeAdapter(sample_svg)

    spec = resolve_gui_session_spec(adapter)

    assert isinstance(spec, GUISessionSpec)
    assert spec.surface_type == "desktop"
    assert "inkscape" in " ".join(spec.launch_command).lower()
    assert spec.run_as_user == "asilgui"


def test_gui_controller_click_maps_window_relative_coordinates(monkeypatch, tmp_path: Path):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, env):
        del check, capture_output, env
        calls.append(cmd)
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "123")
    monkeypatch.setattr("asil.gui_agent.controller._window_geometry", lambda *args, **kwargs: (100, 200, 800, 600))

    controller = X11GUIController()
    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Inkscape",
        launch_command=("inkscape", "file.svg"),
    )

    controller.execute(
        GUIAction(action_type="CLICK", payload={"x": 20, "y": 30}),
        spec=spec,
    )

    joined = [" ".join(command) for command in calls]
    assert any("mousemove --sync 120 230 click" in command for command in joined)


def test_gui_controller_uses_active_window_for_multi_window(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, env):
        del check, capture_output, env
        calls.append(cmd)
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.active_window_id", lambda display=None: "999")
    monkeypatch.setattr("asil.gui_agent.controller._window_geometry", lambda *args, **kwargs: (10, 20, 800, 600))

    controller = X11GUIController()
    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        capture_active_window=True,
    )

    controller.execute(GUIAction(action_type="CLICK", payload={"x": 5, "y": 6}), spec=spec)

    assert any(command[:3] == ["xdotool", "windowactivate", "--sync"] and command[3] == "999" for command in calls)
    assert any("mousemove --sync 15 26 click" in " ".join(command) for command in calls)


def test_gui_controller_uses_last_capture_window_for_multi_window_coordinates(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, env):
        del check, capture_output, env
        calls.append(cmd)
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.active_window_id", lambda display=None: "999")

    def fake_geometry(window_id, **kwargs):
        if window_id == "555":
            return (100, 200, 800, 600)
        return (10, 20, 800, 600)

    monkeypatch.setattr("asil.gui_agent.controller._window_geometry", fake_geometry)

    controller = X11GUIController()
    controller.set_capture_window_id("555")
    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        capture_active_window=True,
    )

    controller.execute(GUIAction(action_type="CLICK", payload={"x": 5, "y": 6}), spec=spec)

    assert any(command[:3] == ["xdotool", "windowactivate", "--sync"] and command[3] == "555" for command in calls)
    assert any("mousemove --sync 105 206 click" in " ".join(command) for command in calls)


def test_gui_controller_typing_uses_option_terminator_for_leading_hyphen(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, env):
        del check, capture_output, env
        calls.append(cmd)
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "123")

    controller = X11GUIController()
    spec = GUISessionSpec(surface_type="desktop", window_title_pattern="Fake")

    controller.execute(
        GUIAction(action_type="TYPING", payload={"text": "- Confirm the release window"}),
        spec=spec,
    )

    assert calls[-1] == [
        "xdotool",
        "type",
        "--delay",
        "0",
        "--",
        "- Confirm the release window",
    ]


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        ("TYPING", {"text": "Renamed Track"}),
        ("CLIPBOARD_PASTE", {"text": "Renamed Track"}),
        ("PRESS", {"key": "enter"}),
        ("KEY_DOWN", {"key": "ctrl"}),
        ("KEY_UP", {"key": "ctrl"}),
        ("HOTKEY", {"keys": ["ctrl", "a"]}),
    ],
)
def test_gui_controller_keyboard_actions_keep_active_transient_dialog_focused(
    monkeypatch,
    action_type,
    payload,
):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        if cmd[0] == "xprop":
            return types.SimpleNamespace(
                stdout="WM_TRANSIENT_FOR(WINDOW): window id # 0x64\n",
            )
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr(
        "asil.gui_agent.controller.shutil.which",
        lambda name: {"xdotool": "xdotool", "xprop": "xprop", "xclip": "xclip"}.get(name),
    )
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "100")
    monkeypatch.setattr("asil.gui_agent.controller.active_window_id", lambda display=None: "200")

    controller = X11GUIController()
    spec = GUISessionSpec(surface_type="desktop", window_title_pattern="Fake")

    controller.execute(
        GUIAction(action_type=action_type, payload=payload),
        spec=spec,
    )

    assert ["xdotool", "windowactivate", "--sync", "200"] in calls
    assert ["xdotool", "windowactivate", "--sync", "100"] not in calls


def test_gui_controller_keyboard_action_activates_main_for_unrelated_active_window(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        if cmd[0] == "xprop":
            return types.SimpleNamespace(stdout="WM_TRANSIENT_FOR:  not found.\n")
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr(
        "asil.gui_agent.controller.shutil.which",
        lambda name: {"xdotool": "xdotool", "xprop": "xprop"}.get(name),
    )
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "100")
    monkeypatch.setattr("asil.gui_agent.controller.active_window_id", lambda display=None: "300")

    controller = X11GUIController()
    spec = GUISessionSpec(surface_type="desktop", window_title_pattern="Fake")

    controller.execute(
        GUIAction(action_type="HOTKEY", payload={"keys": ["ctrl", "a"]}),
        spec=spec,
    )

    assert ["xdotool", "windowactivate", "--sync", "100"] in calls
    assert ["xdotool", "windowactivate", "--sync", "300"] not in calls


def test_gui_controller_click_still_uses_main_window_when_dialog_is_active(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "100")
    monkeypatch.setattr("asil.gui_agent.controller.active_window_id", lambda display=None: "200")
    monkeypatch.setattr("asil.gui_agent.controller._window_geometry", lambda *args, **kwargs: (10, 20, 800, 600))

    controller = X11GUIController()
    spec = GUISessionSpec(surface_type="desktop", window_title_pattern="Fake")

    controller.execute(
        GUIAction(action_type="CLICK", payload={"x": 5, "y": 6}),
        spec=spec,
    )

    assert ["xdotool", "windowactivate", "--sync", "100"] in calls
    assert calls[-1] == [
        "xdotool",
        "mousemove",
        "--sync",
        "15",
        "26",
        "click",
        "--repeat",
        "1",
        "1",
    ]


def test_gui_controller_drag_uses_current_pointer_as_start(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, env):
        del check, capture_output, env
        calls.append(cmd)
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "123")
    monkeypatch.setattr("asil.gui_agent.controller._window_geometry", lambda *args, **kwargs: (100, 200, 800, 600))

    controller = X11GUIController()
    spec = GUISessionSpec(surface_type="desktop", window_title_pattern="Fake")
    controller.execute(GUIAction(action_type="MOVE_TO", payload={"x": 10, "y": 20}), spec=spec)
    controller.execute(GUIAction(action_type="DRAG_TO", payload={"x": 30, "y": 40}), spec=spec)

    assert calls[-3:] == [
        ["xdotool", "mousedown", "1"],
        ["xdotool", "mousemove", "--sync", "130", "240"],
        ["xdotool", "mouseup", "1"],
    ]


def test_gui_controller_drag_releases_mouse_when_move_fails(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, env):
        del check, capture_output, env
        calls.append(cmd)
        if cmd[1:3] == ["mousemove", "--sync"]:
            raise RuntimeError("move failed")
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr("asil.gui_agent.controller.shutil.which", lambda name: "xdotool")
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "123")
    monkeypatch.setattr("asil.gui_agent.controller._window_geometry", lambda *args, **kwargs: (0, 0, 800, 600))

    controller = X11GUIController()
    spec = GUISessionSpec(surface_type="desktop", window_title_pattern="Fake")

    with pytest.raises(RuntimeError, match="move failed"):
        controller.execute(GUIAction(action_type="DRAG_TO", payload={"x": 30, "y": 40}), spec=spec)

    assert calls[-1] == ["xdotool", "mouseup", "1"]


def test_gui_controller_clipboard_paste_uses_xclip_and_hotkey(monkeypatch):
    from asil.gui_agent.controller import GUIAction, X11GUIController

    calls: list[dict[str, object]] = []

    def fake_run(cmd, check, capture_output, env, input=None):
        del check, capture_output, env
        calls.append({"cmd": cmd, "input": input})
        return None

    monkeypatch.setattr("asil.gui_agent.controller.subprocess.run", fake_run)
    monkeypatch.setattr(
        "asil.gui_agent.controller.shutil.which",
        lambda name: {"xdotool": "xdotool", "xclip": "xclip"}.get(name, name),
    )
    monkeypatch.setattr("asil.gui_agent.controller.ensure_virtual_display", lambda display=None: {"DISPLAY": display or ":99"})
    monkeypatch.setattr("asil.gui_agent.controller.wait_for_window", lambda *args, **kwargs: "123")

    controller = X11GUIController()
    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Fake",
        launch_command=("fake",),
    )

    controller.execute(
        GUIAction(action_type="CLIPBOARD_PASTE", payload={"text": "hello\n世界"}),
        spec=spec,
    )

    assert calls[1]["cmd"] == ["xclip", "-selection", "clipboard"]
    assert calls[1]["input"] == "hello\n世界".encode("utf-8")
    assert calls[-1]["cmd"][-1] == "ctrl+v"


def test_gui_session_capture_uses_active_window_for_multi_window(monkeypatch, tmp_path: Path):
    from asil.gui_agent.session import GUISession

    seen: dict[str, object] = {}

    def fake_capture(output_path, **kwargs):
        seen["output_path"] = output_path
        seen.update(kwargs)

    monkeypatch.setattr("asil.gui_agent.session.capture_window_to_png", fake_capture)

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        capture_active_window=True,
    )
    session = GUISession(spec=spec)

    assert session.capture(tmp_path / "step_0.png") is True
    assert seen["active_window"] is True
    assert seen["fallback_window_specs"] == []
    assert seen["prefer_first_fallback"] is True
    assert session.last_capture_metadata == {"capture_complete": True}


def test_gui_session_capture_passes_multi_window_fallback_specs(monkeypatch, tmp_path: Path):
    from asil.gui_agent.session import GUISession

    seen: dict[str, object] = {}

    def fake_capture(output_path, **kwargs):
        del output_path
        seen.update(kwargs)
        kwargs["capture_metadata"].update(
            {
                "capture_complete": True,
                "window_id": "0xcode",
                "fallback_app": "code_server",
            }
        )

    monkeypatch.setattr("asil.gui_agent.session.capture_window_to_png", fake_capture)

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        capture_active_window=True,
        child_specs={
            "code_server": GUISessionSpec(surface_type="browser", window_title_pattern="Code", min_width=900, min_height=700),
            "jupyterlab": GUISessionSpec(surface_type="browser", window_title_pattern="Jupyter", min_width=900, min_height=700),
        },
        primary_child="jupyterlab",
    )
    session = GUISession(spec=spec, active_child="code_server")

    assert session.capture(tmp_path / "step_0.png") is True

    fallback_specs = seen["fallback_window_specs"]
    assert [item["app"] for item in fallback_specs] == ["code_server", "jupyterlab"]
    assert seen["prefer_first_fallback"] is True
    assert session.last_capture_window_id == "0xcode"
    assert session.active_child == "code_server"


def test_gui_session_activate_app_focuses_child_window_and_browser_page(monkeypatch):
    from asil.gui_agent.session import GUISession

    activated: list[tuple[str, str | None]] = []
    brought_to_front: list[str] = []

    class FakePage:
        def bring_to_front(self):
            brought_to_front.append("front")

    monkeypatch.setattr(
        "asil.gui_agent.session.activate_window",
        lambda title, **kwargs: activated.append((title, kwargs.get("window_class_pattern"))) or "0xjupyter",
    )

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        child_specs={
            "jupyterlab": GUISessionSpec(
                surface_type="browser",
                window_title_pattern="Jupyter",
                window_class_pattern="chromium",
                min_width=900,
                min_height=700,
            )
        },
    )
    child = GUISession(spec=spec.child_specs["jupyterlab"], browser_page=FakePage())
    session = GUISession(spec=spec, child_sessions={"jupyterlab": child})

    window_id = session.activate_app("jupyterlab")

    assert window_id == "0xjupyter"
    assert brought_to_front == ["front"]
    assert activated == [("Jupyter", "chromium")]
    assert session.active_child == "jupyterlab"
    assert session.last_capture_window_id == "0xjupyter"


def test_start_gui_session_does_not_fix_permissions_on_system_binaries(monkeypatch, tmp_path: Path):
    from asil.gui_agent.session import start_gui_session

    fixed_paths: list[Path] = []

    class FakeProcess:
        def poll(self):
            return None

    data_file = tmp_path / "writer.odt"
    data_file.write_text("placeholder", encoding="utf-8")
    home_dir = tmp_path / "gui_home"
    (home_dir / "home").mkdir(parents=True)

    monkeypatch.setattr("asil.gui_agent.session.ensure_user_access", lambda path, run_as_user: fixed_paths.append(Path(path)))
    monkeypatch.setattr("asil.gui_agent.session.launch_gui_process", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("asil.gui_agent.session.terminate_process", lambda proc: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)

    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Writer",
        launch_command=("/bin/sh", str(data_file)),
        run_as_user="asilgui",
        extra_env={"HOME": str(home_dir / "home")},
    )

    with start_gui_session(spec):
        pass

    assert data_file in fixed_paths
    assert Path("/bin/sh") not in fixed_paths


def test_start_gui_session_retries_browser_launch_after_early_exit(monkeypatch):
    from asil.gui_agent.session import GUISession, GUISessionStartupError, start_gui_session

    launch_attempts: list[int] = []

    def fake_launch(*args, **kwargs):
        del args, kwargs
        launch_attempts.append(len(launch_attempts) + 1)
        if len(launch_attempts) == 1:
            raise GUISessionStartupError("browser_crashed", "Browser exited early.")
        return GUISession(
            spec=GUISessionSpec(
                surface_type="browser",
                browser_url="http://example.test",
                window_title_pattern=r".*",
            )
        )

    monkeypatch.setattr("asil.gui_agent.session._launch_browser_session", fake_launch)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test",
        window_title_pattern=r".*",
        launch_command=(),
    )

    session = start_gui_session(spec)

    assert launch_attempts == [1, 2]
    session.close()


def test_start_gui_session_runs_browser_post_launch_callback(monkeypatch):
    from asil.gui_agent.session import GUISession, start_gui_session

    callback_calls: list[str] = []

    monkeypatch.setattr(
        "asil.gui_agent.session._launch_browser_session",
        lambda spec: GUISession(spec=spec),
    )
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test",
        window_title_pattern=r".*",
        launch_command=(),
        post_launch_callback=lambda: callback_calls.append("called"),
    )

    session = start_gui_session(spec)

    assert callback_calls == ["called"]
    session.close()


def test_start_gui_session_passes_session_to_optional_session_callback(monkeypatch):
    from asil.gui_agent.session import GUISession, start_gui_session

    seen_sessions: list[GUISession] = []

    monkeypatch.setattr(
        "asil.gui_agent.session._launch_browser_session",
        lambda spec: GUISession(spec=spec),
    )
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    def callback(session=None):
        seen_sessions.append(session)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test",
        window_title_pattern=r".*",
        launch_command=(),
        post_launch_callback=callback,
    )

    session = start_gui_session(spec)

    assert seen_sessions == [session]
    session.close()


def test_start_gui_session_runs_backend_probe_before_launch(monkeypatch):
    from asil.gui_agent.session import start_gui_session

    call_order: list[str] = []

    class FakeProcess:
        def poll(self):
            return None

    monkeypatch.setattr("asil.gui_agent.session.ensure_virtual_display", lambda *args, **kwargs: {"DISPLAY": ":99"})
    monkeypatch.setattr(
        "asil.gui_agent.session.launch_gui_process",
        lambda *args, **kwargs: call_order.append("launch") or FakeProcess(),
    )
    monkeypatch.setattr("asil.gui_agent.session.terminate_process", lambda proc: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Writer",
        launch_command=("libreoffice", "--writer", "/tmp/doc.odt"),
        backend_ready_probe=lambda: call_order.append("backend"),
    )

    session = start_gui_session(spec)

    assert call_order == ["backend", "launch"]
    session.close()


def test_start_gui_session_launches_multi_window_children_and_activates_primary(monkeypatch):
    from asil.gui_agent.session import GUISession, create_startup_diagnostics, start_gui_session

    launched: list[str] = []
    activated: list[str] = []

    def fake_launch_child(spec, diagnostics, **_kwargs):
        launched.append(spec.window_title_pattern)
        return GUISession(spec=spec, startup_diagnostics=diagnostics)

    monkeypatch.setattr("asil.gui_agent.session._launch_child_session", fake_launch_child)
    monkeypatch.setattr("asil.gui_agent.session.activate_window", lambda title, **kwargs: activated.append(title) or "123")
    monkeypatch.setattr("asil.gui_agent.session._cleanup_gui_processes", lambda spec: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        child_specs={
            "code_server": GUISessionSpec(surface_type="desktop", window_title_pattern="Code", launch_command=("code",)),
            "jupyterlab": GUISessionSpec(surface_type="desktop", window_title_pattern="Jupyter", launch_command=("jupyter",)),
        },
        primary_child="jupyterlab",
        capture_active_window=True,
    )
    diagnostics = create_startup_diagnostics(spec)

    session = start_gui_session(spec, startup_diagnostics=diagnostics)

    assert launched == ["Code", "Jupyter"]
    assert activated == ["Jupyter"]
    assert set(session.child_sessions) == {"code_server", "jupyterlab"}
    assert session.active_child == "jupyterlab"
    assert session.last_capture_window_id == "123"
    assert set(diagnostics["children"]) == {"code_server", "jupyterlab"}
    session.close()


def test_start_gui_session_keeps_multi_window_session_when_primary_activation_fails(monkeypatch):
    from asil.gui_agent.session import GUISession, create_startup_diagnostics, start_gui_session

    def fake_launch_child(spec, diagnostics, **_kwargs):
        return GUISession(spec=spec, startup_diagnostics=diagnostics)

    monkeypatch.setattr("asil.gui_agent.session._launch_child_session", fake_launch_child)
    monkeypatch.setattr(
        "asil.gui_agent.session.activate_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("primary not found")),
    )
    monkeypatch.setattr("asil.gui_agent.session._cleanup_gui_processes", lambda spec: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        child_specs={
            "obs": GUISessionSpec(surface_type="desktop", window_title_pattern="OBS", launch_command=("obs",)),
            "thunderbird": GUISessionSpec(surface_type="desktop", window_title_pattern="Thunderbird", launch_command=("thunderbird",)),
        },
        primary_child="obs",
        capture_active_window=True,
    )
    diagnostics = create_startup_diagnostics(spec)

    session = start_gui_session(spec, startup_diagnostics=diagnostics)

    assert set(session.child_sessions) == {"obs", "thunderbird"}
    assert session.active_child == "obs"
    assert session.last_capture_window_id == ""
    primary_phase = [phase for phase in diagnostics["phases"] if phase["name"] == "primary_window_activate"][0]
    assert primary_phase["status"] == "warning"
    assert primary_phase["fallback"] == "capture_phase_main_window_recovery"
    session.close()


def test_start_gui_session_launches_multi_window_browser_children_first(monkeypatch):
    from asil.gui_agent.session import GUISession, start_gui_session

    launched: list[str] = []

    def fake_launch_child(spec, diagnostics, **_kwargs):
        del diagnostics
        launched.append(spec.window_title_pattern)
        return GUISession(spec=spec)

    class FakeSyncPlaywright:
        starts = 0

        def start(self):
            type(self).starts += 1
            return self

        def stop(self):
            pass

    _install_fake_playwright(monkeypatch, lambda: FakeSyncPlaywright())
    monkeypatch.setattr("asil.gui_agent.session._launch_child_session", fake_launch_child)
    monkeypatch.setattr("asil.gui_agent.session.activate_window", lambda *_args, **_kwargs: "123")
    monkeypatch.setattr("asil.gui_agent.session._cleanup_gui_processes", lambda spec: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        child_specs={
            "nautilus": GUISessionSpec(surface_type="desktop", window_title_pattern="Files", launch_command=("nautilus",)),
            "code_server": GUISessionSpec(surface_type="browser", window_title_pattern="Code", launch_command=()),
        },
        primary_child="nautilus",
        capture_active_window=True,
    )

    session = start_gui_session(spec)

    assert launched == ["Code", "Files"]
    assert FakeSyncPlaywright.starts == 0
    session.close()


def test_launch_child_session_closes_session_when_readiness_fails(monkeypatch):
    from asil.gui_agent.session import GUISession, _launch_child_session

    closed: list[str] = []

    class ClosingSession(GUISession):
        def close(self) -> None:
            closed.append(self.spec.window_title_pattern)

    spec = GUISessionSpec(surface_type="browser", window_title_pattern="Jupyter", launch_command=())

    monkeypatch.setattr(
        "asil.gui_agent.session._launch_browser_session_with_diagnostics",
        lambda child_spec, diagnostics: ClosingSession(spec=child_spec, startup_diagnostics=diagnostics),
    )
    monkeypatch.setattr(
        "asil.gui_agent.session._run_post_launch_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not ready")),
    )

    try:
        _launch_child_session(spec, {})
        assert False, "Expected child readiness failure"
    except RuntimeError as exc:
        assert str(exc) == "not ready"

    assert closed == ["Jupyter"]


def test_start_gui_session_resets_virtual_display_before_each_launch_attempt(monkeypatch):
    from asil.gui_agent.session import GUISession, start_gui_session

    call_order: list[str] = []

    monkeypatch.setattr("asil.gui_agent.session._cleanup_gui_processes", lambda spec: call_order.append("cleanup"))
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: call_order.append("reset"))
    monkeypatch.setattr(
        "asil.gui_agent.session._launch_browser_session",
        lambda spec: call_order.append("launch") or GUISession(spec=spec),
    )
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test",
        window_title_pattern=r".*",
        launch_command=(),
    )

    session = start_gui_session(spec)

    assert call_order[:3] == ["cleanup", "reset", "launch"]
    session.close()


def test_gui_session_close_cleans_up_stray_processes(monkeypatch):
    from asil.gui_agent.session import GUISession

    cleaned: list[str] = []
    monkeypatch.setattr("asil.gui_agent.session._cleanup_gui_processes", lambda spec: cleaned.append(spec.surface_type))
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="about:blank",
        window_title_pattern=r".*",
        launch_command=(),
    )

    session = GUISession(spec=spec)
    session.close()

    assert cleaned == ["browser"]


def test_cleanup_gui_processes_matches_exact_process_names(monkeypatch):
    from asil.gui_agent.session import _cleanup_gui_processes

    calls: list[list[str]] = []

    monkeypatch.setattr("asil.gui_agent.session.shutil.which", lambda name: "/usr/bin/pkill" if name == "pkill" else None)
    monkeypatch.setattr("asil.gui_agent.session.subprocess.run", lambda cmd, **kwargs: calls.append(cmd))
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Blender",
        launch_command=("blender",),
    )

    _cleanup_gui_processes(spec)

    assert calls == [
        ["/usr/bin/pkill", "-TERM", "-x", "blender"],
        ["/usr/bin/pkill", "-KILL", "-x", "blender"],
    ]


def test_prelaunch_multi_window_cleanup_includes_desktop_children():
    from asil.gui_agent.session import _prelaunch_cleanup_spec, _session_process_names

    spec = GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=r".*",
        child_specs={
            "nautilus": GUISessionSpec(
                surface_type="desktop",
                window_title_pattern="Files",
                launch_command=("nautilus",),
            ),
            "gitea": GUISessionSpec(
                surface_type="browser",
                window_title_pattern="Gitea",
            ),
        },
    )

    process_names = _session_process_names(_prelaunch_cleanup_spec(spec))

    assert "nautilus" in process_names
    assert "chromium" in process_names


def test_start_gui_session_retries_when_backend_probe_is_temporarily_unready(monkeypatch):
    from asil.gui_agent.session import GUISessionStartupError, start_gui_session

    backend_attempts: list[int] = []
    launch_attempts: list[int] = []

    class FakeProcess:
        def poll(self):
            return None

    def backend_probe():
        backend_attempts.append(len(backend_attempts) + 1)
        if len(backend_attempts) == 1:
            raise GUISessionStartupError("backend_unready", "Service is still starting.")

    def fake_launch(*args, **kwargs):
        del args, kwargs
        launch_attempts.append(len(launch_attempts) + 1)
        return FakeProcess()

    monkeypatch.setattr("asil.gui_agent.session.ensure_virtual_display", lambda *args, **kwargs: {"DISPLAY": ":99"})
    monkeypatch.setattr("asil.gui_agent.session.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.gui_agent.session.terminate_process", lambda proc: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Writer",
        launch_command=("libreoffice", "--writer", "/tmp/doc.odt"),
        backend_ready_probe=backend_probe,
    )

    session = start_gui_session(spec)

    assert backend_attempts == [1, 2]
    assert launch_attempts == [1]
    session.close()


def test_start_gui_session_raises_categorized_backend_unready_error_after_retries(monkeypatch):
    from asil.gui_agent.session import GUISessionStartupError, start_gui_session

    monkeypatch.setattr("asil.gui_agent.session.ensure_virtual_display", lambda *args, **kwargs: {"DISPLAY": ":99"})
    monkeypatch.setattr("asil.gui_agent.session.launch_gui_process", lambda *args, **kwargs: object())
    monkeypatch.setattr("asil.gui_agent.session.terminate_process", lambda proc: None)
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Writer",
        launch_command=("libreoffice", "--writer", "/tmp/doc.odt"),
        backend_ready_probe=lambda: (_ for _ in ()).throw(
            GUISessionStartupError("backend_unready", "Service is still starting.")
        ),
    )

    try:
        start_gui_session(spec)
        assert False, "Expected backend_unready startup failure"
    except GUISessionStartupError as exc:
        assert exc.category == "backend_unready"
        assert "Service is still starting" in str(exc)


def test_start_gui_session_times_out_stuck_post_launch_callback(monkeypatch):
    from asil.gui_agent.session import (
        GUISession,
        GUISessionStartupError,
        _StartupPhaseTimedOut,
        start_gui_session,
    )

    monkeypatch.setattr(
        "asil.gui_agent.session._launch_browser_session",
        lambda spec: GUISession(spec=spec),
    )
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    class TimeoutGuard:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            raise _StartupPhaseTimedOut("post-launch callback timed out.")

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("asil.gui_agent.session._StartupTimeoutGuard", TimeoutGuard)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test",
        window_title_pattern=r".*",
        launch_command=(),
        post_launch_callback=lambda: None,
    )

    try:
        start_gui_session(spec)
        assert False, "Expected stuck post-launch callback to become startup failure"
    except GUISessionStartupError as exc:
        assert exc.category == "window_timeout"


def test_start_gui_session_honors_adapter_startup_timeout_above_sixty_seconds(monkeypatch):
    from asil.gui_agent.session import GUISession, start_gui_session

    observed_timeouts = []

    monkeypatch.setattr(
        "asil.gui_agent.session._launch_browser_session",
        lambda spec: GUISession(spec=spec),
    )
    monkeypatch.setattr("asil.gui_agent.session.stop_virtual_display", lambda: None)
    monkeypatch.setattr("asil.gui_agent.session.time.sleep", lambda *_args, **_kwargs: None)

    class RecordingTimeoutGuard:
        def __init__(self, seconds, message):
            observed_timeouts.append((seconds, message))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("asil.gui_agent.session._StartupTimeoutGuard", RecordingTimeoutGuard)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="about:blank",
        window_title_pattern=r".*",
        startup_timeout_s=120.0,
        post_launch_callback=lambda: None,
        ui_ready_probe=lambda session: None,
    )

    session = start_gui_session(spec)
    session.close()

    assert observed_timeouts == [
        (240.0, "post-launch callback timed out."),
        (240.0, "UI readiness probe timed out."),
    ]


def test_browser_page_failure_category_detects_blank_shell_and_crash_pages():
    from asil.gui_agent.session import _browser_page_failure_category

    class FakePage:
        def __init__(self, *, url: str, body_text: str):
            self.url = url
            self._body_text = body_text

        def content(self) -> str:
            return f"<html><body>{self._body_text}</body></html>"

    assert _browser_page_failure_category(FakePage(url="http://code-server:8080", body_text="")) == "blank_shell"
    assert (
        _browser_page_failure_category(
            FakePage(url="chrome-error://chromewebdata/", body_text="Aw, Snap! Something went wrong")
        )
        == "browser_crashed"
    )

    class CrashedPage:
        url = "http://gitea:3000/user/login"

        def content(self) -> str:
            raise RuntimeError("Page.content: Target crashed")

    assert _browser_page_failure_category(CrashedPage()) == "browser_crashed"


def test_assert_browser_page_ready_allows_transient_blank_shell_before_selectors_appear():
    from asil.gui_agent.session import GUISession, _assert_browser_page_ready

    class FakeLocator:
        def __init__(self, page):
            self._page = page

        def inner_text(self, timeout: int = 0):
            return self._page.body_text

    class FakePage:
        def __init__(self) -> None:
            self.url = "http://code-server:8080"
            self.body_text = ""
            self.waited_for = []

        def content(self) -> str:
            return "<html><body></body></html>"

        def locator(self, selector: str):
            return FakeLocator(self)

        def wait_for_selector(self, selector: str, timeout: int = 0):
            self.waited_for.append((selector, timeout))
            self.body_text = "Workbench loaded"

        def wait_for_function(self, script: str, timeout: int = 0):
            return None

    page = FakePage()
    session = GUISession(
        spec=GUISessionSpec(
            surface_type="browser",
            browser_url="http://code-server:8080",
            window_title_pattern=r".*",
        ),
        browser_page=page,
    )

    _assert_browser_page_ready(
        session,
        required_selectors=(".monaco-workbench",),
        app_name="code-server",
        timeout_ms=1000,
    )


def test_assert_browser_page_ready_uses_document_body_probe_for_body_selector():
    from asil.gui_agent.session import GUISession, _assert_browser_page_ready

    class FakePage:
        def __init__(self) -> None:
            self.url = "http://jupyterlab:8888/lab/tree/notebooks"
            self.function_calls = []
            self.selector_calls = []

        def content(self) -> str:
            return "<html><body>ready</body></html>"

        def locator(self, selector: str):
            class Locator:
                def inner_text(self, timeout: int = 0):
                    del timeout
                    return "ready"

            return Locator()

        def wait_for_function(self, script: str, timeout: int = 0):
            self.function_calls.append((script, timeout))

        def wait_for_selector(self, selector: str, timeout: int = 0):
            self.selector_calls.append((selector, timeout))

    page = FakePage()
    session = GUISession(
        spec=GUISessionSpec(surface_type="browser", browser_url=page.url, window_title_pattern=r".*"),
        browser_page=page,
    )

    _assert_browser_page_ready(
        session,
        required_selectors=("body", ".jp-LabShell"),
        app_name="JupyterLab",
        timeout_ms=1234,
    )

    assert len(page.function_calls) == 1
    assert page.function_calls[0][1] == 1234
    assert "document.body" in page.function_calls[0][0]
    assert "innerText" in page.function_calls[0][0]
    assert (".jp-LabShell", 1234) in page.selector_calls


def test_navigate_browser_target_falls_back_to_current_page_without_context():
    from asil.gui_agent.session import GUISession, navigate_browser_target

    class FakePage:
        def __init__(self) -> None:
            self.url = "http://code-server:8080"
            self.goto_calls = []
            self.load_state_calls = []

        def goto(self, url: str, wait_until=None, timeout=None) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

        def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
            self.load_state_calls.append((state, timeout))

    page = FakePage()
    session = GUISession(
        spec=GUISessionSpec(surface_type="browser", browser_url=page.url, window_title_pattern=r".*"),
        browser_page=page,
    )

    navigate_browser_target(session, "http://code-server:8080/?folder=/tmp/workspace")

    assert page.goto_calls == [("http://code-server:8080/?folder=/tmp/workspace", "commit", 60000)]
    assert page.load_state_calls == [("domcontentloaded", 15000)]


def test_navigate_browser_target_can_use_current_page_even_with_context():
    from asil.gui_agent.session import GUISession, navigate_browser_target

    class FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"
            self.goto_calls = []
            self.load_state_calls = []

        def goto(self, url: str, wait_until=None, timeout=None) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

        def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
            self.load_state_calls.append((state, timeout))

    class FakeContext:
        def __init__(self) -> None:
            self.new_page_calls = 0

        def new_page(self):
            self.new_page_calls += 1
            raise AssertionError("current_page navigation should not allocate a new tab")

    page = FakePage()
    context = FakeContext()
    session = GUISession(
        spec=GUISessionSpec(
            surface_type="browser",
            browser_url="about:blank",
            browser_navigation_mode="current_page",
            window_title_pattern=r".*",
        ),
        browser_page=page,
        browser_context=context,
    )

    navigate_browser_target(session, "http://jupyterlab:8888/lab/tree/notebooks/analysis.ipynb")

    assert page.goto_calls == [("http://jupyterlab:8888/lab/tree/notebooks/analysis.ipynb", "commit", 60000)]
    assert page.load_state_calls == [("domcontentloaded", 15000)]
    assert context.new_page_calls == 0


def test_navigate_browser_target_uses_fresh_page_when_context_is_available():
    from asil.gui_agent.session import GUISession, navigate_browser_target

    class FakePage:
        def __init__(self, url: str) -> None:
            self.url = url
            self.goto_calls = []
            self.load_state_calls = []
            self.closed = False
            self.brought_to_front = False

        def goto(self, url: str, wait_until=None, timeout=None) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

        def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
            self.load_state_calls.append((state, timeout))

        def bring_to_front(self) -> None:
            self.brought_to_front = True

        def close(self) -> None:
            self.closed = True

    class FakeContext:
        def __init__(self) -> None:
            self.created_page = FakePage("about:blank")

        def new_page(self) -> FakePage:
            return self.created_page

    original_page = FakePage("http://drawio:8080/?offline=1&stealth=1")
    context = FakeContext()
    session = GUISession(
        spec=GUISessionSpec(surface_type="browser", browser_url=original_page.url, window_title_pattern=r".*"),
        browser_page=original_page,
        browser_context=context,
    )

    navigate_browser_target(session, "http://drawio:8080/?offline=1&stealth=1#R<mxfile>")

    assert context.created_page.goto_calls == [("http://drawio:8080/?offline=1&stealth=1#R<mxfile>", "commit", 60000)]
    assert context.created_page.load_state_calls == [("domcontentloaded", 15000)]
    assert context.created_page.brought_to_front is True
    assert session.browser_page is context.created_page
    assert original_page.closed is True


def test_launch_browser_session_uses_capture_friendly_window_size(monkeypatch):
    from asil.gui_agent.session import _launch_browser_session

    launch_kwargs: dict[str, object] = {}

    class FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"

        def goto(self, url: str, wait_until=None, timeout=None) -> None:
            self.url = url

        def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
            return None

    class FakeContext:
        def __init__(self) -> None:
            self.pages = []
            self._page = FakePage()

        def new_page(self) -> FakePage:
            self.pages.append(self._page)
            return self._page

        def close(self) -> None:
            return None

    class FakeChromium:
        def launch_persistent_context(self, **kwargs):
            launch_kwargs.update(kwargs)
            return FakeContext()

    class FakePlaywright:
        def __init__(self) -> None:
            self.chromium = FakeChromium()

        def stop(self) -> None:
            return None

    class FakeSyncPlaywright:
        def start(self) -> FakePlaywright:
            return FakePlaywright()

    _install_fake_playwright(monkeypatch, lambda: FakeSyncPlaywright())
    monkeypatch.setattr("asil.gui_agent.session.ensure_virtual_display", lambda run_as_user=None: {"DISPLAY": ":99"})

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test",
        window_title_pattern=r".*",
    )

    session = _launch_browser_session(spec)

    assert launch_kwargs["timeout"] == 45_000
    assert launch_kwargs["viewport"] == {"width": 1280, "height": 800}
    assert "--window-size=1360,900" in launch_kwargs["args"]
    assert "--start-maximized" not in launch_kwargs["args"]
    session.close()


def test_launch_browser_session_uses_robust_target_navigation(monkeypatch):
    from asil.gui_agent.session import _launch_browser_session

    navigate_calls: list[tuple[str, int]] = []

    class FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"
            self.goto_calls = []

        def goto(self, url: str, wait_until=None, timeout=None) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

        def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeContext:
        def __init__(self) -> None:
            self.pages = []
            self._page = FakePage()

        def new_page(self) -> FakePage:
            return self._page

        def close(self) -> None:
            return None

    class FakeChromium:
        def launch_persistent_context(self, **kwargs):
            return FakeContext()

    class FakePlaywright:
        def __init__(self) -> None:
            self.chromium = FakeChromium()

        def stop(self) -> None:
            return None

    class FakeSyncPlaywright:
        def start(self) -> FakePlaywright:
            return FakePlaywright()

    def fake_navigate(session, target_url: str, timeout_ms: int = 0):
        navigate_calls.append((target_url, timeout_ms))
        session.browser_page.url = target_url

    _install_fake_playwright(monkeypatch, lambda: FakeSyncPlaywright())
    monkeypatch.setattr("asil.gui_agent.session.ensure_virtual_display", lambda run_as_user=None: {"DISPLAY": ":99"})
    monkeypatch.setattr("asil.gui_agent.session.navigate_browser_target", fake_navigate)

    spec = GUISessionSpec(
        surface_type="browser",
        browser_url="http://example.test/login",
        window_title_pattern=r".*",
    )

    session = _launch_browser_session(spec)

    assert navigate_calls == [("http://example.test/login", 45000)]
    assert session.browser_page.url == "http://example.test/login"
    session.close()


def test_resolve_gui_session_spec_covers_all_current_benchmark_software(tmp_path: Path):
    from asil.gui_agent.session import resolve_gui_session_spec

    def dummy_adapter(class_name: str, **attrs):
        cls = type(class_name, (), {})
        adapter = cls()
        setattr(adapter, "get_gui_session_spec", lambda: None)
        for key, value in attrs.items():
            setattr(adapter, key, value)
        return adapter

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source.file"
    source.write_text("placeholder", encoding="utf-8")
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"fake")

    adapters = {
        "inkscape": dummy_adapter("InkscapeAdapter", source_path=source),
        "libreoffice": dummy_adapter("LibreOfficeAdapter", source_path=source),
        "blender": dummy_adapter(
            "BlenderAdapter",
            blender_bin="blender",
            blend_path=blend,
            _ensure_workfile=lambda: None,
        ),
        "obs": dummy_adapter("OBSAdapter"),
        "gitea": dummy_adapter(
            "GiteaAdapter",
            base_url="http://gitea:3000",
            _current_ui_path="/asil_admin/test-repo",
        ),
        "gimp": dummy_adapter("GimpAdapter", image_path=image),
        "libreoffice_writer": dummy_adapter("LibreOfficeWriterAdapter", source_path=source),
        "libreoffice_impress": dummy_adapter("LibreOfficeImpressAdapter", source_path=source),
        "code_server": dummy_adapter(
            "CodeServerAdapter",
            base_url="http://code-server:8080",
            workspace_path=workspace,
            _active_file=Path("notes.txt"),
        ),
        "thunderbird": dummy_adapter("ThunderbirdAdapter"),
        "nautilus": dummy_adapter("NautilusAdapter", workspace_path=workspace),
        "kdenlive": dummy_adapter("KdenliveAdapter", source_path=source),
        "audacity": dummy_adapter("AudacityAdapter", source_path=source),
        "drawio": dummy_adapter("DrawioAdapter", _live_editor_url=lambda: "http://drawio:8080/?embed=1"),
        "jupyterlab": dummy_adapter(
            "JupyterLabAdapter",
            base_url="http://jupyterlab:8888",
            _active_file="notebooks/demo.ipynb",
        ),
    }

    specs = {software: resolve_gui_session_spec(adapter) for software, adapter in adapters.items()}

    assert set(specs) == {
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
    assert specs["gitea"].surface_type == "browser"
    assert specs["code_server"].surface_type == "browser"
    assert specs["drawio"].surface_type == "browser"
    assert specs["jupyterlab"].surface_type == "browser"
    assert all(
        specs[name].surface_type == "desktop"
        for name in {
            "inkscape",
            "libreoffice",
            "blender",
            "obs",
            "gimp",
            "libreoffice_writer",
            "libreoffice_impress",
            "thunderbird",
            "nautilus",
            "kdenlive",
            "audacity",
        }
    )


def test_gui_runner_resets_reused_llm_and_syncs_adapter_before_observe(monkeypatch, tmp_path: Path):
    from asil.gui_agent.runner import run_gui_agent_task
    from asil.gui_agent.llm import GUIModelOutput

    sync_calls = []
    reset_calls = []
    reset_counts_seen_by_llm = []

    class FakeAdapter:
        app_name = "FakeApp"

        def get_gui_session_spec(self):
            return GUISessionSpec(
                surface_type="desktop",
                window_title_pattern="Fake",
                launch_command=("fake",),
            )

        def observe(self):
            return _fake_observation()

        def sync_from_gui(self):
            sync_calls.append("sync")

    class FakeSession:
        spec = GUISessionSpec(
            surface_type="desktop",
            window_title_pattern="Fake",
            launch_command=("fake",),
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def capture(self, output_path):
            Path(output_path).write_bytes(b"png")
            return True

    class FakeEval:
        score = 1.0
        success = True

        def to_dict(self):
            return {"score": 1.0, "success": True}

    class FakeTask:
        id = "fake_01"
        description = "Fake task"
        instruction = "Do one thing."
        software = "fake"
        snapshot = "fake"
        validation = {}
        evaluator = {}
        gui_expectations = {}
        render_target = {}
        difficulty = "simple"

    class ResettableLLM:
        def reset(self):
            reset_calls.append("reset")

        def __call__(self, prompt, image_bytes):
            del prompt, image_bytes
            reset_counts_seen_by_llm.append(len(reset_calls))
            return GUIModelOutput(
                text="Thought: stop\nAction: DONE",
                provider="mock",
                model="mock-gui",
            )

    def fake_render_step(session, adapter, task_dir, step_num):
        output = task_dir / f"step_{step_num}.png"
        output.write_bytes(b"png")
        (task_dir / f"step_{step_num}.render.json").write_text(
            json.dumps({"actual_page": True, "capture_complete": True}),
            encoding="utf-8",
        )
        return output.name, True

    monkeypatch.setattr("asil.gui_agent.runner.start_gui_session", lambda spec: FakeSession())
    monkeypatch.setattr("asil.gui_agent.runner._render_step", fake_render_step)
    monkeypatch.setattr("asil.gui_agent.runner._image_size", lambda path: (100, 100))
    monkeypatch.setattr("asil.gui_agent.runner._load_png_bytes", lambda path: b"png")
    monkeypatch.setattr("asil.gui_agent.runner.evaluate_task_result", lambda task, obs: FakeEval())
    monkeypatch.setattr(
        "asil.gui_agent.runner.X11GUIController.execute",
        lambda self, action, spec: None,
    )

    llm_fn = ResettableLLM()
    results = [
        run_gui_agent_task(
            FakeAdapter(),
            FakeTask(),
            llm_fn,
            max_steps=1,
            task_dir=tmp_path / f"fake_task_{run_index}",
        )
        for run_index in range(2)
    ]

    assert reset_calls == ["reset", "reset"]
    assert reset_counts_seen_by_llm == [1, 2]
    assert sync_calls == ["sync", "sync"]
    assert all(result.success for result in results)
    startup_diagnostics = json.loads(
        (tmp_path / "fake_task_0" / "startup_diagnostics.json").read_text(encoding="utf-8")
    )
    assert any(phase["name"] == "step0_capture" for phase in startup_diagnostics["phases"])


def test_gui_runner_activate_app_action_uses_session_and_updates_controller():
    from asil.gui_agent.controller import X11GUIController
    from asil.gui_agent.parser import GUIAction
    from asil.gui_agent.runner import _execute_gui_action

    class FakeSession:
        def activate_app(self, app):
            assert app == "jupyterlab"
            return "0xjupyter"

    controller = X11GUIController()
    _execute_gui_action(
        controller,
        FakeSession(),
        GUIAction(action_type="ACTIVATE_APP", payload={"app": "jupyterlab"}),
        spec=GUISessionSpec(surface_type="multi_window", window_title_pattern=r".*"),
    )

    assert controller.last_capture_window_id == "0xjupyter"


def test_gui_runner_syncs_adapter_with_session_and_persists_before_observe(monkeypatch, tmp_path: Path):
    from asil.adapter import ASILAdapter
    from asil.gui_agent.llm import GUIModelOutput
    from asil.gui_agent.runner import run_gui_agent_task

    call_order: list[str] = []

    class FakeAdapter:
        app_name = "Inkscape"
        gui_eval_mode = ASILAdapter.gui_eval_mode

        def get_gui_session_spec(self):
            return GUISessionSpec(
                surface_type="desktop",
                window_title_pattern="Fake",
                launch_command=("fake",),
                persist_shortcuts=("ctrl+s",),
            )

        def observe(self):
            call_order.append("observe")
            return _fake_observation()

        def sync_from_gui(self, session=None):
            assert session is not None
            call_order.append("sync")

    class FakeSession:
        spec = GUISessionSpec(
            surface_type="desktop",
            window_title_pattern="Fake",
            launch_command=("fake",),
            persist_shortcuts=("ctrl+s",),
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def capture(self, output_path):
            Path(output_path).write_bytes(b"png")
            return True

    class FakeEval:
        score = 1.0
        success = True

        def to_dict(self):
            return {"score": 1.0, "success": True}

    class FakeTask:
        id = "fake_01b"
        description = "Fake task"
        instruction = "Persist before observe."
        software = "fake"
        snapshot = "fake"
        validation = {}
        evaluator = {}
        gui_expectations = {}
        render_target = {}
        difficulty = "simple"

    def fake_render_step(session, adapter, task_dir, step_num):
        output = task_dir / f"step_{step_num}.png"
        output.write_bytes(b"png")
        (task_dir / f"step_{step_num}.render.json").write_text(
            json.dumps({"actual_page": True, "capture_complete": True}),
            encoding="utf-8",
        )
        return output.name, True

    monkeypatch.setattr("asil.gui_agent.runner.start_gui_session", lambda spec: FakeSession())
    monkeypatch.setattr("asil.gui_agent.runner._render_step", fake_render_step)
    monkeypatch.setattr("asil.gui_agent.runner._image_size", lambda path: (100, 100))
    monkeypatch.setattr("asil.gui_agent.runner._load_png_bytes", lambda path: b"png")
    monkeypatch.setattr("asil.gui_agent.runner.evaluate_task_result", lambda task, obs: FakeEval())
    monkeypatch.setattr("asil.gui_agent.runner.X11GUIController.execute", lambda self, action, spec: None)
    monkeypatch.setattr(
        "asil.gui_agent.runner.X11GUIController.persist",
        lambda self, spec: call_order.append("persist"),
    )

    result = run_gui_agent_task(
        FakeAdapter(),
        FakeTask(),
        lambda prompt, image_bytes: GUIModelOutput(
            text="Thought: persist now\nAction: DONE",
            provider="mock",
            model="mock-gui",
        ),
        max_steps=1,
        task_dir=tmp_path / "fake_task_persist",
    )

    assert call_order == ["observe", "persist", "sync", "observe"]
    assert result.success is True


def test_gui_runner_turns_parse_failure_into_task_failure(monkeypatch, tmp_path: Path):
    from asil.gui_agent.runner import run_gui_agent_task
    from asil.gui_agent.llm import GUIModelOutput

    class FakeAdapter:
        app_name = "FakeApp"

        def get_gui_session_spec(self):
            return GUISessionSpec(
                surface_type="desktop",
                window_title_pattern="Fake",
                launch_command=("fake",),
            )

        def observe(self):
            return _fake_observation()

        def sync_from_gui(self):
            return None

    class FakeSession:
        spec = GUISessionSpec(
            surface_type="desktop",
            window_title_pattern="Fake",
            launch_command=("fake",),
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def capture(self, output_path):
            Path(output_path).write_bytes(b"png")
            return True

    class FakeEval:
        score = 0.0
        success = False

        def to_dict(self):
            return {"score": 0.0, "success": False}

    class FakeTask:
        id = "fake_02"
        description = "Fake task"
        instruction = "Do one thing."
        software = "fake"
        snapshot = "fake"
        validation = {}
        evaluator = {}
        gui_expectations = {}
        render_target = {}
        difficulty = "simple"

    def fake_render_step(session, adapter, task_dir, step_num):
        output = task_dir / f"step_{step_num}.png"
        output.write_bytes(b"png")
        (task_dir / f"step_{step_num}.render.json").write_text(
            json.dumps({"actual_page": True, "capture_complete": True}),
            encoding="utf-8",
        )
        return output.name, True

    monkeypatch.setattr("asil.gui_agent.runner.start_gui_session", lambda spec: FakeSession())
    monkeypatch.setattr("asil.gui_agent.runner._render_step", fake_render_step)
    monkeypatch.setattr("asil.gui_agent.runner._image_size", lambda path: (100, 100))
    monkeypatch.setattr("asil.gui_agent.runner._load_png_bytes", lambda path: b"png")
    monkeypatch.setattr("asil.gui_agent.runner.evaluate_task_result", lambda task, obs: FakeEval())
    monkeypatch.setattr(
        "asil.gui_agent.runner.parse_gui_response",
        lambda text: (_ for _ in ()).throw(ValueError("No executable pyautogui action found in response.")),
    )

    result = run_gui_agent_task(
        FakeAdapter(),
        FakeTask(),
        lambda prompt, image_bytes: GUIModelOutput(
            text="I cannot comply",
            provider="mock",
            model="mock-gui",
        ),
        max_steps=3,
        task_dir=tmp_path / "fake_task_parse_failure",
    )

    assert result.success is False
    assert result.score == 0.0
    assert result.steps == 1
    assert result.step_results[0].action_type == "FAIL"
    assert result.step_results[0].success is False
    assert (tmp_path / "fake_task_parse_failure" / "result.txt").read_text(encoding="utf-8") == "0.0"


def test_gui_runner_turns_model_timeout_into_fail_action(monkeypatch, tmp_path: Path):
    from threading import Event

    from asil.gui_agent.llm import GUIModelOutput
    from asil.gui_agent.runner import run_gui_agent_task

    class FakeAdapter:
        app_name = "FakeTimeoutApp"

        def get_gui_session_spec(self):
            return GUISessionSpec(
                surface_type="desktop",
                window_title_pattern="Fake",
                launch_command=("fake",),
            )

        def observe(self):
            return _fake_observation()

        def sync_from_gui(self):
            return None

    class FakeSession:
        spec = GUISessionSpec(
            surface_type="desktop",
            window_title_pattern="Fake",
            launch_command=("fake",),
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def capture(self, output_path):
            Path(output_path).write_bytes(b"png")
            return True

    class FakeEval:
        score = 0.0
        success = False

        def to_dict(self):
            return {"score": 0.0, "success": False}

    class FakeTask:
        id = "fake_timeout_01"
        description = "Fake timeout task"
        instruction = "Wait for a model that never returns."
        software = "fake"
        snapshot = "fake"
        validation = {}
        evaluator = {}
        gui_expectations = {}
        render_target = {}
        difficulty = "simple"

    def fake_render_step(session, adapter, task_dir, step_num):
        output = task_dir / f"step_{step_num}.png"
        output.write_bytes(b"png")
        (task_dir / f"step_{step_num}.render.json").write_text(
            json.dumps({"actual_page": True, "capture_complete": True}),
            encoding="utf-8",
        )
        return output.name, True

    release_model = Event()

    class SlowLLM:
        def __init__(self):
            self.cancel_calls = 0

        def __call__(self, prompt, image_bytes):
            del prompt, image_bytes
            release_model.wait(timeout=5.0)
            return GUIModelOutput(text="Thought: too late\nAction: DONE", provider="mock", model="slow")

        def cancel_pending(self):
            self.cancel_calls += 1
            release_model.set()

    slow_llm = SlowLLM()

    monkeypatch.setenv("ASIL_GUI_LLM_CALL_TIMEOUT_S", "0.01")
    monkeypatch.setattr("asil.gui_agent.runner.start_gui_session", lambda spec: FakeSession())
    monkeypatch.setattr("asil.gui_agent.runner._render_step", fake_render_step)
    monkeypatch.setattr("asil.gui_agent.runner._image_size", lambda path: (100, 100))
    monkeypatch.setattr("asil.gui_agent.runner._load_png_bytes", lambda path: b"png")
    monkeypatch.setattr("asil.gui_agent.runner.evaluate_task_result", lambda task, obs: FakeEval())
    monkeypatch.setattr("asil.gui_agent.runner.X11GUIController.execute", lambda self, action, spec: None)

    result = run_gui_agent_task(
        FakeAdapter(),
        FakeTask(),
        slow_llm,
        max_steps=1,
        task_dir=tmp_path / "fake_task_model_timeout",
    )
    release_model.set()

    assert result.success is False
    assert result.steps == 1
    assert result.step_results[0].action_type == "FAIL"
    action_payload = json.loads(
        (tmp_path / "fake_task_model_timeout" / "step_1_action.json").read_text(encoding="utf-8")
    )
    assert action_payload["action"]["action_type"] == "FAIL"
    assert action_payload["provider"] == "timeout_guard"
    assert "model request exceeded" in action_payload["raw_text"]
    assert slow_llm.cancel_calls == 1


def test_gui_runner_does_not_deadlock_when_same_ids_keep_changing_value(monkeypatch, tmp_path: Path):
    from asil.gui_agent.runner import run_gui_agent_task
    from asil.gui_agent.llm import GUIModelOutput
    from asil.protocol import Element

    class FakeAdapter:
        app_name = "FakeApp"

        def __init__(self):
            self.observe_index = 0
            self.observations = [
                Observation.model_validate(
                    {
                        **_fake_observation().model_dump(),
                        "interactive_elements": [
                            Element(id="shape-1", type="rect", value={"fill": f"#{idx}"}, metadata={"fill": f"#{idx}"}).model_dump()
                        ],
                    }
                )
                for idx in range(8)
            ]

        def get_gui_session_spec(self):
            return GUISessionSpec(
                surface_type="desktop",
                window_title_pattern="Fake",
                launch_command=("fake",),
            )

        def observe(self):
            obs = self.observations[min(self.observe_index, len(self.observations) - 1)]
            self.observe_index += 1
            return obs

        def sync_from_gui(self):
            return None

    class FakeSession:
        spec = GUISessionSpec(
            surface_type="desktop",
            window_title_pattern="Fake",
            launch_command=("fake",),
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def capture(self, output_path):
            Path(output_path).write_bytes(b"png")
            return True

    class FakeEval:
        score = 0.0
        success = False

        def to_dict(self):
            return {"score": 0.0, "success": False}

    class FakeTask:
        id = "fake_03"
        description = "Fake task"
        instruction = "Keep editing."
        software = "fake"
        snapshot = "fake"
        validation = {}
        evaluator = {}
        gui_expectations = {}
        render_target = {}
        difficulty = "simple"

    def fake_render_step(session, adapter, task_dir, step_num):
        output = task_dir / f"step_{step_num}.png"
        output.write_bytes(b"png")
        (task_dir / f"step_{step_num}.render.json").write_text(
            json.dumps({"actual_page": True, "capture_complete": True}),
            encoding="utf-8",
        )
        return output.name, True

    monkeypatch.setattr("asil.gui_agent.runner.start_gui_session", lambda spec: FakeSession())
    monkeypatch.setattr("asil.gui_agent.runner._render_step", fake_render_step)
    monkeypatch.setattr("asil.gui_agent.runner._image_size", lambda path: (100, 100))
    monkeypatch.setattr("asil.gui_agent.runner._load_png_bytes", lambda path: b"png")
    monkeypatch.setattr("asil.gui_agent.runner.evaluate_task_result", lambda task, obs: FakeEval())
    monkeypatch.setattr(
        "asil.gui_agent.runner.X11GUIController.execute",
        lambda self, action, spec: None,
    )

    result = run_gui_agent_task(
        FakeAdapter(),
        FakeTask(),
        lambda prompt, image_bytes: GUIModelOutput(
            text="Thought: keep editing\nAction: WAIT",
            provider="mock",
            model="mock-gui",
        ),
        max_steps=7,
        task_dir=tmp_path / "fake_task_progress",
    )

    assert result.deadlocked is False
    assert result.steps == 7


def test_gui_runner_does_not_deadlock_when_screenshot_changes_but_state_does_not(monkeypatch, tmp_path: Path):
    from asil.gui_agent.runner import run_gui_agent_task
    from asil.gui_agent.llm import GUIModelOutput

    class FakeAdapter:
        app_name = "JupyterLab"

        def get_gui_session_spec(self):
            return GUISessionSpec(
                surface_type="browser",
                browser_url="http://example.test",
                window_title_pattern=r".*",
                launch_command=(),
            )

        def observe(self):
            return _fake_observation()

        def sync_from_gui(self):
            return None

    class FakeSession:
        spec = GUISessionSpec(
            surface_type="browser",
            browser_url="http://example.test",
            window_title_pattern=r".*",
            launch_command=(),
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def capture(self, output_path):
            Path(output_path).write_bytes(b"png")
            return True

    class FakeEval:
        score = 0.0
        success = False

        def to_dict(self):
            return {"score": 0.0, "success": False}

    class FakeTask:
        id = "fake_04"
        description = "Fake task"
        instruction = "Keep navigating."
        software = "jupyterlab"
        snapshot = "fake"
        validation = {}
        evaluator = {}
        gui_expectations = {}
        render_target = {}
        difficulty = "simple"

    def fake_render_step(session, adapter, task_dir, step_num):
        output = task_dir / f"step_{step_num}.png"
        output.write_bytes(f"png-{step_num}".encode("utf-8"))
        (task_dir / f"step_{step_num}.render.json").write_text(
            json.dumps({"actual_page": True, "capture_complete": True}),
            encoding="utf-8",
        )
        return output.name, True

    monkeypatch.setattr("asil.gui_agent.runner.start_gui_session", lambda spec: FakeSession())
    monkeypatch.setattr("asil.gui_agent.runner._render_step", fake_render_step)
    monkeypatch.setattr("asil.gui_agent.runner._image_size", lambda path: (100, 100))
    monkeypatch.setattr("asil.gui_agent.runner._load_png_bytes", lambda path: b"png")
    monkeypatch.setattr("asil.gui_agent.runner.evaluate_task_result", lambda task, obs: FakeEval())
    monkeypatch.setattr(
        "asil.gui_agent.runner.X11GUIController.execute",
        lambda self, action, spec: None,
    )

    result = run_gui_agent_task(
        FakeAdapter(),
        FakeTask(),
        lambda prompt, image_bytes: GUIModelOutput(
            text="Thought: keep navigating\nAction: WAIT",
            provider="mock",
            model="mock-gui",
        ),
        max_steps=7,
        task_dir=tmp_path / "fake_task_visual_progress",
    )

    assert result.deadlocked is False
    assert result.steps == 7


def test_render_step_turns_initial_capture_failure_into_startup_failure(tmp_path: Path):
    from asil.gui_agent.runner import _render_step
    from asil.gui_agent.session import GUISessionStartupError

    class FakeAdapter:
        app_name = "JupyterLab"

    class FakeSession:
        def capture(self, output_path):
            del output_path
            raise TimeoutError("window capture stalled")

    try:
        _render_step(FakeSession(), FakeAdapter(), tmp_path, 0)
        assert False, "Expected step_0 capture failure to become startup failure"
    except GUISessionStartupError as exc:
        assert exc.category == "window_timeout"
        assert "step_0 capture" in str(exc)


def test_initialization_watchdog_uses_forgiving_default_budget(monkeypatch):
    from asil.gui_agent.runner import _initialization_watchdog_timeout_s

    monkeypatch.delenv("ASIL_GUI_INIT_WATCHDOG_S", raising=False)

    desktop_spec = GUISessionSpec(
        surface_type="desktop",
        window_title_pattern="Fake",
        startup_timeout_s=45.0,
    )
    browser_spec = GUISessionSpec(
        surface_type="browser",
        window_title_pattern=r".*",
        startup_timeout_s=120.0,
    )

    assert _initialization_watchdog_timeout_s(desktop_spec) == 240.0
    assert _initialization_watchdog_timeout_s(browser_spec) == 600.0

    monkeypatch.setenv("ASIL_GUI_INIT_WATCHDOG_S", "42")
    assert _initialization_watchdog_timeout_s(browser_spec) == 42.0


def test_initialization_watchdog_writes_runtime_error_without_result(tmp_path: Path):
    from threading import Event

    from asil.gui_agent.runner import _TaskInitializationWatchdog

    timeout_seen = Event()

    class FakeAdapter:
        app_name = "FakeApp"

    class FakeTask:
        id = "fake_99"
        software = "fake"

    spec = GUISessionSpec(
        surface_type="browser",
        window_title_pattern=r".*",
        startup_timeout_s=45.0,
    )
    task_dir = tmp_path / "fake_99"

    with _TaskInitializationWatchdog(
        task_dir=task_dir,
        task=FakeTask(),
        adapter=FakeAdapter(),
        spec=spec,
        timeout_s=0.02,
        startup_diagnostics={"phases": [], "snapshots": []},
        on_timeout=timeout_seen.set,
        poll_interval_s=0.005,
    ):
        assert timeout_seen.wait(1.0)

    runtime_error = task_dir / "runtime_error.txt"
    assert runtime_error.exists()
    assert "startup_watchdog_timeout" in runtime_error.read_text(encoding="utf-8")
    assert (task_dir / "initialization_watchdog.json").exists()
    startup_diagnostics = json.loads((task_dir / "startup_diagnostics.json").read_text(encoding="utf-8"))
    assert startup_diagnostics["phases"][-1]["name"] == "initialization_watchdog"
    assert not (task_dir / "result.txt").exists()
