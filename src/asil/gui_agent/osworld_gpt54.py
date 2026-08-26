"""OSWorld GPT-5.4 computer-use adapter for the ASIL GUI runner.

This module ports the local OSWorld implementation in
`OSWorld/mm_agents/openai_cua_agent.py` into ASIL's screenshot + X11 controller loop.
It keeps the Responses API computer tool contract while returning ASIL-native
`GUIAction` objects so the existing benchmark environment, screenshots,
validators, and result layout can be reused.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import threading
import time
from datetime import datetime
from typing import Any

from asil.gui_agent.llm import GUIModelOutput
from asil.gui_agent.parser import GUIAction


logger = logging.getLogger(__name__)

OPERATOR_PROMPT = """

Here are some helpful tips:
- You are operating an {PLATFORM} desktop with internet access.
- My computer password is "{CLIENT_PASSWORD}" when sudo is needed.
- The current date is {CURRENT_DATE}.
- The home directory is "{HOME_DIR}".
- Stick to the website or application already opened for the task when possible.
- Prefer Chrome over Firefox/Chromium unless the task says otherwise.
- You can act without asking for confirmation.
- If content may be off-screen, scroll or zoom out before deciding it is unavailable.
- When possible, bundle multiple GUI actions into one computer-use turn.
- If the task is infeasible because of missing apps, permissions, contradictory requirements, or other hard blockers, output exactly "[INFEASIBLE]".
"""


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _model_dump(item) for key, item in value.items()}
    return value


def _get_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _retry_delay_s(attempt: int) -> float:
    base = _env_float("ASIL_GUI_LLM_RETRY_BASE_S", 2.0)
    cap = _env_float("ASIL_GUI_LLM_RETRY_CAP_S", 5.0)
    jitter = _env_float("ASIL_GUI_LLM_RETRY_JITTER_S", 0.0)
    delay = min(cap, base * (2 ** attempt))
    if jitter > 0:
        delay += random.uniform(0.0, jitter)
    return delay


def _png_data_url(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def _message_text(item: Any) -> str:
    content = _get_field(item, "content", [])
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if _get_field(part, "type") == "output_text":
                text = _get_field(part, "text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def _reasoning_text(item: Any) -> str:
    summary = _get_field(item, "summary", [])
    if isinstance(summary, list):
        parts: list[str] = []
        for part in summary:
            text = _get_field(part, "text", "")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(summary or "")


def _action_to_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        action_type = action.get("type")
        action_args = {key: _model_dump(value) for key, value in action.items() if key != "type"}
        return {"type": action_type, "args": action_args}

    if hasattr(action, "model_dump"):
        raw = action.model_dump()
        action_type = raw.get("type")
        action_args = {key: _model_dump(value) for key, value in raw.items() if key != "type"}
        return {"type": action_type, "args": action_args}

    action_type = getattr(action, "type", None)
    action_args: dict[str, Any] = {}
    for attr in dir(action):
        if attr.startswith("_") or attr == "type":
            continue
        try:
            action_args[attr] = _model_dump(getattr(action, attr))
        except Exception:
            continue
    return {"type": action_type, "args": action_args}


def _point_xy(point: Any) -> tuple[Any, Any]:
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return point[0], point[1]
    if isinstance(point, dict):
        return point.get("x"), point.get("y")
    return getattr(point, "x", None), getattr(point, "y", None)


def _keypress_actions(keys: Any) -> list[GUIAction]:
    if not keys:
        return []
    if not isinstance(keys, (list, tuple)):
        keys = [keys]

    key_mapping = {
        "alt": "alt",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "arrowup": "up",
        "backspace": "backspace",
        "capslock": "capslock",
        "cmd": "command",
        "command": "command",
        "ctrl": "ctrl",
        "delete": "delete",
        "end": "end",
        "enter": "enter",
        "esc": "esc",
        "home": "home",
        "insert": "insert",
        "option": "option",
        "pagedown": "pagedown",
        "pageup": "pageup",
        "shift": "shift",
        "space": "space",
        "super": "super",
        "tab": "tab",
        "win": "win",
    }
    mapped = [key_mapping.get(str(key).lower(), str(key).lower()) for key in keys]
    if len(mapped) == 1:
        return [GUIAction("PRESS", {"key": mapped[0]})]
    return [GUIAction("HOTKEY", {"keys": mapped})]


def computer_action_to_gui_actions(action_type: str, args: dict[str, Any]) -> list[GUIAction]:
    """Map Responses API computer actions to ASIL GUI actions."""

    if not action_type:
        return []

    if action_type == "click":
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return []
        button = str(args.get("button", "left") or "left").lower()
        if button not in {"left", "middle", "right"}:
            button = "left"
        return [GUIAction("CLICK", {"x": x, "y": y, "button": button})]

    if action_type == "double_click":
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return []
        return [GUIAction("DOUBLE_CLICK", {"x": x, "y": y})]

    if action_type == "move":
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return []
        return [GUIAction("MOVE_TO", {"x": x, "y": y})]

    if action_type == "drag":
        path = args.get("path")
        if not path and args.get("from") and args.get("to"):
            path = [args["from"], args["to"]]
        if not path or len(path) < 2:
            return []
        first_x, first_y = _point_xy(path[0])
        if first_x is None or first_y is None:
            return []
        actions = [
            GUIAction("MOVE_TO", {"x": first_x, "y": first_y}),
            GUIAction("MOUSE_DOWN", {"button": "left"}),
        ]
        for point in path[1:]:
            x, y = _point_xy(point)
            if x is None or y is None:
                return []
            actions.append(GUIAction("MOVE_TO", {"x": x, "y": y}))
        actions.append(GUIAction("MOUSE_UP", {"button": "left"}))
        return actions

    if action_type == "type":
        text = str(args.get("text", ""))
        if text == "":
            return [GUIAction("WAIT", {"seconds": 0.1})]
        if text.isascii() and "\n" not in text:
            return [GUIAction("TYPING", {"text": text})]
        return [GUIAction("CLIPBOARD_PASTE", {"text": text})]

    if action_type == "keypress":
        return _keypress_actions(args.get("keys") or args.get("key"))

    if action_type == "scroll":
        scroll_x = int(args.get("scroll_x") or args.get("delta_x") or args.get("deltaX") or 0)
        scroll_y = int(args.get("scroll_y") or args.get("delta_y") or args.get("deltaY") or 0)
        payload: dict[str, Any] = {"dx": -scroll_x, "dy": -scroll_y}
        if args.get("x") is not None and args.get("y") is not None:
            payload["x"] = args["x"]
            payload["y"] = args["y"]
        return [GUIAction("SCROLL", payload)]

    if action_type == "wait":
        seconds = max(0.1, float(args.get("ms", 1000)) / 1000.0)
        return [GUIAction("WAIT", {"seconds": seconds})]

    if action_type == "screenshot":
        return [GUIAction("WAIT", {"seconds": 0.1})]

    logger.warning("Unsupported GPT-5.4 computer action: %s", action_type)
    return []


class OSWorldGPT54ComputerLLM:
    """Stateful Responses computer-use backend compatible with GUILLMFunction."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        api_key: str = "",
        base_url: str = "",
        timeout_s: float = 180.0,
        reasoning_effort: str = "xhigh",
        client_password: str = "",
        platform: str = "ubuntu",
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model or "gpt-5.4"
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.client_password = client_password or os.environ.get("OSWORLD_CLIENT_PASSWORD", "password")
        self.platform = platform
        self.max_output_tokens = max_output_tokens
        self.cua_messages: list[Any] = []
        self.pending_call_id: str = ""
        self.pending_safety_checks: list[Any] = []
        self._state_lock = threading.Lock()
        self._generation = 0

    def reset(self) -> None:
        self.cancel_pending()

    def cancel_pending(self) -> None:
        with self._state_lock:
            self._generation += 1
            self.cua_messages = []
            self.pending_call_id = ""
            self.pending_safety_checks = []

    def _generation_is_current(self, generation: int) -> bool:
        with self._state_lock:
            return generation == self._generation

    def _instructions(self) -> str:
        home_dir = "C:\\Users\\user" if self.platform.lower().startswith("win") else "/home/user"
        return OPERATOR_PROMPT.format(
            CLIENT_PASSWORD=self.client_password,
            CURRENT_DATE=datetime.now().strftime("%A, %B %d, %Y"),
            HOME_DIR=home_dir,
            PLATFORM=self.platform,
        )

    def _use_gateway(self) -> bool:
        """The DashScope compatible-mode gateway serves the GPT Responses backend
        (hosted computer tool + reasoning) at the /chat/completions path for the
        `mr.gpt-*-responses` model ids; the OpenAI SDK's /responses path 404s there.
        Post the Responses body over raw HTTP in that case."""
        if os.environ.get("ASIL_GUI_GATEWAY_RESPONSES", "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
        return self.model.strip().lower().endswith("-responses")

    def _create_response_via_gateway(self, request: dict[str, Any]) -> Any:
        import json as _json
        import urllib.request

        base = (self.base_url or os.environ.get("OPENAI_BASE_URL", "")
                or os.environ.get("OPENAI_API_BASE", "")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # routify (and the public OpenAI API) serve the Responses body at /responses;
        # only the legacy DashScope compat gateway takes it at /chat/completions.
        path = "/chat/completions" if "compatible-mode" in base else "/responses"
        url = base.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self.api_key or os.environ.get('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=_json.dumps(request).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return _json.loads(resp.read().decode("utf-8"))

    def _create_response(self, request_input: list[Any]) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": self._instructions(),
            "input": request_input,
            "tools": [{"type": "computer"}],
            "parallel_tool_calls": False,
            "reasoning": {"effort": self.reasoning_effort, "summary": "concise"},
            "truncation": "auto",
        }
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        if self._use_gateway():
            return self._create_response_via_gateway(request)

        import openai

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key or os.environ.get("OPENAI_API_KEY", ""),
            "timeout": self.timeout_s,
            # This backend owns the retry policy so it can refresh screenshots
            # and stop retrying when the runner cancels a timed-out generation.
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**client_kwargs)
        return client.responses.create(**request)

    def _request_input(self, prompt: str, screenshot_bytes: bytes) -> list[Any]:
        if not self.cua_messages:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": _png_data_url(screenshot_bytes), "detail": "original"},
                        {"type": "input_text", "text": "\n        " + prompt + self._instructions()},
                    ],
                }
            ]

        if self.pending_call_id:
            return [
                *self.cua_messages,
                {
                    "type": "computer_call_output",
                    "call_id": self.pending_call_id,
                    "acknowledged_safety_checks": self.pending_safety_checks,
                    "output": {
                        "type": "input_image",
                        "image_url": _png_data_url(screenshot_bytes),
                        "detail": "original",
                    },
                },
            ]

        return [
            *self.cua_messages,
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": _png_data_url(screenshot_bytes), "detail": "original"},
                    {"type": "input_text", "text": "Continue from the latest screenshot."},
                ],
            },
        ]

    def _refresh_request_screenshot(self, request_input: list[Any], screenshot_bytes: bytes) -> None:
        if not request_input:
            return
        image_url = _png_data_url(screenshot_bytes)
        last_message = request_input[-1]
        if isinstance(last_message, dict) and "output" in last_message:
            output = last_message.get("output")
            if isinstance(output, dict):
                output["image_url"] = image_url
            return
        content = _get_field(last_message, "content", [])
        if isinstance(content, list):
            for part in content:
                if _get_field(part, "type") == "input_image":
                    if isinstance(part, dict):
                        part["image_url"] = image_url
                    else:
                        setattr(part, "image_url", image_url)
                    break

    def __call__(self, prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
        with self._state_lock:
            request_generation = self._generation
            request_input = self._request_input(prompt, screenshot_bytes)
        last_error: Exception | None = None
        max_attempts = _env_int("ASIL_GUI_LLM_RETRIES", 20)
        for attempt in range(max_attempts):
            if not self._generation_is_current(request_generation):
                raise RuntimeError("OpenAI GPT-5.4 computer-use call cancelled after task reset")
            try:
                response = self._create_response(request_input)
                break
            except Exception as exc:
                last_error = exc
                if not self._generation_is_current(request_generation):
                    raise RuntimeError(
                        "OpenAI GPT-5.4 computer-use call cancelled after task reset"
                    ) from exc
                if attempt + 1 >= max_attempts:
                    logger.error("OpenAI GPT-5.4 computer-use call failed: %s", exc)
                    continue
                delay_s = _retry_delay_s(attempt)
                logger.error(
                    "OpenAI GPT-5.4 computer-use call failed on attempt %d/%d; retrying in %.1fs: %s",
                    attempt + 1,
                    max_attempts,
                    delay_s,
                    exc,
                )
                self._refresh_request_screenshot(request_input, screenshot_bytes)
                time.sleep(delay_s)
        else:
            raise RuntimeError(f"OpenAI GPT-5.4 computer-use call failed too many times: {last_error}")

        output_items = _get_field(response, "output", []) or []

        actions: list[GUIAction] = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        next_call_id = ""
        next_safety_checks: list[Any] = []
        unsupported: list[str] = []

        for item in output_items:
            item_type = _get_field(item, "type")
            if item_type == "message":
                message = _message_text(item)
                if message:
                    text_parts.append(message)
            elif item_type == "reasoning":
                reasoning = _reasoning_text(item)
                if reasoning:
                    reasoning_parts.append(reasoning)
            elif item_type == "computer_call":
                next_call_id = str(_get_field(item, "call_id", "") or "")
                next_safety_checks = _model_dump(_get_field(item, "pending_safety_checks", []) or [])
                raw_actions = _get_field(item, "actions", None)
                if raw_actions is None:
                    raw_action = _get_field(item, "action", None)
                    raw_actions = [raw_action] if raw_action is not None else []
                for raw_action in raw_actions:
                    action_info = _action_to_dict(raw_action)
                    converted = computer_action_to_gui_actions(
                        str(action_info.get("type") or ""),
                        dict(action_info.get("args") or {}),
                    )
                    if converted:
                        actions.extend(converted)
                    else:
                        unsupported.append(str(action_info.get("type") or "unknown"))

        # A runner timeout cannot cancel the request thread. If a later task has
        # reset this backend while the request was still in flight, return the
        # stale output to its original caller without restoring the old task's
        # conversation over the new generation.
        with self._state_lock:
            if request_generation == self._generation:
                self.cua_messages = [*request_input, *output_items]
                self.pending_call_id = next_call_id
                self.pending_safety_checks = next_safety_checks
        if unsupported:
            text_parts.append("Unsupported computer action(s): " + ", ".join(unsupported))

        text = "\n".join(part for part in text_parts if part).strip()
        if not text and actions:
            text = "Thought: Execute the computer-use action batch."
        if unsupported and not actions:
            text = (text + "\nAction: FAIL").strip()

        return GUIModelOutput(
            text=text,
            reasoning_summary="\n".join(reasoning_parts).strip(),
            provider="openai-osworld-gpt54",
            model=self.model,
            actions=tuple(actions),
            metadata={
                "backend": "osworld_gpt54",
                "computer_tool": "computer",
                "reasoning_effort": self.reasoning_effort,
            },
        )

def should_use_osworld_gpt54_backend(provider: str, model: str) -> bool:
    backend = os.environ.get("ASIL_GUI_AGENT_BACKEND", "").strip().lower()
    if backend in {"legacy", "text", "textual", "prompt"}:
        return False
    if backend in {"osworld_gpt54", "gpt54", "computer", "computer_use"}:
        return True
    normalized_model = (model or "").lower().replace("_", "-")
    return provider == "openai" and (
        normalized_model.startswith("gpt-5.4")
        or normalized_model.startswith("gpt54")
    )


def create_osworld_gpt54_llm_fn(
    *,
    model: str,
    api_key: str = "",
    base_url: str = "",
    timeout_s: float = 180.0,
    reasoning_effort: str = "xhigh",
) -> OSWorldGPT54ComputerLLM:
    return OSWorldGPT54ComputerLLM(
        model=model or "gpt-5.4",
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        reasoning_effort=reasoning_effort,
    )
