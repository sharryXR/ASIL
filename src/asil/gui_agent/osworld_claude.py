"""Anthropic computer-use GUI backend for the ASIL GUI runner, faithful to the
ScienceBoard bench claude agent (mm_agents/anthropic).

It drives `mr.claude-sonnet-4-6-*` through the shared DashScope compatible-mode
gateway using the native-protocol passthrough (dashscope_extend_params.using_
native_protocol) with the hosted computer_20251124 tool + computer-use beta.
It keeps the full multi-turn Anthropic message history (tool_use/tool_result),
JPEG-compresses screenshots, and trims all but the N most recent tool_result
images to stay under the gateway's ~6 MiB request-body limit on long runs —
mirroring ScienceBoard's `_maybe_filter_to_n_most_recent_images`.

Computer tool_use actions are translated to ASIL GUIActions with the same
semantics as ScienceBoard's `parse_actions_from_tool_call`.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import threading
import time
import urllib.request
from typing import Any

from asil.gui_agent.llm import GUIModelOutput
from asil.gui_agent.parser import GUIAction

logger = logging.getLogger(__name__)

GATEWAY_DEFAULT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
COMPUTER_TOOL_TYPE = "computer_20251124"
COMPUTER_USE_BETA = "computer-use-2025-11-24"

_KEY_CONV = {"page_down": "pagedown", "page_up": "pageup", "super_l": "win", "super": "command", "escape": "esc"}

SYSTEM_PROMPT = (
    "You are a GUI agent operating a real Linux desktop application via the computer tool. "
    "You see a screenshot of the active application window each turn. Coordinates are pixels "
    "relative to that screenshot (top-left origin). Take exactly one computer action per turn "
    "toward completing the task, and prefer actions that change the saved document/application "
    "state. If the task is complete, say so and stop. If it is impossible, reply with [INFEASIBLE]."
)


def _jpeg_b64(png_bytes: bytes, max_w: int = 1280) -> tuple[str, int, int]:
    """Resize (cap width) + JPEG-compress a screenshot to keep the request body small."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        if w > max_w:
            h = int(h * max_w / w)
            w = max_w
            img = img.resize((w, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii"), w, h
    except Exception:  # pragma: no cover - PIL should be present in the eval image
        return base64.b64encode(png_bytes).decode("ascii"), 1280, 720


def _filter_recent_images(messages: list[dict[str, Any]], keep: int) -> None:
    """Remove all but the last `keep` tool_result screenshots in place (ScienceBoard)."""
    if keep is None:
        return
    tool_results = [
        item
        for m in messages
        for item in (m["content"] if isinstance(m.get("content"), list) else [])
        if isinstance(item, dict) and item.get("type") == "tool_result"
    ]
    total = sum(
        1
        for tr in tool_results
        for c in (tr.get("content") or [])
        if isinstance(c, dict) and c.get("type") == "image"
    )
    to_remove = total - keep
    if to_remove <= 0:
        return
    for tr in tool_results:
        if not isinstance(tr.get("content"), list):
            continue
        new_content = []
        for c in tr["content"]:
            if isinstance(c, dict) and c.get("type") == "image" and to_remove > 0:
                to_remove -= 1
                continue
            new_content.append(c)
        tr["content"] = new_content


def _mods(text: str | None) -> list[str]:
    if not text:
        return []
    return [_KEY_CONV.get(k.strip().lower(), k.strip().lower()) for k in text.split("+")]


def anthropic_action_to_gui_actions(inp: dict[str, Any]) -> list[GUIAction]:
    """Translate a computer tool_use action to ASIL GUIActions (ScienceBoard semantics)."""
    action = inp.get("action")
    action = {"left click": "left_click", "right click": "right_click"}.get(action, action)
    text = inp.get("text")
    coord = inp.get("coordinate")
    start = inp.get("start_coordinate")
    sdir = inp.get("scroll_direction")
    samt = int(inp.get("scroll_amount") or 3)
    x = y = None
    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        x, y = int(coord[0]), int(coord[1])

    if action in ("screenshot", "cursor_position", "wait"):
        return [GUIAction("WAIT", {"seconds": 0.5})]
    if action == "left_mouse_down":
        return [GUIAction("MOUSE_DOWN", {"button": "left"})]
    if action == "left_mouse_up":
        return [GUIAction("MOUSE_UP", {"button": "left"})]
    if action == "mouse_move" and x is not None:
        return [GUIAction("MOVE_TO", {"x": x, "y": y})]
    if action == "left_click_drag" and x is not None:
        acts = []
        if isinstance(start, (list, tuple)) and len(start) == 2:
            acts.append(GUIAction("MOVE_TO", {"x": int(start[0]), "y": int(start[1])}))
        acts.append(GUIAction("DRAG_TO", {"x": x, "y": y}))
        return acts
    if action in ("key", "hold_key") and text:
        keys = _mods(text)
        if len(keys) == 1:
            return [GUIAction("PRESS", {"key": keys[0]})]
        return [GUIAction("HOTKEY", {"keys": keys})]
    if action == "type" and text is not None:
        if text and (not text.isascii() or "\n" in text):
            return [GUIAction("CLIPBOARD_PASTE", {"text": text})]
        return [GUIAction("TYPING", {"text": text})]
    if action == "scroll":
        payload: dict[str, Any] = {"dx": 0, "dy": 0}
        if sdir in ("up", "down"):
            payload["dy"] = samt if sdir == "up" else -samt
        elif sdir in ("left", "right"):
            payload["dx"] = samt if sdir == "right" else -samt
        if x is not None:
            payload["x"], payload["y"] = x, y
        wrap = _mods(text)
        acts = [GUIAction("KEY_DOWN", {"key": k}) for k in wrap]
        acts.append(GUIAction("SCROLL", payload))
        acts += [GUIAction("KEY_UP", {"key": k}) for k in reversed(wrap)]
        return acts
    if action in ("left_click", "right_click", "double_click", "middle_click", "triple_click", "left_press"):
        wrap = _mods(text)
        acts = [GUIAction("KEY_DOWN", {"key": k}) for k in wrap]
        pos = {"x": x, "y": y} if x is not None else {}
        if action == "left_click":
            acts.append(GUIAction("CLICK", {**pos, "button": "left"}))
        elif action == "right_click":
            acts.append(GUIAction("RIGHT_CLICK", pos))
        elif action == "double_click":
            acts.append(GUIAction("DOUBLE_CLICK", pos))
        elif action == "middle_click":
            acts.append(GUIAction("CLICK", {**pos, "button": "middle"}))
        elif action == "triple_click":
            acts += [GUIAction("CLICK", {**pos, "button": "left"}) for _ in range(3)]
        elif action == "left_press":
            if pos:
                acts.append(GUIAction("MOVE_TO", pos))
            acts += [GUIAction("MOUSE_DOWN", {"button": "left"}), GUIAction("WAIT", {"seconds": 1.0}), GUIAction("MOUSE_UP", {"button": "left"})]
        acts += [GUIAction("KEY_UP", {"key": k}) for k in reversed(wrap)]
        return acts
    return []


class GatewayClaudeComputerLLM:
    """Stateful Anthropic computer-use backend compatible with GUILLMFunction."""

    def __init__(self, *, model: str, api_key: str = "", base_url: str = "",
                 timeout_s: float = 180.0, reasoning_effort: str = "high",
                 images_to_keep: int = 5, max_tokens: int = 1500) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
                         or os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("OPENAI_API_BASE", "")
                         or GATEWAY_DEFAULT)
        # routify (and any real Anthropic endpoint) speaks native Messages at
        # /v1/messages; only the legacy DashScope compat gateway takes the
        # native-protocol-wrapped body at /chat/completions.
        self.use_native = "compatible-mode" not in self.base_url
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.images_to_keep = images_to_keep
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []
        self.pending_tool_use_id: str = ""
        self._lock = threading.Lock()
        self._generation = 0

    def reset(self) -> None:
        self.cancel_pending()

    def cancel_pending(self) -> None:
        with self._lock:
            self._generation += 1
            self.messages = []
            self.pending_tool_use_id = ""

    def _url(self) -> str:
        base = self.base_url.rstrip("/")
        return base + ("/v1/messages" if self.use_native else "/chat/completions")

    def _payload(self, w: int, h: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "tools": [{"type": COMPUTER_TOOL_TYPE, "name": "computer",
                       "display_width_px": w, "display_height_px": h}],
            "messages": self.messages,
        }
        if not self.use_native:
            # DashScope compat gateway: native-protocol passthrough + beta via param.
            payload["dashscope_extend_params"] = {"using_native_protocol": "true"}
            payload["model_specific_params"] = {"anthropic-beta": COMPUTER_USE_BETA}
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.use_native:
            # Real Anthropic Messages endpoint (routify /protocol/anthropic): version
            # header is required and the computer-use beta rides as an HTTP header.
            headers["anthropic-version"] = "2023-06-01"
            headers["anthropic-beta"] = COMPUTER_USE_BETA
        req = urllib.request.Request(
            self._url(), data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def __call__(self, prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
        img_b64, w, h = _jpeg_b64(screenshot_bytes)
        image_block = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}}
        with self._lock:
            gen = self._generation
            if not self.messages:
                self.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}, image_block]})
            elif self.pending_tool_use_id:
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": self.pending_tool_use_id, "content": [image_block]}]})
            else:
                self.messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Continue from the latest screenshot."}, image_block]})
            _filter_recent_images(self.messages, self.images_to_keep)
            payload = self._payload(w, h)

        retries = int(os.environ.get("ASIL_GUI_LLM_RETRIES", "12") or 12)
        last_err: Exception | None = None
        data = None
        for attempt in range(max(1, retries)):
            if gen != self._generation:
                raise RuntimeError("claude computer-use call cancelled after task reset")
            try:
                data = self._post(payload)
                break
            except Exception as exc:  # noqa: BLE001 - backend owns retry
                last_err = exc
                if attempt + 1 >= max(1, retries):
                    break
                time.sleep(min(10.0, 2.0 * (attempt + 1)) + random.uniform(0.0, 3.0))
        if data is None:
            raise RuntimeError(f"claude computer-use gateway failed: {last_err}")

        content = data.get("content", []) if isinstance(data, dict) else []
        thoughts: list[str] = []
        actions: list[GUIAction] = []
        tool_use_id = ""
        assistant_blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                thoughts.append(block["text"])
                assistant_blocks.append({"type": "text", "text": block["text"]})
            elif btype == "thinking":
                assistant_blocks.append(block)
            elif btype == "tool_use":
                tool_use_id = str(block.get("id", "") or "")
                assistant_blocks.append({"type": "tool_use", "id": tool_use_id,
                                         "name": block.get("name", "computer"), "input": block.get("input", {})})
                actions.extend(anthropic_action_to_gui_actions(block.get("input", {}) or {}))

        with self._lock:
            if gen == self._generation:
                if assistant_blocks:
                    self.messages.append({"role": "assistant", "content": assistant_blocks})
                self.pending_tool_use_id = tool_use_id

        text = "\n".join(thoughts).strip()
        if not actions:
            up = text.upper()
            if "[INFEASIBLE]" in up or "FAIL" in up:
                text = (text + "\nAction: FAIL").strip()
            elif any(k in up for k in ("TASK COMPLETE", "DONE", "COMPLETED", "FINISHED")):
                text = (text + "\nAction: DONE").strip()
        return GUIModelOutput(
            text=text or "Thought: execute computer action.",
            provider="anthropic-computer-use",
            model=self.model,
            actions=tuple(actions),
            metadata={"backend": "osworld_claude", "tool": COMPUTER_TOOL_TYPE, "reasoning_effort": self.reasoning_effort},
        )


def create_osworld_claude_llm_fn(*, model: str, api_key: str = "", base_url: str = "",
                                 timeout_s: float = 180.0, reasoning_effort: str = "high") -> GatewayClaudeComputerLLM:
    return GatewayClaudeComputerLLM(model=model, api_key=api_key, base_url=base_url,
                                    timeout_s=timeout_s, reasoning_effort=reasoning_effort,
                                    images_to_keep=int(os.environ.get("ASIL_GUI_IMAGES_TO_KEEP", "5") or 5))
