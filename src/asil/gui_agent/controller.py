"""Execute pyautogui-style GUI actions through the existing X11 toolchain."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Iterable

from asil.adapter import GUISessionSpec
from asil.gui_agent.parser import GUIAction
from asil.rendering import _window_geometry, active_window_id, ensure_virtual_display, wait_for_window


_KEYBOARD_ACTION_TYPES = frozenset(
    {"TYPING", "CLIPBOARD_PASTE", "PRESS", "KEY_DOWN", "KEY_UP", "HOTKEY"}
)


def _key_name(key: str) -> str:
    mapping = {
        "enter": "Return",
        "return": "Return",
        "esc": "Escape",
        "escape": "Escape",
        "backspace": "BackSpace",
        "pageup": "Prior",
        "pagedown": "Next",
        "win": "Super_L",
        "command": "Super_L",
        "cmd": "Super_L",
        "option": "Alt_L",
    }
    normalized = key.strip()
    return mapping.get(normalized.lower(), normalized)


def _button_code(button: str | None) -> str:
    mapping = {"left": "1", "middle": "2", "right": "3"}
    return mapping.get((button or "left").lower(), "1")


class X11GUIController:
    """Window-relative GUI action executor."""

    def __init__(self, *, display: str | None = None) -> None:
        self.display = display
        self.last_capture_window_id = ""

    def set_capture_window_id(self, window_id: str | None) -> None:
        self.last_capture_window_id = str(window_id or "")

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(ensure_virtual_display(self.display))
        return env

    def _xdotool(self) -> str:
        tool = shutil.which("xdotool")
        return tool or "xdotool"

    def _xclip(self) -> str:
        tool = shutil.which("xclip")
        return tool or "xclip"

    @staticmethod
    def _window_ids_match(first: str, second: str) -> bool:
        def as_int(window_id: str) -> int:
            value = str(window_id).strip().lower()
            return int(value, 16 if value.startswith("0x") else 10)

        try:
            return as_int(first) == as_int(second)
        except ValueError:
            return str(first).strip() == str(second).strip()

    def _transient_parent_window_id(self, window_id: str) -> str:
        xprop = shutil.which("xprop")
        if xprop is None:
            return ""
        try:
            result = subprocess.run(
                [xprop, "-id", str(window_id), "WM_TRANSIENT_FOR"],
                check=False,
                capture_output=True,
                text=True,
                env=self._env(),
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        match = re.search(r"window id #\s*(0x[0-9a-fA-F]+|\d+)", result.stdout or "")
        return match.group(1) if match else ""

    def _keyboard_focus_window_id(self, target_window_id: str) -> str:
        try:
            active_id = active_window_id(display=self.display)
        except Exception:
            return target_window_id
        if self._window_ids_match(active_id, target_window_id):
            return target_window_id

        candidate_id = active_id
        visited: set[str] = set()
        for _ in range(8):
            normalized = str(candidate_id).strip().lower()
            if normalized in visited:
                break
            visited.add(normalized)
            candidate_id = self._transient_parent_window_id(candidate_id)
            if not candidate_id:
                break
            if self._window_ids_match(candidate_id, target_window_id):
                return active_id
        return target_window_id

    def _window_id(self, spec: GUISessionSpec) -> str:
        if spec.capture_active_window or spec.surface_type == "multi_window":
            if self.last_capture_window_id:
                return self.last_capture_window_id
            return active_window_id(display=self.display)
        return wait_for_window(
            spec.window_title_pattern,
            window_class_pattern=spec.window_class_pattern,
            display=self.display,
            timeout=spec.startup_timeout_s,
            min_width=spec.min_width,
            min_height=spec.min_height,
        )

    def _to_screen_coords(self, spec: GUISessionSpec, x: float, y: float) -> tuple[int, int]:
        window_id = self._window_id(spec)
        left, top, _width, _height = _window_geometry(window_id, display=self.display)
        return left + int(round(float(x))), top + int(round(float(y)))

    def _run(self, *args: str) -> None:
        subprocess.run(
            [self._xdotool(), *args],
            check=True,
            capture_output=True,
            env=self._env(),
        )

    def _write_clipboard(self, text: str) -> None:
        subprocess.run(
            [self._xclip(), "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            env=self._env(),
        )

    def _persist(self, spec: GUISessionSpec) -> None:
        if not spec.persist_shortcuts:
            return
        for shortcut in spec.persist_shortcuts:
            self.press_combo(shortcut.split("+"))
            time.sleep(0.2)

    def persist(self, spec: GUISessionSpec) -> None:
        window_id = self._window_id(spec)
        self._run("windowactivate", "--sync", window_id)
        self._persist(spec)

    def activate_window_id(self, window_id: str) -> None:
        self._run("windowactivate", "--sync", str(window_id))

    def press_combo(self, keys: Iterable[str]) -> None:
        combo = "+".join(_key_name(key) for key in keys)
        self._run("key", "--clearmodifiers", combo)

    def execute(self, action: GUIAction, *, spec: GUISessionSpec) -> None:
        action_type = action.action_type.upper()
        payload = action.payload

        if action_type in {"WAIT", "DONE", "FAIL"}:
            if action_type == "WAIT":
                time.sleep(float(payload.get("seconds", 1.0)))
            return
        if action_type == "ACTIVATE_APP":
            raise ValueError("ACTIVATE_APP must be handled by the GUI session before controller execution.")

        window_id = self._window_id(spec)
        focus_window_id = (
            self._keyboard_focus_window_id(window_id)
            if action_type in _KEYBOARD_ACTION_TYPES
            else window_id
        )
        self._run("windowactivate", "--sync", focus_window_id)

        if action_type == "MOVE_TO":
            sx, sy = self._to_screen_coords(spec, payload["x"], payload["y"])
            self._run("mousemove", "--sync", str(sx), str(sy))
        elif action_type in {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}:
            button = payload.get("button")
            if action_type == "RIGHT_CLICK":
                button = "right"
            num_clicks = 2 if action_type == "DOUBLE_CLICK" else int(payload.get("num_clicks", 1))
            if "x" in payload and "y" in payload:
                sx, sy = self._to_screen_coords(spec, payload["x"], payload["y"])
                self._run(
                    "mousemove",
                    "--sync",
                    str(sx),
                    str(sy),
                    "click",
                    "--repeat",
                    str(num_clicks),
                    _button_code(button),
                )
            else:
                self._run("click", "--repeat", str(num_clicks), _button_code(button))
        elif action_type == "MOUSE_DOWN":
            self._run("mousedown", _button_code(payload.get("button")))
        elif action_type == "MOUSE_UP":
            self._run("mouseup", _button_code(payload.get("button")))
        elif action_type == "DRAG_TO":
            sx, sy = self._to_screen_coords(spec, payload["x"], payload["y"])
            button = _button_code(payload.get("button"))
            self._run("mousedown", button)
            try:
                self._run("mousemove", "--sync", str(sx), str(sy))
            finally:
                self._run("mouseup", button)
        elif action_type == "SCROLL":
            if "x" in payload and "y" in payload:
                sx, sy = self._to_screen_coords(spec, payload["x"], payload["y"])
                self._run("mousemove", "--sync", str(sx), str(sy))
            dy = int(payload.get("dy", 0))
            if dy:
                button = "4" if dy > 0 else "5"
                for _ in range(abs(dy)):
                    self._run("click", button)
            dx = int(payload.get("dx", 0))
            if dx:
                button = "7" if dx > 0 else "6"
                for _ in range(abs(dx)):
                    self._run("click", button)
        elif action_type == "TYPING":
            self._run("type", "--delay", "0", "--", str(payload["text"]))
        elif action_type == "CLIPBOARD_PASTE":
            self._write_clipboard(str(payload.get("text", "")))
            self.press_combo(["ctrl", "v"])
        elif action_type == "PRESS":
            self._run("key", "--clearmodifiers", _key_name(str(payload["key"])))
        elif action_type == "KEY_DOWN":
            self._run("keydown", _key_name(str(payload["key"])))
        elif action_type == "KEY_UP":
            self._run("keyup", _key_name(str(payload["key"])))
        elif action_type == "HOTKEY":
            self.press_combo([str(key) for key in payload["keys"]])
        else:
            raise ValueError(f"Unsupported GUI action type: {action_type}")
