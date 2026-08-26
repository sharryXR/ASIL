"""Parsers for the real GUI-agent textual response."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GUIAction:
    action_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ParsedGUIResponse:
    thought: str
    action: GUIAction
    raw_text: str


@dataclass
class GUIAgentTrace:
    instruction: str
    thought: str
    action: GUIAction
    raw_text: str
    provider: str = ""
    model: str = ""
    reasoning_summary: str = ""
    model_latency_ms: float = 0.0
    action_execution_latency_ms: float = 0.0
    render_latency_ms: float = 0.0
    evaluation_latency_ms: float = 0.0
    step_total_latency_ms: float = 0.0
    metadata: dict[str, Any] | None = None


_SPECIAL_TOKENS = {"WAIT", "DONE", "FAIL"}


def _split_thought_action(text: str) -> tuple[str, str]:
    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", text, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action:\s*(.+)$", text, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else ""
    action_text = action_match.group(1).strip() if action_match else text.strip()
    return thought, action_text


def _extract_first_json(text: str) -> str:
    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:]


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    if isinstance(func, ast.Name):
        return func.id
    raise ValueError("Unsupported action function.")


def _kwargs(call: ast.Call) -> dict[str, Any]:
    return {
        keyword.arg: _literal(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _parse_pyautogui_call(call: ast.Call) -> GUIAction:
    name = _call_name(call)
    kwargs = _kwargs(call)
    args = [_literal(arg) for arg in call.args]

    def positional(index: int, default: Any = None) -> Any:
        return args[index] if index < len(args) else default

    if name == "time.sleep":
        return GUIAction("WAIT", {"seconds": float(positional(0, 1.0))})
    if name == "pyautogui.moveTo":
        return GUIAction("MOVE_TO", {"x": kwargs.get("x", positional(0)), "y": kwargs.get("y", positional(1))})
    if name == "pyautogui.click":
        payload: dict[str, Any] = {}
        if "x" in kwargs or len(args) >= 1:
            payload["x"] = kwargs.get("x", positional(0))
        if "y" in kwargs or len(args) >= 2:
            payload["y"] = kwargs.get("y", positional(1))
        if "button" in kwargs:
            payload["button"] = kwargs["button"]
        return GUIAction("CLICK", payload)
    if name == "pyautogui.doubleClick":
        payload = {}
        if "x" in kwargs or len(args) >= 1:
            payload["x"] = kwargs.get("x", positional(0))
        if "y" in kwargs or len(args) >= 2:
            payload["y"] = kwargs.get("y", positional(1))
        if "button" in kwargs:
            payload["button"] = kwargs["button"]
        return GUIAction("DOUBLE_CLICK", payload)
    if name == "pyautogui.rightClick":
        payload = {"button": "right"}
        if "x" in kwargs or len(args) >= 1:
            payload["x"] = kwargs.get("x", positional(0))
        if "y" in kwargs or len(args) >= 2:
            payload["y"] = kwargs.get("y", positional(1))
        return GUIAction("RIGHT_CLICK", payload)
    if name == "pyautogui.dragTo":
        payload = {
            "x": kwargs.get("x", positional(0)),
            "y": kwargs.get("y", positional(1)),
        }
        if "button" in kwargs:
            payload["button"] = kwargs["button"]
        return GUIAction("DRAG_TO", payload)
    if name == "pyautogui.scroll":
        payload = {"dy": int(kwargs.get("clicks", positional(0, 0)))}
        if "x" in kwargs:
            payload["x"] = kwargs["x"]
        if "y" in kwargs:
            payload["y"] = kwargs["y"]
        return GUIAction("SCROLL", payload)
    if name == "pyautogui.hscroll":
        payload = {"dx": int(kwargs.get("clicks", positional(0, 0)))}
        if "x" in kwargs:
            payload["x"] = kwargs["x"]
        if "y" in kwargs:
            payload["y"] = kwargs["y"]
        return GUIAction("SCROLL", payload)
    if name in {"pyautogui.write", "pyautogui.typewrite"}:
        return GUIAction("TYPING", {"text": str(kwargs.get("message", positional(0, "")))})
    if name == "pyautogui.press":
        return GUIAction("PRESS", {"key": str(kwargs.get("keys", positional(0)))})
    if name == "pyautogui.keyDown":
        return GUIAction("KEY_DOWN", {"key": str(kwargs.get("key", positional(0)))})
    if name == "pyautogui.keyUp":
        return GUIAction("KEY_UP", {"key": str(kwargs.get("key", positional(0)))})
    if name == "pyautogui.hotkey":
        return GUIAction("HOTKEY", {"keys": [str(arg) for arg in args]})
    raise ValueError(f"Unsupported GUI action call: {name}")


def _parse_python_action(text: str) -> GUIAction:
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else text.strip()
    action_match = re.search(r"Action:\s*(.+)$", code, re.DOTALL | re.IGNORECASE)
    if action_match:
        code = action_match.group(1).strip()
    code = "\n".join(
        line for line in code.splitlines() if not re.match(r"^\s*Thought:\s*", line, re.IGNORECASE)
    ).strip()
    module = ast.parse(code)
    for statement in reversed(module.body):
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            return _parse_pyautogui_call(statement.value)
    raise ValueError("No executable pyautogui action found in response.")


def _parse_embedded_python_action(text: str) -> GUIAction:
    call_pattern = re.compile(
        r"(pyautogui\.(?:moveTo|click|doubleClick|rightClick|dragTo|scroll|hscroll|write|typewrite|press|keyDown|keyUp|hotkey)\s*\([^)]*\)|time\.sleep\s*\([^)]*\))",
        re.DOTALL,
    )
    matches = call_pattern.findall(text)
    if not matches:
        raise ValueError("No executable pyautogui action found in response.")
    return _parse_python_action(matches[-1])


def _parse_multiline_typing_action(text: str) -> GUIAction:
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else text.strip()
    action_match = re.search(r"Action:\s*(.+)$", code, re.DOTALL | re.IGNORECASE)
    if action_match:
        code = action_match.group(1).strip()
    code = "\n".join(
        line for line in code.splitlines() if not re.match(r"^\s*Thought:\s*", line, re.IGNORECASE)
    ).strip()

    call_match = re.search(r"pyautogui\.(write|typewrite)\s*\(", code, re.DOTALL)
    if not call_match:
        raise ValueError("No multiline typing action found in response.")

    args_text = code[call_match.end():]
    depth = 1
    closing_index = None
    for index, char in enumerate(args_text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                closing_index = index
                break
    if closing_index is None:
        raise ValueError("Typing action did not contain a complete call.")

    args_text = args_text[:closing_index].strip()
    if args_text.startswith("message="):
        args_text = args_text[len("message="):].strip()

    quote = None
    triple = False
    if args_text.startswith('"""') or args_text.startswith("'''"):
        quote = args_text[:3]
        triple = True
    elif args_text.startswith('"') or args_text.startswith("'"):
        quote = args_text[0]

    if quote is None:
        raise ValueError("Typing action did not start with a quoted string.")

    if triple:
        end = args_text.rfind(quote)
        if end <= 2:
            raise ValueError("Typing action did not contain a complete triple-quoted string.")
        text_value = args_text[3:end]
    else:
        end = args_text.rfind(quote)
        if end <= 0:
            raise ValueError("Typing action did not contain a complete quoted string.")
        text_value = args_text[1:end]

    return GUIAction("TYPING", {"text": text_value})


def _parse_loose_typing_json_action(text: str) -> GUIAction:
    normalized = text
    typing_match = re.search(
        r"""['"]?action_type['"]?\s*:\s*['"]TYPING['"]""",
        normalized,
        re.IGNORECASE | re.DOTALL,
    )
    if not typing_match:
        raise ValueError("No loose JSON typing action found in response.")

    text_match = re.search(
        r"""['"]?text['"]?\s*:\s*""",
        normalized[typing_match.end():],
        re.IGNORECASE | re.DOTALL,
    )
    if not text_match:
        raise ValueError("Loose JSON typing action did not include a text field.")

    args_text = normalized[typing_match.end() + text_match.end():].lstrip()

    quote = None
    triple = False
    if args_text.startswith('"""') or args_text.startswith("'''"):
        quote = args_text[:3]
        triple = True
    elif args_text.startswith('"') or args_text.startswith("'"):
        quote = args_text[0]

    if quote is None:
        raise ValueError("Loose JSON typing action did not start with a quoted string.")

    if triple:
        end = args_text.find(quote, 3)
        if end == -1:
            raise ValueError("Loose JSON typing action did not contain a closing triple quote.")
        text_value = args_text[3:end]
    else:
        escaped = False
        end = None
        for index, char in enumerate(args_text[1:], 1):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote:
                end = index
                break
        if end is None:
            end = args_text.rfind(quote)
            if end <= 0:
                raise ValueError("Loose JSON typing action did not contain a closing quote.")
        text_value = args_text[1:end]

    return GUIAction("TYPING", {"text": text_value})


def parse_gui_response(text: str) -> ParsedGUIResponse:
    normalized_text = text.replace("\\n", "\n")
    thought, action_text = _split_thought_action(normalized_text)
    normalized = action_text.strip().strip("`").strip()
    upper = normalized.upper()
    if upper in _SPECIAL_TOKENS:
        return ParsedGUIResponse(
            thought=thought,
            action=GUIAction(action_type=upper, payload={}),
            raw_text=text,
        )

    try:
        data = json.loads(_extract_first_json(normalized))
        action_type = str(data.pop("action_type", "")).upper()
        payload: dict[str, Any] = {} if action_type in _SPECIAL_TOKENS else data
        action = GUIAction(action_type=action_type, payload=payload)
    except Exception:
        try:
            action = _parse_python_action(normalized)
        except Exception:
            try:
                action = _parse_embedded_python_action(normalized)
            except Exception:
                try:
                    action = _parse_multiline_typing_action(normalized)
                except Exception:
                    action = _parse_loose_typing_json_action(normalized)
    return ParsedGUIResponse(
        thought=thought,
        action=action,
        raw_text=text,
    )
