"""Prompt templates for the real screenshot-driven GUI agent."""

from __future__ import annotations

import json
from typing import Any, Sequence


GUI_SYSTEM_PROMPT = """\
You are the GUI baseline participant for the ASIL benchmark.
You control real software by looking only at the latest screenshot of the active window.

You must output exactly two sections:
Thought: <brief reasoning about the visible GUI and next step>
Action: <one pyautogui command OR one special token WAIT/DONE/FAIL>

The screenshot you receive is a crop of the active application window.
All x/y coordinates must be relative to that screenshot:
- top-left is (0, 0)
- bottom-right is (window_width - 1, window_height - 1)

Use only one action per step. Valid forms include:
- pyautogui.moveTo(x, y)
- pyautogui.click(x, y)
- pyautogui.doubleClick(x, y)
- pyautogui.rightClick(x, y)
- pyautogui.dragTo(x, y, duration=0.5, button='left')
- pyautogui.scroll(amount)
- pyautogui.scroll(amount, x=..., y=...)
- pyautogui.hscroll(amount)
- pyautogui.write("text")
- pyautogui.typewrite("text")
- pyautogui.press("enter")
- pyautogui.keyDown("shift")
- pyautogui.keyUp("shift")
- pyautogui.hotkey("ctrl", "s")
- {"action_type":"CLICK","x":123,"y":456}
- {"action_type":"TYPING","text":"..."}
- {"action_type":"ACTIVATE_APP","app":"code_server"}
- WAIT
- DONE
- FAIL

Rules:
1. Use only what is visible in the screenshot. Do not use hidden structured state.
2. Prefer one atomic GUI action per step.
3. If the UI is still changing, use WAIT.
4. Do not stop early. Use DONE only when the instruction appears complete in the visible GUI and the change has been truly applied, not merely selected or prepared.
5. If a click or key press does not appear to change the visible document, retry with a better-targeted click, a WAIT, or a more direct editing action instead of repeating the same ineffective move.
6. Prefer actions that directly change the document or saved application state over transient UI-only actions such as repeatedly reselecting the same object or tool.
7. Use FAIL only for hard blockers after trying a reasonable recovery such as WAIT, retargeting a click, scrolling, or focusing the intended field.
8. If recent steps made no visible or scored progress, change strategy instead of repeating the same ineffective action.
9. Do not emit arbitrary JSON or markdown outside the required Thought and Action lines.
10. For any action that enters text containing a newline, a quote, or more than a short phrase, use ONLY the JSON TYPING form: {"action_type":"TYPING","text":"..."}.
11. In the JSON TYPING form, escape newlines as \\n and quotes as \\\". Do not use pyautogui.write for multi-line text.
12. The Action line must contain only the single command or JSON action itself. Do not wrap it in explanation such as "use ... now".
13. If you include imports such as "import pyautogui", the action still must contain exactly one executable pyautogui command.
14. Keep Thought short.
""".strip()


def build_gui_user_prompt(
    *,
    instruction: str,
    window_width: int,
    window_height: int,
    history: Sequence[dict[str, Any]],
    software: str,
    related_apps: Sequence[str] = (),
) -> str:
    lines = [
        f"Software: {software}",
        f"Instruction: {instruction}",
        f"Visible window size: {window_width}x{window_height}",
        "",
        "Recent interaction history:",
    ]
    if history:
        for item in history[-5:]:
            score = item.get("step_score")
            score_repr = "none" if score is None else score
            history_line = (
                f"- step {item['step_num']}: action={json.dumps(item['action'], ensure_ascii=False)} "
                f"score={score_repr}"
            )
            if "error" in item:
                history_line += f" error={item['error']}"
            lines.append(
                history_line
            )
    else:
        lines.append("- no prior GUI actions")

    if software.lower() == "jupyterlab":
        lines.extend(
            [
                "",
                "JupyterLab tips:",
                "- Ignore the browser chrome and act inside the JupyterLab work area.",
                "- Dismiss popups like the Jupyter news prompt if they block the target area.",
                "- Notebook tasks: Esc to enter command mode, Enter to edit a cell, A/B to insert above/below, M/Y to switch markdown/code, Shift+Enter to run the selected cell.",
                "- Text-file tasks: open the file from the file browser, click inside the editor body before Ctrl+A, then replace or append text and save with Ctrl+S.",
                "- For multi-line text entry, use Action JSON such as {\"action_type\":\"TYPING\",\"text\":\"line1\\nline2\\n\"}; do not describe the typing action in prose.",
                "- If renaming a file and the extension is already present in the rename field, change only the basename instead of duplicating .md or .py.",
            ]
        )

    if software.lower() == "multi_apps":
        apps = ", ".join(related_apps)
        app_set = {app.lower() for app in related_apps}
        lines.extend(
            [
                "",
                "Multi-app tips:",
                f"- This task involves these apps: {apps}.",
                "- Prefer {\"action_type\":\"ACTIVATE_APP\",\"app\":\"<app_name>\"} before acting in a different app.",
                "- Use Alt+Tab or Ctrl+Tab only as a fallback when ACTIVATE_APP cannot reach the intended window.",
                "- The screenshot is the current working window; coordinates are relative to that screenshot only.",
                "- If the screenshot shows a tiny dialog, popup, or the wrong app, activate the intended app before clicking.",
                "- Keep using exactly one pyautogui action per step.",
            ]
        )
        if "code_server" in app_set:
            lines.extend(
                [
                    "- code-server: if the target file is open, click inside the editor, use Ctrl+A only when replacing the whole file, enter the required text with JSON TYPING, and save with Ctrl+S.",
                    "- code-server: use Ctrl+P to open a file by path if the wrong file is visible; avoid right-click context menus for normal editing.",
                ]
            )
        if "jupyterlab" in app_set:
            lines.extend(
                [
                    "- JupyterLab: use the file browser or Ctrl+L/visible editor area; for notebooks use Esc/Enter and Shift+Enter, and save with Ctrl+S.",
                    "- JupyterLab: for multi-line text, use JSON TYPING with escaped \\n after focusing the target cell or text editor.",
                ]
            )
        if "gitea" in app_set:
            lines.extend(
                [
                    "- Gitea: use the visible repository web UI; click Issues/New Issue or files/commits as needed, then type into title/body fields and submit.",
                    "- Gitea: prefer direct form fields and visible buttons over browser address-bar edits unless navigation is clearly required.",
                ]
            )
        if "nautilus" in app_set:
            lines.append("- Nautilus: use the file list directly; F2 renames a selected file, Ctrl+L focuses the path bar, and Enter opens the selected item.")
        if any(app in app_set for app in {"libreoffice", "libreoffice_writer", "libreoffice_impress"}):
            lines.append("- LibreOffice: click into the document/slide body before typing, use Ctrl+A only for full replacement, and save with Ctrl+S.")

    lines.extend(
        [
            "",
            "Return exactly:",
            "Thought: ...",
            "Action: ...",
        ]
    )
    return "\n".join(lines)
