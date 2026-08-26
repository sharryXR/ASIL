"""Real GUI-agent task loop with screenshot observation and pyautogui-style actions."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from asil.eval.evaluator import evaluate_task_result
from asil.eval.metrics import StepResult, TaskResult
from asil.gui_eval import requires_gui_persist, sync_adapter_from_gui
from asil.gui_agent.controller import X11GUIController
from asil.gui_agent.llm import GUILLMFunction, GUIModelOutput
from asil.gui_agent.parser import GUIAction, GUIAgentTrace, parse_gui_response
from asil.gui_agent.prompts import GUI_SYSTEM_PROMPT, build_gui_user_prompt
from asil.gui_agent.session import (
    GUISessionStartupError,
    browser_page_snapshot,
    create_startup_diagnostics,
    record_startup_phase,
    resolve_gui_session_spec,
    start_gui_session,
)
from asil.protocol import Observation


def _load_png_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _model_request_timeout_s() -> float:
    if os.environ.get("ASIL_GUI_LLM_CALL_TIMEOUT_S", "").strip():
        return _env_float("ASIL_GUI_LLM_CALL_TIMEOUT_S", 180.0)
    return _env_float("ASIL_GUI_LLM_TIMEOUT_S", 180.0)


def _timeout_model_output(timeout_s: float) -> GUIModelOutput:
    return GUIModelOutput(
        text=(
            f"Thought: GUI model request exceeded {timeout_s:.1f}s; stop this task "
            "so the batch can continue.\nAction: FAIL"
        ),
        provider="timeout_guard",
        model="timeout_guard",
    )


def _call_llm_with_timeout(
    llm_fn: GUILLMFunction,
    prompt: str,
    screenshot_bytes: bytes,
    *,
    timeout_s: float,
) -> tuple[GUIModelOutput, bool]:
    if timeout_s <= 0:
        return llm_fn(prompt, screenshot_bytes), False

    done = threading.Event()
    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["output"] = llm_fn(prompt, screenshot_bytes)
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread.
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_target, name="asil-gui-llm-call", daemon=True)
    thread.start()
    if not done.wait(timeout_s):
        cancel_pending = getattr(llm_fn, "cancel_pending", None)
        if callable(cancel_pending):
            cancel_pending()
        return _timeout_model_output(timeout_s), True
    if "error" in result:
        raise result["error"]
    return result["output"], False


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.width, image.height


def _screenshot_progress_fingerprint(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _write_step_action(task_dir: Path, step_num: int, trace: GUIAgentTrace) -> None:
    payload = {
        "thought": trace.thought,
        "raw_text": trace.raw_text,
        "action": {"action_type": trace.action.action_type, **trace.action.payload},
        "provider": trace.provider,
        "model": trace.model,
        "reasoning_summary": trace.reasoning_summary,
        "metadata": _json_safe(trace.metadata or {}),
        "model_latency_ms": round(trace.model_latency_ms, 2),
        "action_execution_latency_ms": round(trace.action_execution_latency_ms, 2),
        "render_latency_ms": round(trace.render_latency_ms, 2),
        "evaluation_latency_ms": round(trace.evaluation_latency_ms, 2),
        "step_total_latency_ms": round(trace.step_total_latency_ms, 2),
    }
    (task_dir / f"step_{step_num}_action.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        return str(value)


def _write_startup_diagnostics(task_dir: Path, diagnostics: dict[str, Any] | None) -> None:
    if diagnostics is None:
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "startup_diagnostics.json").write_text(
        json.dumps(_json_safe(diagnostics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _start_gui_session_with_diagnostics(spec, startup_diagnostics: dict[str, Any]):
    signature = inspect.signature(start_gui_session)
    supports_diagnostics = "startup_diagnostics" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if supports_diagnostics:
        return start_gui_session(spec, startup_diagnostics=startup_diagnostics)
    return start_gui_session(spec)


def _render_step(session, adapter, task_dir: Path, step_num: int) -> tuple[str, bool]:
    output_path = task_dir / f"step_{step_num}.png"
    try:
        capture_complete = session.capture(output_path)
    except Exception as exc:
        if step_num == 0:
            raise GUISessionStartupError(
                "window_timeout",
                f"{adapter.app_name} step_0 capture did not complete.",
            ) from exc
        raise
    capture_metadata = _json_safe(getattr(session, "last_capture_metadata", {}) or {})
    render_meta = {
        "filename": output_path.name,
        "kind": "app_window" if session.spec.surface_type in {"desktop", "multi_window"} else "web_ui_screenshot",
        "backend": "x11-window-capture",
        "actual_page": True,
        "description": adapter.app_name,
        "capture_complete": capture_complete,
        "capture_metadata": capture_metadata,
    }
    if isinstance(capture_metadata, dict):
        for key in (
            "display",
            "screen",
            "root_capture_size",
            "cropped_size",
            "window_id",
            "window_geometry",
            "fallback_reason",
            "fallback_used",
            "fallback_window_id",
            "fallback_app",
            "fallback_maximize_used",
            "active_window_too_small",
            "retry_capture_used",
            "window_visible",
            "window_visible_strict",
            "window_visible_tolerance_px",
            "crop_meets_expectation",
        ):
            if key in capture_metadata:
                render_meta[key] = capture_metadata[key]
    (task_dir / f"step_{step_num}.render.json").write_text(
        json.dumps(render_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path.name, capture_complete


def _sync_controller_capture_target(controller: X11GUIController, session) -> None:
    setter = getattr(controller, "set_capture_window_id", None)
    if callable(setter):
        setter(getattr(session, "last_capture_window_id", ""))


def _execute_gui_action(
    controller: X11GUIController,
    session,
    action: GUIAction,
    *,
    spec,
) -> None:
    if action.action_type.upper() == "ACTIVATE_APP":
        app = action.payload.get("app") or action.payload.get("target")
        if not app:
            raise ValueError("ACTIVATE_APP requires an 'app' field.")
        window_id = session.activate_app(str(app))
        controller.set_capture_window_id(window_id)
        return
    controller.execute(action, spec=spec)


def _initialization_watchdog_timeout_s(spec) -> float:
    """Return a forgiving hard cap for task startup through step_0 capture."""

    override = os.environ.get("ASIL_GUI_INIT_WATCHDOG_S", "").strip()
    if override:
        try:
            parsed = float(override)
        except ValueError:
            parsed = 0.0
        if parsed > 0:
            return parsed

    startup_timeout_s = max(float(getattr(spec, "startup_timeout_s", 45.0) or 45.0), 10.0)
    if getattr(spec, "surface_type", "") == "browser":
        return float(min(max(300.0, startup_timeout_s * 5), 900.0))
    return float(min(max(240.0, startup_timeout_s * 4), 720.0))


def _write_initialization_watchdog_failure(
    *,
    task_dir: Path,
    task,
    adapter,
    spec,
    timeout_s: float,
    startup_diagnostics: dict[str, Any] | None = None,
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    message = (
        f"startup_watchdog_timeout: {task.software}/{task.id} GUI initialization "
        f"exceeded {timeout_s:.1f}s before step_0 completed."
    )
    (task_dir / "runtime_error.txt").write_text(message + "\n", encoding="utf-8")
    payload = {
        "error_category": "startup_watchdog_timeout",
        "task_id": task.id,
        "software": task.software,
        "app_name": getattr(adapter, "app_name", ""),
        "timeout_s": timeout_s,
        "surface_type": getattr(spec, "surface_type", ""),
        "startup_timeout_s": getattr(spec, "startup_timeout_s", None),
        "timestamp": datetime.now().isoformat(),
        "note": "No result.txt is written so resume treats this task as incomplete and rerunnable.",
    }
    (task_dir / "initialization_watchdog.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if startup_diagnostics is not None:
        record_startup_phase(
            startup_diagnostics,
            "initialization_watchdog",
            time.monotonic() - timeout_s,
            status="error",
            extra={
                "error_category": "startup_watchdog_timeout",
                "timeout_s": timeout_s,
            },
        )
        _write_startup_diagnostics(task_dir, startup_diagnostics)


class _TaskInitializationWatchdog:
    """Hard-stop a worker if GUI startup gets stuck before step_0 exists.

    The normal startup path raises categorized Python exceptions. This watchdog
    exists for the nastier case: a Playwright/X11/native call stops returning at
    all. In production it exits the eval worker so managed resume can retry the
    task later; tests can inject `on_timeout` to avoid hard exit.
    """

    def __init__(
        self,
        *,
        task_dir: Path,
        task,
        adapter,
        spec,
        timeout_s: float,
        startup_diagnostics: dict[str, Any] | None = None,
        on_timeout: Callable[[], None] | None = None,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.task_dir = task_dir
        self.task = task
        self.adapter = adapter
        self.spec = spec
        self.timeout_s = max(float(timeout_s), 0.0)
        self.startup_diagnostics = startup_diagnostics
        self.on_timeout = on_timeout
        self.poll_interval_s = max(float(poll_interval_s), 0.001)
        self._completed = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_TaskInitializationWatchdog":
        if self.timeout_s <= 0:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name=f"asil-gui-init-watchdog-{self.task.software}-{self.task.id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disarm()
        return None

    def disarm(self) -> None:
        self._completed.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=min(self.poll_interval_s * 2, 1.0))

    def _run(self) -> None:
        if self._completed.wait(self.timeout_s):
            return

        _write_initialization_watchdog_failure(
            task_dir=self.task_dir,
            task=self.task,
            adapter=self.adapter,
            spec=self.spec,
            timeout_s=self.timeout_s,
            startup_diagnostics=self.startup_diagnostics,
        )
        try:
            from asil.gui_agent.session import _cleanup_gui_processes
            from asil.rendering import stop_virtual_display

            _cleanup_gui_processes(self.spec)
            stop_virtual_display()
        except Exception:
            pass

        message = (
            f"[ASIL GUI watchdog] {self.task.software}/{self.task.id} exceeded "
            f"{self.timeout_s:.1f}s during initialization; terminating worker for resume.\n"
        )
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
        except Exception:
            pass
        if self.on_timeout is not None:
            self.on_timeout()
            return
        os._exit(124)


def _build_trace(
    *,
    instruction: str,
    model_output: GUIModelOutput,
    parsed_action: GUIAction,
    thought: str,
) -> GUIAgentTrace:
    return GUIAgentTrace(
        instruction=instruction,
        thought=thought,
        action=parsed_action,
        raw_text=model_output.text,
        provider=model_output.provider,
        model=model_output.model,
        reasoning_summary=model_output.reasoning_summary,
        metadata=model_output.metadata,
    )


def _action_to_payload(action: GUIAction) -> dict[str, Any]:
    return {"action_type": action.action_type, **dict(action.payload)}


def _trace_action_for_actions(actions: list[GUIAction]) -> GUIAction:
    if len(actions) == 1:
        return actions[0]
    return GUIAction("BATCH", {"actions": [_action_to_payload(action) for action in actions]})


def _actions_from_model_output(model_output: GUIModelOutput) -> tuple[list[GUIAction], GUIAction, str]:
    if model_output.actions:
        actions = list(model_output.actions)
        thought = model_output.reasoning_summary or "Execute the OSWorld GPT-5.4 computer-use action batch."
        return actions, _trace_action_for_actions(actions), thought

    parsed = parse_gui_response(model_output.text)
    return [parsed.action], parsed.action, parsed.thought


def _execute_gui_actions(
    controller: X11GUIController,
    session,
    actions: list[GUIAction],
    *,
    spec,
) -> None:
    for action in actions:
        _execute_gui_action(controller, session, action, spec=spec)


def _persist_gui_state(adapter, controller: X11GUIController, spec) -> None:
    persist_method = getattr(adapter, "persist_gui_state", None)
    if callable(persist_method):
        persist_method(controller, spec)
        return
    if not requires_gui_persist(adapter):
        return
    if not spec.persist_shortcuts:
        return
    controller.persist(spec)


def _observation_progress_fingerprint(obs: Observation) -> str:
    payload = {
        "app_state": {
            "current_view": obs.app_state.current_view,
            "active_document": obs.app_state.active_document,
            "document_path": obs.app_state.document_path,
        },
        "elements": [
            {
                "id": element.id,
                "type": element.type,
                "label": element.label,
                "value": element.value,
                "metadata": element.metadata,
            }
            for element in obs.interactive_elements
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def run_gui_agent_task(
    adapter,
    task,
    llm_fn: GUILLMFunction,
    *,
    max_steps: int,
    task_dir: Path,
) -> TaskResult:
    # Stateful CUA backends keep intra-task conversation history; never carry it
    # across benchmark task boundaries when a worker reuses the same callable.
    reset_llm = getattr(llm_fn, "reset", None)
    if callable(reset_llm):
        reset_llm()

    task_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("step_*.png", "step_*.render.json", "step_*_action.json"):
        for old_file in task_dir.glob(pattern):
            old_file.unlink()

    task_info = {
        "task_id": task.id,
        "task_name": task.description,
        "instruction": task.instruction,
        "software": task.software,
        "related_apps": getattr(task, "related_apps", []),
        "app_initial_states": getattr(task, "app_initial_states", {}),
        "primary_app": getattr(task, "primary_app", ""),
        "snapshot": task.snapshot,
        "initial_state": getattr(task, "initial_state", "default"),
        "validation": task.validation,
        "evaluator": task.evaluator,
        "gui_expectations": getattr(task, "gui_expectations", {}),
        "render_target": getattr(task, "render_target", {}),
    }
    (task_dir / "task_info.json").write_text(
        json.dumps(task_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    controller = X11GUIController()
    spec = resolve_gui_session_spec(adapter)
    startup_diagnostics = create_startup_diagnostics(spec)
    startup_diagnostics.update(
        {
            "task_id": task.id,
            "software": task.software,
            "app_name": getattr(adapter, "app_name", ""),
            "task_dir": str(task_dir),
        }
    )
    start = time.monotonic()
    step_results: list[StepResult] = []
    traj_entries: list[dict[str, Any]] = []

    obs = adapter.observe()
    initial_element_count = len(obs.interactive_elements)

    init_watchdog = _TaskInitializationWatchdog(
        task_dir=task_dir,
        task=task,
        adapter=adapter,
        spec=spec,
        timeout_s=_initialization_watchdog_timeout_s(spec),
        startup_diagnostics=startup_diagnostics,
    )
    init_watchdog.__enter__()
    try:
        session_cm = _start_gui_session_with_diagnostics(spec, startup_diagnostics)
    except BaseException as exc:
        init_watchdog.disarm()
        record_startup_phase(startup_diagnostics, "startup_exception", start, status="error", exc=exc)
        _write_startup_diagnostics(task_dir, startup_diagnostics)
        raise

    with session_cm as session:
        step0_started = time.monotonic()
        try:
            step_0_file, step_0_capture_complete = _render_step(session, adapter, task_dir, 0)
        except BaseException as exc:
            record_startup_phase(
                startup_diagnostics,
                "step0_capture",
                step0_started,
                status="error",
                exc=exc,
                extra={"page_snapshot": browser_page_snapshot(getattr(session, "browser_page", None))},
            )
            _write_startup_diagnostics(task_dir, startup_diagnostics)
            raise
        else:
            record_startup_phase(
                startup_diagnostics,
                "step0_capture",
                step0_started,
                extra={
                    "capture_complete": step_0_capture_complete,
                    "filename": step_0_file,
                    "capture_metadata": getattr(session, "last_capture_metadata", {}),
                    "page_snapshot": browser_page_snapshot(getattr(session, "browser_page", None)),
                },
            )
            _write_startup_diagnostics(task_dir, startup_diagnostics)
        finally:
            init_watchdog.disarm()
        _sync_controller_capture_target(controller, session)
        history: list[dict[str, Any]] = []

        deadlocked = False
        last_obs_hash = ""
        last_screenshot_hash = _screenshot_progress_fingerprint(task_dir / step_0_file)
        last_step_score: float | None = None
        repeat_count = 0

        for step_num in range(1, max_steps + 1):
            screenshot_path = task_dir / f"step_{step_num - 1}.png"
            width, height = _image_size(screenshot_path)
            prompt = build_gui_user_prompt(
                instruction=task.instruction,
                window_width=width,
                window_height=height,
                history=history,
                software=task.software,
                related_apps=getattr(task, "related_apps", ()),
            )

            model_start = time.monotonic()
            print(f"    step_{step_num}: model request from {screenshot_path.name}", flush=True)
            model_timeout_s = _model_request_timeout_s()
            model_output, model_timed_out = _call_llm_with_timeout(
                llm_fn,
                f"{GUI_SYSTEM_PROMPT}\n\n{prompt}",
                _load_png_bytes(screenshot_path),
                timeout_s=model_timeout_s,
            )
            model_latency_ms = (time.monotonic() - model_start) * 1000
            if model_timed_out:
                print(
                    f"    step_{step_num}: model request exceeded {model_timeout_s:.1f}s; emitting FAIL",
                    flush=True,
                )

            parsed_action: GUIAction | None = None
            capture_complete = True
            screenshot_file = screenshot_path.name
            evaluation = None
            trace = None
            step_error: str | None = None

            try:
                actions_to_execute, parsed_action, thought = _actions_from_model_output(model_output)
                trace = _build_trace(
                    instruction=task.instruction,
                    model_output=model_output,
                    parsed_action=parsed_action,
                    thought=thought,
                )
                trace.model_latency_ms = round(model_latency_ms, 2)

                action_start = time.monotonic()
                _execute_gui_actions(controller, session, actions_to_execute, spec=spec)
                trace.action_execution_latency_ms = round((time.monotonic() - action_start) * 1000, 2)

                render_start = time.monotonic()
                screenshot_file, capture_complete = _render_step(session, adapter, task_dir, step_num)
                _sync_controller_capture_target(controller, session)
                trace.render_latency_ms = round((time.monotonic() - render_start) * 1000, 2)

                _persist_gui_state(adapter, controller, spec)
                sync_adapter_from_gui(adapter, session)
                obs = adapter.observe()
                evaluation_start = time.monotonic()
                evaluation = evaluate_task_result(task, obs)
                trace.evaluation_latency_ms = round((time.monotonic() - evaluation_start) * 1000, 2)
                trace.step_total_latency_ms = round(
                    trace.model_latency_ms
                    + trace.action_execution_latency_ms
                    + trace.render_latency_ms
                    + trace.evaluation_latency_ms,
                    2,
                )
            except Exception as exc:
                step_error = str(exc)
                try:
                    _persist_gui_state(adapter, controller, spec)
                    sync_adapter_from_gui(adapter, session)
                    obs = adapter.observe()
                except Exception:
                    pass
                failure_action = GUIAction("FAIL", {"error": step_error})
                failure_thought = trace.thought if trace is not None else "Model response could not be executed."
                trace = _build_trace(
                    instruction=task.instruction,
                    model_output=model_output,
                    parsed_action=failure_action,
                    thought=failure_thought,
                )
                trace.model_latency_ms = round(model_latency_ms, 2)
                trace.step_total_latency_ms = round(
                    trace.model_latency_ms
                    + trace.action_execution_latency_ms
                    + trace.render_latency_ms
                    + trace.evaluation_latency_ms,
                    2,
                )

            assert trace is not None
            _write_step_action(task_dir, step_num, trace)

            obs_hash = _observation_progress_fingerprint(obs)
            current_score = evaluation.score if evaluation is not None else None
            current_screenshot_hash = _screenshot_progress_fingerprint(task_dir / screenshot_file)
            if (
                obs_hash == last_obs_hash
                and current_score == last_step_score
                and current_screenshot_hash == last_screenshot_hash
            ):
                repeat_count += 1
                if repeat_count >= 5:
                    deadlocked = True
            else:
                repeat_count = 0
                last_obs_hash = obs_hash
                last_screenshot_hash = current_screenshot_hash
                last_step_score = current_score

            action = trace.action
            step_results.append(
                StepResult(
                    step_num=step_num,
                    action_type=action.action_type,
                    target=adapter.app_name,
                    params=dict(action.payload),
                    success=step_error is None and action.action_type not in {"FAIL"},
                    latency_ms=trace.step_total_latency_ms,
                    observation_element_count=len(obs.interactive_elements),
                    observation_source="screenshot",
                    agent_response=model_output.text,
                )
            )
            entry = {
                "step_num": step_num,
                "action_timestamp": datetime.now().isoformat(),
                "instruction": task.instruction,
                "thought": trace.thought,
                "action": _action_to_payload(action),
                "screenshot_file": screenshot_file,
                "render_metadata_file": f"step_{step_num}.render.json" if step_error is None else f"step_{step_num - 1}.render.json",
                "render_actual_page": True,
                "render_capture_complete": capture_complete,
                "provider": trace.provider,
                "model": trace.model,
                "metadata": _json_safe(trace.metadata or {}),
                "model_latency_ms": trace.model_latency_ms,
                "action_execution_latency_ms": trace.action_execution_latency_ms,
                "render_latency_ms": trace.render_latency_ms,
                "evaluation_latency_ms": trace.evaluation_latency_ms,
                "step_total_latency_ms": trace.step_total_latency_ms,
                "step_score": evaluation.score if evaluation is not None else None,
                "step_success_snapshot": evaluation.success if evaluation is not None else False,
                "observation_source": "screenshot",
            }
            if step_error is not None:
                entry["error"] = step_error
            history.append(entry)
            traj_entries.append(entry)
            print(
                f"    step_{step_num}: {action.action_type} "
                f"({trace.step_total_latency_ms:.0f}ms, {screenshot_file})",
                flush=True,
            )

            if step_error is not None or action.action_type in {"DONE", "FAIL"} or deadlocked:
                break

        final_evaluation = evaluate_task_result(task, obs)
        with open(task_dir / "traj.jsonl", "w", encoding="utf-8") as handle:
            for entry in traj_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        (task_dir / "evaluation.json").write_text(
            json.dumps(final_evaluation.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (task_dir / "result.txt").write_text(str(final_evaluation.score), encoding="utf-8")

    total_time = time.monotonic() - start
    return TaskResult(
        task_id=task.id,
        software=task.software,
        difficulty=task.difficulty,
        instruction=task.instruction,
        success=final_evaluation.success,
        score=final_evaluation.score,
        steps=len(step_results),
        step_results=step_results,
        e2e_time_s=total_time,
        deadlocked=deadlocked,
        observation_element_count=initial_element_count,
        total_element_count=initial_element_count,
    )
