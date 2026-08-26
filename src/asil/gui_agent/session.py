"""GUI session resolution and lifecycle helpers."""

from __future__ import annotations

import inspect
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.rendering import (
    activate_window,
    capture_window_to_png,
    ensure_user_access,
    ensure_virtual_display,
    launch_gui_process,
    stop_virtual_display,
    terminate_process,
)


_BROWSER_CLASS_PATTERN = r"chromium|Chromium|chrome|Google-chrome"
_SYSTEM_PATH_PREFIXES = (
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/lib"),
    Path("/lib64"),
    Path("/opt"),
)

_BROWSER_PROCESS_NAMES = ("chromium", "chromium-browser", "chrome", "Google-chrome")
_DESKTOP_PROCESS_PATTERN_OVERRIDES = {
    "libreoffice": ("soffice.bin", "soffice", "libreoffice"),
}


@dataclass
class GUISession(AbstractContextManager["GUISession"]):
    spec: GUISessionSpec
    process: Any | None = None
    playwright: Any | None = None
    browser_context: Any | None = None
    browser_page: Any | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    startup_diagnostics: dict[str, Any] | None = None
    child_sessions: dict[str, "GUISession"] = field(default_factory=dict)
    active_child: str = ""
    last_capture_metadata: dict[str, Any] = field(default_factory=dict)
    last_capture_window_id: str = ""

    def _capture_fallback_specs(self) -> list[dict[str, Any]]:
        if self.spec.surface_type != "multi_window" or not self.spec.child_specs:
            return []

        ordered_apps: list[str] = []
        for app_name in (self.active_child, self.spec.primary_child):
            if app_name and app_name in self.spec.child_specs and app_name not in ordered_apps:
                ordered_apps.append(app_name)
        for app_name in self.spec.child_specs:
            if app_name not in ordered_apps:
                ordered_apps.append(app_name)

        fallback_specs: list[dict[str, Any]] = []
        for app_name in ordered_apps:
            child_spec = self.spec.child_specs[app_name]
            fallback_specs.append(
                {
                    "app": app_name,
                    "title_pattern": child_spec.window_title_pattern,
                    "window_class_pattern": child_spec.window_class_pattern,
                    "min_width": child_spec.min_width,
                    "min_height": child_spec.min_height,
                }
            )
        return fallback_specs

    def capture(self, output_path: str | Path) -> bool:
        capture_metadata: dict[str, Any] = {"capture_complete": True}
        capture_window_to_png(
            output_path,
            title_pattern=self.spec.window_title_pattern,
            window_class_pattern=self.spec.window_class_pattern,
            timeout=self.spec.startup_timeout_s,
            settle_delay=self.spec.post_launch_delay_s,
            min_width=self.spec.min_width,
            min_height=self.spec.min_height,
            capture_metadata=capture_metadata,
            active_window=self.spec.capture_active_window or self.spec.surface_type == "multi_window",
            fallback_window_specs=self._capture_fallback_specs(),
            prefer_first_fallback=self.spec.surface_type == "multi_window",
        )
        self.last_capture_metadata = capture_metadata
        self.last_capture_window_id = str(
            capture_metadata.get("window_id") or capture_metadata.get("fallback_window_id") or ""
        )
        fallback_app = str(capture_metadata.get("fallback_app") or "")
        if fallback_app:
            self.active_child = fallback_app
        return bool(capture_metadata.get("capture_complete", True))

    def activate_app(self, app: str) -> str:
        if self.spec.surface_type != "multi_window":
            raise ValueError("ACTIVATE_APP is only available for multi-window GUI sessions.")
        app_name = str(app)
        child_session = self.child_sessions.get(app_name)
        child_spec = self.spec.child_specs.get(app_name)
        if child_session is None or child_spec is None:
            known = ", ".join(sorted(self.child_sessions or self.spec.child_specs))
            raise ValueError(f"Unknown multi-app window {app_name!r}. Available apps: {known}")

        page = getattr(child_session, "browser_page", None)
        bring_to_front = getattr(page, "bring_to_front", None)
        if callable(bring_to_front):
            try:
                bring_to_front()
            except Exception:
                pass

        window_id = activate_window(
            child_spec.window_title_pattern,
            window_class_pattern=child_spec.window_class_pattern,
            timeout=child_spec.startup_timeout_s,
            min_width=child_spec.min_width,
            min_height=child_spec.min_height,
        )
        self.active_child = app_name
        self.last_capture_window_id = str(window_id)
        return str(window_id)

    def close(self) -> None:
        for child in list(self.child_sessions.values()):
            try:
                child.close()
            except Exception:
                pass
        self.child_sessions.clear()
        if self.spec.close_callback is not None:
            try:
                self.spec.close_callback()
            except Exception:
                pass
        if self.browser_context is not None:
            try:
                self.browser_context.close()
            except Exception:
                pass
            self.browser_context = None
        self.browser_page = None
        if self.playwright is not None:
            try:
                self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
        if self.process is not None:
            terminate_process(self.process)
            self.process = None
        _cleanup_gui_processes(self.spec)
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None
        stop_virtual_display()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None


class GUISessionStartupError(RuntimeError):
    """Raised when a real GUI session cannot reach a usable initial state."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class _StartupPhaseTimedOut(TimeoutError):
    """Internal helper used to interrupt stuck GUI startup phases."""


class _StartupTimeoutGuard:
    def __init__(self, seconds: float, message: str) -> None:
        self.seconds = max(float(seconds), 0.0)
        self.message = message
        self._previous_handler = None

    def __enter__(self):
        if self.seconds <= 0 or os.name == "nt":
            return self

        def _handle_timeout(signum, frame):
            del signum, frame
            raise _StartupPhaseTimedOut(self.message)

        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0 and os.name != "nt":
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
        return None


def create_startup_diagnostics(spec: GUISessionSpec) -> dict[str, Any]:
    return {
        "surface_type": spec.surface_type,
        "startup_timeout_s": spec.startup_timeout_s,
        "browser_url": spec.browser_url,
        "window_title_pattern": spec.window_title_pattern,
        "phases": [],
        "snapshots": [],
        "browser_events": {
            "console": [],
            "pageerror": [],
            "requestfailed": [],
        },
        "child_apps": sorted(spec.child_specs),
    }


def _append_limited(items: list[Any], item: Any, *, limit: int = 50) -> None:
    items.append(item)
    if len(items) > limit:
        del items[: len(items) - limit]


def browser_page_snapshot(page: Any | None) -> dict[str, Any]:
    if page is None:
        return {"page_available": False}

    snapshot: dict[str, Any] = {"page_available": True}
    snapshot["url"] = str(getattr(page, "url", "") or "")
    title_fn = getattr(page, "title", None)
    if callable(title_fn):
        try:
            snapshot["title"] = str(title_fn() or "")
        except Exception as exc:
            snapshot["title_error"] = str(exc)
    else:
        snapshot["title"] = str(getattr(page, "title", "") or "")

    try:
        locator_factory = getattr(page, "locator", None)
        if callable(locator_factory):
            body_text = str(locator_factory("body").inner_text(timeout=1_000) or "")
            snapshot["body_len"] = len(body_text.strip())
            snapshot["body_excerpt"] = body_text.strip()[:500]
    except Exception as exc:
        snapshot["body_error"] = str(exc)

    try:
        evaluate = getattr(page, "evaluate", None)
        if callable(evaluate):
            element_count = evaluate("() => document.querySelectorAll('*').length")
            snapshot["element_count"] = int(element_count)
    except Exception as exc:
        snapshot["element_count_error"] = str(exc)

    try:
        is_closed = getattr(page, "is_closed", None)
        if callable(is_closed):
            snapshot["closed"] = bool(is_closed())
    except Exception:
        pass
    return snapshot


def record_browser_snapshot(
    diagnostics: dict[str, Any] | None,
    page: Any | None,
    *,
    label: str,
) -> None:
    if diagnostics is None:
        return
    payload = {"label": label, "timestamp_monotonic_s": round(time.monotonic(), 3)}
    payload.update(browser_page_snapshot(page))
    _append_limited(diagnostics.setdefault("snapshots", []), payload)


def record_startup_phase(
    diagnostics: dict[str, Any] | None,
    name: str,
    started_at: float,
    *,
    status: str = "ok",
    exc: BaseException | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if diagnostics is None:
        return
    phase: dict[str, Any] = {
        "name": name,
        "status": status,
        "duration_s": round(time.monotonic() - started_at, 3),
    }
    if exc is not None:
        phase["error"] = str(exc)
        phase["error_type"] = type(exc).__name__
        category = getattr(exc, "category", None)
        if category:
            phase["error_category"] = category
    if extra:
        phase.update(extra)
    diagnostics.setdefault("phases", []).append(phase)


def _install_browser_diagnostics(page: Any, diagnostics: dict[str, Any] | None) -> None:
    if diagnostics is None:
        return
    try:
        if getattr(page, "_asil_startup_diagnostics_installed", False):
            return
        setattr(page, "_asil_startup_diagnostics_installed", True)
    except Exception:
        pass
    on = getattr(page, "on", None)
    if not callable(on):
        return
    events = diagnostics.setdefault(
        "browser_events",
        {"console": [], "pageerror": [], "requestfailed": []},
    )
    try:
        on(
            "console",
            lambda msg: _append_limited(
                events.setdefault("console", []),
                {"type": getattr(msg, "type", ""), "text": getattr(msg, "text", "")},
            ),
        )
        on(
            "pageerror",
            lambda exc: _append_limited(events.setdefault("pageerror", []), str(exc)),
        )
        on(
            "requestfailed",
            lambda req: _append_limited(
                events.setdefault("requestfailed", []),
                {
                    "url": getattr(req, "url", ""),
                    "failure": str(
                        getattr(req, "failure", None)()
                        if callable(getattr(req, "failure", None))
                        else getattr(req, "failure", "")
                    ),
                },
            ),
        )
    except Exception:
        return


def _startup_phase_timeout_s(spec: GUISessionSpec) -> float:
    if spec.surface_type == "browser":
        return min(max(spec.startup_timeout_s * 2, 60.0), 240.0)
    return min(max(spec.startup_timeout_s, 10.0), 180.0)


def _invoke_session_callback(callback, session: GUISession) -> Any:
    if callback is None:
        return None
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(session)

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
    )
    if accepts_varargs or positional_params:
        return callback(session)
    return callback()


def _normalize_startup_error(exc: Exception) -> GUISessionStartupError:
    if isinstance(exc, GUISessionStartupError):
        return exc
    message = str(exc)
    if "Timed out waiting for window matching" in message:
        return GUISessionStartupError("window_timeout", message)
    return GUISessionStartupError("window_timeout", message)


def _session_process_names(spec: GUISessionSpec) -> tuple[str, ...]:
    if spec.surface_type == "multi_window":
        names: list[str] = []
        for child_spec in spec.child_specs.values():
            names.extend(_session_process_names(child_spec))
        return tuple(dict.fromkeys(names))
    if spec.surface_type == "browser":
        return _BROWSER_PROCESS_NAMES
    if not spec.launch_command:
        return ()
    command_name = Path(str(spec.launch_command[0])).name
    if command_name in _DESKTOP_PROCESS_PATTERN_OVERRIDES:
        return _DESKTOP_PROCESS_PATTERN_OVERRIDES[command_name]
    return (command_name,)


def _cleanup_gui_processes(spec: GUISessionSpec) -> None:
    pkill = shutil.which("pkill")
    if pkill is None:
        return
    user_args: list[str] = []
    if spec.run_as_user:
        user_args = ["-u", spec.run_as_user]
    for process_name in _session_process_names(spec):
        subprocess.run(
            [pkill, *user_args, "-TERM", "-x", process_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(0.2)
    for process_name in _session_process_names(spec):
        subprocess.run(
            [pkill, *user_args, "-KILL", "-x", process_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _prelaunch_cleanup_spec(spec: GUISessionSpec) -> GUISessionSpec:
    if spec.surface_type != "multi_window":
        return spec
    return GUISessionSpec(
        surface_type="multi_window",
        window_title_pattern=spec.window_title_pattern,
        window_class_pattern=spec.window_class_pattern,
        child_specs=spec.child_specs,
    )


def _browser_page_failure_category(page: Any) -> str | None:
    url = str(getattr(page, "url", "") or "").lower()
    if url.startswith("chrome-error://"):
        return "browser_crashed"
    try:
        content = str(page.content() or "")
    except Exception as exc:
        if "target crashed" in str(exc).lower():
            return "browser_crashed"
        content = ""
    lowered = content.lower()
    if "aw, snap" in lowered or "page crashed" in lowered or "chrome-error" in lowered:
        return "browser_crashed"
    body_text = ""
    locator_factory = getattr(page, "locator", None)
    if callable(locator_factory):
        try:
            body_text = str(locator_factory("body").inner_text(timeout=1_000) or "").strip()
        except Exception as exc:
            if "target crashed" in str(exc).lower():
                return "browser_crashed"
            body_text = ""
    elif "<body" in lowered:
        body_text = re.sub(r"<[^>]+>", " ", lowered).strip()
    if not body_text:
        return "blank_shell"
    return None


def _browser_body_ready_script() -> str:
    return (
        "() => {"
        " const body = document.body;"
        " if (!body) return false;"
        " const text = (body.innerText || '').trim();"
        " return text.length > 0 || body.querySelectorAll('*').length > 0;"
        " }"
    )


def _assert_browser_page_ready(
    session: GUISession,
    *,
    required_selectors: tuple[str, ...] = (),
    ready_script: str | None = None,
    app_name: str,
    timeout_ms: int = 45_000,
) -> None:
    page = session.browser_page
    if page is None:
        raise GUISessionStartupError("browser_crashed", f"{app_name} browser page is not available.")

    failure_category = _browser_page_failure_category(page)
    if failure_category == "browser_crashed":
        raise GUISessionStartupError(failure_category, f"{app_name} page failed before it became ready.")

    try:
        for selector in required_selectors:
            if selector == "body":
                if hasattr(page, "wait_for_function"):
                    page.wait_for_function(_browser_body_ready_script(), timeout=timeout_ms)
                else:
                    page.wait_for_selector("body", timeout=timeout_ms)
            else:
                page.wait_for_selector(selector, timeout=timeout_ms)
        if ready_script:
            if hasattr(page, "wait_for_function"):
                page.wait_for_function(ready_script, timeout=timeout_ms)
            else:
                raise AttributeError("wait_for_function is required for ready_script checks")
    except Exception as exc:
        failure_category = _browser_page_failure_category(page)
        if failure_category is not None:
            raise GUISessionStartupError(failure_category, f"{app_name} page failed before it became ready.") from exc
        raise GUISessionStartupError(
            "window_timeout",
            f"{app_name} UI did not become ready within {timeout_ms}ms.",
        ) from exc

    failure_category = _browser_page_failure_category(page)
    if failure_category == "browser_crashed":
        raise GUISessionStartupError(failure_category, f"{app_name} page failed before it became ready.")


def navigate_browser_target(
    session: GUISession,
    target_url: str,
    *,
    timeout_ms: int = 60_000,
) -> None:
    """Navigate a browser session to a target URL using a fresh page when possible.

    Reusing the initial shell page for a deep-link navigation proved brittle in
    long-running benchmark sessions. When a browser context is available, open
    the target in a new page and only replace the session page after navigation
    has committed. Fall back to the current page for unit-test fakes or
    adapters without a real browser context.
    """
    current_page = session.browser_page
    if current_page is None:
        raise RuntimeError("Browser page is not available for target navigation.")
    if str(getattr(current_page, "url", "")).startswith(target_url):
        return

    diagnostics = getattr(session, "startup_diagnostics", None)
    phase_started = time.monotonic()
    spec = getattr(session, "spec", None)
    navigation_mode = str(getattr(spec, "browser_navigation_mode", "fresh_page") or "fresh_page")
    post_navigation_settle_ms = int(getattr(spec, "browser_post_navigation_settle_ms", 0) or 0)
    browser_context = getattr(session, "browser_context", None)
    try:
        if browser_context is None or navigation_mode == "current_page":
            current_page.goto(target_url, wait_until="commit", timeout=timeout_ms)
            try:
                current_page.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 15_000))
            except Exception:
                pass
            if post_navigation_settle_ms > 0:
                try:
                    current_page.wait_for_timeout(post_navigation_settle_ms)
                except Exception:
                    pass
            record_browser_snapshot(diagnostics, current_page, label="after_navigate")
            record_startup_phase(diagnostics, "navigate", phase_started, extra={"target_url": target_url})
            return

        new_page = browser_context.new_page()
        _install_browser_diagnostics(new_page, diagnostics)
        try:
            new_page.goto(target_url, wait_until="commit", timeout=timeout_ms)
            try:
                new_page.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 15_000))
            except Exception:
                pass
            if post_navigation_settle_ms > 0:
                try:
                    new_page.wait_for_timeout(post_navigation_settle_ms)
                except Exception:
                    pass
            try:
                new_page.bring_to_front()
            except Exception:
                pass
            session.browser_page = new_page
            try:
                current_page.close()
            except Exception:
                pass
            record_browser_snapshot(diagnostics, new_page, label="after_navigate")
            record_startup_phase(diagnostics, "navigate", phase_started, extra={"target_url": target_url})
        except Exception:
            try:
                new_page.close()
            except Exception:
                pass
            raise
    except Exception as exc:
        record_startup_phase(
            diagnostics,
            "navigate",
            phase_started,
            status="error",
            exc=exc,
            extra={"target_url": target_url},
        )
        raise


def _browser_url_for_adapter(adapter: ASILAdapter) -> str:
    class_name = type(adapter).__name__
    if class_name == "GiteaAdapter":
        return f"{adapter.base_url}{adapter._current_ui_path}"
    if class_name == "DrawioAdapter":
        return adapter._live_editor_url()
    if class_name == "CodeServerAdapter":
        return (
            f"{adapter.base_url}/?folder={quote(str(adapter.workspace_path), safe='')}"
            f"&file={quote(str(adapter.workspace_path / adapter._active_file), safe='')}"
        )
    if class_name == "JupyterLabAdapter":
        return f"{adapter.base_url}/lab/tree/{quote(adapter._active_file)}"
    raise RuntimeError(f"Browser URL resolution is not implemented for {class_name}.")


def resolve_gui_session_spec(adapter: ASILAdapter) -> GUISessionSpec:
    explicit = adapter.get_gui_session_spec()
    if explicit is not None:
        return explicit

    class_name = type(adapter).__name__
    source_path = getattr(adapter, "source_path", None)
    if callable(source_path):
        source_path = source_path()

    if class_name == "InkscapeAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("inkscape", str(adapter.source_path)),
            window_title_pattern=r".*Inkscape|.* - Inkscape",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "LibreOfficeAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("libreoffice", "--calc", str(adapter.source_path)),
            window_title_pattern=r".*LibreOffice Calc|.* - LibreOffice Calc",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "BlenderAdapter":
        adapter._ensure_workfile()
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(adapter.blender_bin, "--factory-startup", str(adapter.blend_path)),
            window_title_pattern="Blender",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "OBSAdapter":
        # A first observe() call will launch real OBS when enabled.
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(),
            window_title_pattern="OBS",
            min_width=900,
            min_height=600,
        )
    if class_name == "GimpAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("gimp", "--new-instance", "--no-splash", str(adapter.image_path)),
            window_title_pattern="GIMP|GNU Image Manipulation Program",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "LibreOfficeWriterAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("libreoffice", "--writer", str(adapter.source_path)),
            window_title_pattern=r".*LibreOffice Writer|.* - LibreOffice Writer",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "LibreOfficeImpressAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("libreoffice", "--impress", str(adapter.source_path)),
            window_title_pattern=r".*LibreOffice Impress|.* - LibreOffice Impress",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "ThunderbirdAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("thunderbird",),
            window_title_pattern=r".*Thunderbird.*",
            run_as_user="asilgui",
        )
    if class_name == "NautilusAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("nautilus", "--new-window", str(adapter.workspace_path)),
            window_title_pattern=r".*",
            window_class_pattern=r"org.gnome.Nautilus|Org.gnome.Nautilus|nautilus",
            run_as_user="asilgui",
        )
    if class_name == "KdenliveAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("kdenlive", str(adapter.source_path)),
            window_title_pattern=r".*Kdenlive.*",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name == "AudacityAdapter":
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=("audacity", str(adapter.source_path)),
            window_title_pattern=r".*Audacity.*",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )
    if class_name in {"CodeServerAdapter", "JupyterLabAdapter", "DrawioAdapter", "GiteaAdapter"}:
        return GUISessionSpec(
            surface_type="browser",
            browser_url=_browser_url_for_adapter(adapter),
            launch_command=(),
            window_title_pattern=r".*",
            window_class_pattern=_BROWSER_CLASS_PATTERN,
            min_width=1000,
            min_height=700,
        )

    raise RuntimeError(f"Real GUI session spec is not implemented for {class_name}.")


def start_gui_session(
    spec: GUISessionSpec,
    startup_diagnostics: dict[str, Any] | None = None,
) -> GUISession:
    last_error: GUISessionStartupError | None = None
    phase_timeout_s = _startup_phase_timeout_s(spec)
    diagnostics = startup_diagnostics
    for attempt in range(1, 4):
        session: GUISession | None = None
        try:
            if diagnostics is not None:
                diagnostics["attempt"] = attempt
            if spec.backend_ready_probe is not None:
                phase_started = time.monotonic()
                with _StartupTimeoutGuard(
                    phase_timeout_s,
                    "backend probe timed out.",
                ):
                    spec.backend_ready_probe()
                record_startup_phase(diagnostics, "backend_probe", phase_started)

            # Start every task from a fresh display/session state so previous GUI
            # apps cannot leak window-manager or browser state into the next one.
            _cleanup_gui_processes(_prelaunch_cleanup_spec(spec))
            stop_virtual_display()

            if spec.surface_type == "desktop":
                phase_started = time.monotonic()
                session = _launch_desktop_session(spec)
                session.startup_diagnostics = diagnostics
                record_startup_phase(diagnostics, "desktop_launch", phase_started)
            elif spec.surface_type == "browser":
                session = _launch_browser_session_with_diagnostics(spec, diagnostics)
            elif spec.surface_type == "multi_window":
                session = _launch_multi_window_session(spec, diagnostics)
            else:
                raise ValueError(f"Unsupported GUI surface_type: {spec.surface_type}")

            if spec.post_launch_delay_s > 0:
                time.sleep(spec.post_launch_delay_s)
            if spec.post_launch_callback is not None:
                phase_started = time.monotonic()
                with _StartupTimeoutGuard(
                    phase_timeout_s,
                    "post-launch callback timed out.",
                ):
                    _invoke_session_callback(spec.post_launch_callback, session)
                record_startup_phase(diagnostics, "post_launch_callback", phase_started)
            if spec.ui_ready_probe is not None:
                phase_started = time.monotonic()
                with _StartupTimeoutGuard(
                    phase_timeout_s,
                    "UI readiness probe timed out.",
                ):
                    _invoke_session_callback(spec.ui_ready_probe, session)
                record_startup_phase(diagnostics, "ui_ready_probe", phase_started)
            return session
        except Exception as exc:
            if isinstance(exc, _StartupPhaseTimedOut):
                last_error = GUISessionStartupError("window_timeout", str(exc))
            else:
                last_error = _normalize_startup_error(exc)
            record_startup_phase(
                diagnostics,
                "attempt",
                time.monotonic(),
                status="error",
                exc=last_error,
                extra={"attempt": attempt},
            )
            if session is not None:
                session.close()
            else:
                stop_virtual_display()
            time.sleep(1.0)

    assert last_error is not None
    raise last_error


def _run_post_launch_readiness(spec: GUISessionSpec, session: GUISession, diagnostics: dict[str, Any] | None) -> None:
    phase_timeout_s = _startup_phase_timeout_s(spec)
    if spec.post_launch_delay_s > 0:
        time.sleep(spec.post_launch_delay_s)
    if spec.post_launch_callback is not None:
        phase_started = time.monotonic()
        with _StartupTimeoutGuard(phase_timeout_s, "post-launch callback timed out."):
            _invoke_session_callback(spec.post_launch_callback, session)
        record_startup_phase(diagnostics, "post_launch_callback", phase_started)
    if spec.ui_ready_probe is not None:
        phase_started = time.monotonic()
        with _StartupTimeoutGuard(phase_timeout_s, "UI readiness probe timed out."):
            _invoke_session_callback(spec.ui_ready_probe, session)
        record_startup_phase(diagnostics, "ui_ready_probe", phase_started)


def _launch_child_session(
    spec: GUISessionSpec,
    diagnostics: dict[str, Any] | None,
    *,
    browser_playwright: Any | None = None,
) -> GUISession:
    session: GUISession | None = None
    if spec.backend_ready_probe is not None:
        phase_started = time.monotonic()
        with _StartupTimeoutGuard(_startup_phase_timeout_s(spec), "backend probe timed out."):
            spec.backend_ready_probe()
        record_startup_phase(diagnostics, "backend_probe", phase_started)
    try:
        if spec.surface_type == "desktop":
            phase_started = time.monotonic()
            session = _launch_desktop_session(spec)
            session.startup_diagnostics = diagnostics
            record_startup_phase(diagnostics, "desktop_launch", phase_started)
        elif spec.surface_type == "browser":
            if browser_playwright is not None:
                session = _launch_browser_session(
                    spec,
                    startup_diagnostics=diagnostics,
                    playwright=browser_playwright,
                )
            else:
                session = _launch_browser_session_with_diagnostics(spec, diagnostics)
        else:
            raise ValueError(f"Unsupported child GUI surface_type: {spec.surface_type}")
        _run_post_launch_readiness(spec, session, diagnostics)
        return session
    except Exception:
        if session is not None:
            session.close()
        raise


def _launch_multi_window_session(
    spec: GUISessionSpec,
    diagnostics: dict[str, Any] | None,
) -> GUISession:
    if not spec.child_specs:
        raise ValueError("multi_window GUI sessions require child_specs.")

    parent = GUISession(spec=spec, startup_diagnostics=diagnostics)
    children_diag = diagnostics.setdefault("children", {}) if diagnostics is not None else {}
    launched_apps: list[str] = []
    shared_playwright = None
    browser_child_count = sum(
        1 for child_spec in spec.child_specs.values() if child_spec.surface_type == "browser"
    )
    try:
        launch_items = sorted(
            spec.child_specs.items(),
            key=lambda item: 0 if item[1].surface_type == "browser" else 1,
        )
        for app, child_spec in launch_items:
            child_diag = create_startup_diagnostics(child_spec)
            if isinstance(children_diag, dict):
                children_diag[app] = child_diag
            phase_started = time.monotonic()
            if child_spec.surface_type == "browser" and browser_child_count > 1 and shared_playwright is None:
                from playwright.sync_api import sync_playwright

                shared_playwright = sync_playwright().start()
                parent.playwright = shared_playwright
            child_session = _launch_child_session(
                child_spec,
                child_diag,
                browser_playwright=shared_playwright if child_spec.surface_type == "browser" else None,
            )
            parent.child_sessions[app] = child_session
            launched_apps.append(app)
            record_startup_phase(
                diagnostics,
                "child_start",
                phase_started,
                extra={"app": app, "surface_type": child_spec.surface_type},
            )

        primary = spec.primary_child or launched_apps[0]
        primary_spec = spec.child_specs.get(primary)
        if primary_spec is not None:
            phase_started = time.monotonic()
            parent.active_child = primary
            try:
                window_id = activate_window(
                    primary_spec.window_title_pattern,
                    window_class_pattern=primary_spec.window_class_pattern,
                    timeout=primary_spec.startup_timeout_s,
                    min_width=primary_spec.min_width,
                    min_height=primary_spec.min_height,
                )
                parent.last_capture_window_id = str(window_id)
                record_startup_phase(diagnostics, "primary_window_activate", phase_started, extra={"app": primary})
            except Exception as exc:
                record_startup_phase(
                    diagnostics,
                    "primary_window_activate",
                    phase_started,
                    status="warning",
                    exc=exc,
                    extra={"app": primary, "fallback": "capture_phase_main_window_recovery"},
                )
        return parent
    except Exception:
        parent.close()
        raise


def _launch_browser_session_with_diagnostics(
    spec: GUISessionSpec,
    diagnostics: dict[str, Any] | None,
) -> GUISession:
    signature = inspect.signature(_launch_browser_session)
    supports_diagnostics = "startup_diagnostics" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if supports_diagnostics:
        return _launch_browser_session(spec, startup_diagnostics=diagnostics)
    session = _launch_browser_session(spec)
    session.startup_diagnostics = diagnostics
    return session


def _launch_desktop_session(spec: GUISessionSpec) -> GUISession:
    process = None
    if spec.launch_command:
        if spec.cwd is not None:
            ensure_user_access(spec.cwd, run_as_user=spec.run_as_user)
        for env_key, env_value in spec.extra_env.items():
            if not env_value:
                continue
            if env_key.endswith(("_HOME", "_DIR")):
                candidate = Path(env_value)
                candidate.mkdir(parents=True, exist_ok=True)
                ensure_user_access(candidate, run_as_user=spec.run_as_user)
        for argument in spec.launch_command:
            try:
                candidate = Path(argument)
            except TypeError:
                continue
            if candidate.exists():
                if candidate.is_absolute() and any(
                    candidate == prefix or candidate.is_relative_to(prefix)
                    for prefix in _SYSTEM_PATH_PREFIXES
                ):
                    continue
                ensure_user_access(candidate, run_as_user=spec.run_as_user)
        process = launch_gui_process(
            list(spec.launch_command),
            cwd=spec.cwd,
            extra_env=spec.extra_env,
            run_as_user=spec.run_as_user,
        )
    return GUISession(spec=spec, process=process)


def _launch_browser_session(
    spec: GUISessionSpec,
    *,
    startup_diagnostics: dict[str, Any] | None = None,
    playwright: Any | None = None,
) -> GUISession:
    if playwright is None:
        from playwright.sync_api import sync_playwright

    browser_env = os.environ.copy()
    browser_env.update(ensure_virtual_display(run_as_user=spec.run_as_user))
    if spec.extra_env:
        browser_env.update(spec.extra_env)

    temp_dir = tempfile.TemporaryDirectory(prefix="asil_gui_browser_")
    owns_playwright = playwright is None
    pw = playwright
    context = None
    launch_timeout_ms = int(max(spec.startup_timeout_s, 30.0) * 1000)
    phase_started = time.monotonic()
    try:
        if pw is None:
            pw = sync_playwright().start()
        context = pw.chromium.launch_persistent_context(
            user_data_dir=temp_dir.name,
            headless=False,
            env=browser_env,
            timeout=launch_timeout_ms,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-accelerated-2d-canvas",
                "--disable-software-rasterizer",
                "--window-size=1360,900",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-search-engine-choice-screen",
                "--disable-features=ChromeWhatsNewUI,HelpAppWelcomeTips,UseSkiaRenderer,Vulkan",
                "--use-gl=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
            viewport={"width": 1280, "height": 800},
        )
        try:
            context.set_default_timeout(5_000)
            context.set_default_navigation_timeout(min(launch_timeout_ms, 60_000))
        except Exception:
            pass
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.set_default_timeout(5_000)
            page.set_default_navigation_timeout(min(launch_timeout_ms, 60_000))
        except Exception:
            pass
        _install_browser_diagnostics(page, startup_diagnostics)
        session = GUISession(
            spec=spec,
            browser_context=context,
            browser_page=page,
            temp_dir=temp_dir,
            playwright=pw if owns_playwright else None,
            startup_diagnostics=startup_diagnostics,
        )
        record_startup_phase(startup_diagnostics, "browser_launch", phase_started)
        record_browser_snapshot(startup_diagnostics, page, label="after_browser_launch")
        target_url = str(spec.browser_url or "").strip()
        if target_url and target_url != "about:blank":
            navigate_browser_target(session, target_url, timeout_ms=45_000)
            page = session.browser_page
        try:
            page.wait_for_load_state("networkidle", timeout=min(launch_timeout_ms, 15_000))
        except Exception:
            pass
        return session
    except Exception as exc:
        record_startup_phase(startup_diagnostics, "browser_launch", phase_started, status="error", exc=exc)
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if owns_playwright and pw is not None:
            try:
                pw.stop()
            except Exception:
                pass
        temp_dir.cleanup()
        raise
