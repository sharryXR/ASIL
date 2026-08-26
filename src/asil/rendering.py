"""Helpers for per-step page rendering artifacts."""

from __future__ import annotations

import html
import os
import pwd
import re
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw


@dataclass(slots=True)
class RenderArtifact:
    """Metadata for a per-step visual artifact."""

    filename: str
    kind: str
    backend: str
    actual_page: bool
    description: str
    capture_complete: bool = True

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def capture_html_to_png(
    html_content: str,
    output_path: str | Path,
    *,
    width: int = 1440,
    height: int = 1024,
) -> Path:
    """Render HTML content to a PNG via wkhtmltoimage."""
    tool = shutil.which("wkhtmltoimage")
    if tool is None:
        raise RuntimeError(
            "wkhtmltoimage is not installed. Install wkhtmltopdf/wkhtmltoimage to render page screenshots."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as handle:
        handle.write(html_content)
        html_file = Path(handle.name)

    try:
        subprocess.run(
            [
                tool,
                "--enable-local-file-access",
                "--width",
                str(width),
                "--height",
                str(height),
                str(html_file),
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        html_file.unlink(missing_ok=True)

    return out


def capture_url_to_png(
    url: str,
    output_path: str | Path,
    *,
    width: int = 1440,
    height: int = 1024,
    wait_for_selectors: Sequence[str] | None = None,
    optional_click_selectors: Sequence[str] | None = None,
    double_click_selectors: Sequence[str] | None = None,
    click_selectors: Sequence[str] | None = None,
    keyboard_steps: Sequence[dict[str, Any]] | None = None,
    wait_for_selectors_after_actions: Sequence[str] | None = None,
    ready_script: str | None = None,
    backend: str = "playwright",
    timeout_ms: int = 30_000,
    full_page: bool = True,
    reject_blank: bool = True,
    initial_wait_ms: int = 0,
) -> Path:
    """Capture a live webpage to a PNG via a real browser."""
    if backend != "playwright":
        raise ValueError(f"Unsupported webpage capture backend: {backend}")

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "playwright is required for live webpage screenshots. Install the package and browser runtime."
        ) from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    attempts = 2
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--no-sandbox",
                    ],
                )
                try:
                    page = browser.new_page(viewport={"width": width, "height": height})
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10_000))
                    except PlaywrightTimeoutError:
                        pass
                    if initial_wait_ms > 0:
                        page.wait_for_timeout(initial_wait_ms)
                    for selector in wait_for_selectors or ():
                        page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
                    for selector in optional_click_selectors or ():
                        try:
                            page.locator(selector).first.click(timeout=1_500)
                            page.wait_for_timeout(500)
                        except PlaywrightTimeoutError:
                            continue
                    for selector in click_selectors or ():
                        page.locator(selector).first.click(timeout=timeout_ms)
                        page.wait_for_timeout(500)
                    for selector in double_click_selectors or ():
                        page.locator(selector).first.dblclick(timeout=timeout_ms)
                    for step in keyboard_steps or ():
                        if "press" in step:
                            page.keyboard.press(str(step["press"]))
                        if "type" in step:
                            page.keyboard.type(str(step["type"]))
                        if "wait_ms" in step:
                            page.wait_for_timeout(int(step["wait_ms"]))
                    for selector in wait_for_selectors_after_actions or ():
                        page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
                    if ready_script:
                        page.wait_for_function(ready_script, timeout=timeout_ms)
                    page.screenshot(path=str(out), full_page=full_page)
                finally:
                    browser.close()

            if reject_blank:
                assert_png_not_blank(out)
            return out
        except Exception as exc:
            last_exc = exc
            message = str(exc)
            if attempt < attempts and ("Target crashed" in message or "page crashed" in message):
                time.sleep(1)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    return out


def rasterize_pdf_first_page(pdf_path: str | Path, output_path: str | Path) -> Path:
    """Convert the first PDF page to PNG via pdftoppm."""
    return rasterize_pdf_pages(pdf_path, output_path, [1])


def rasterize_pdf_pages(
    pdf_path: str | Path,
    output_path: str | Path,
    page_numbers: Sequence[int],
) -> Path:
    """Convert one or more PDF pages to PNG via pdftoppm.

    When multiple pages are requested, renders each page separately and combines
    them into a simple vertical contact sheet so downstream consumers still get a
    single `step_i.png`.
    """
    tool = shutil.which("pdftoppm")
    if tool is None:
        raise RuntimeError(
            "pdftoppm is not installed. Install poppler-utils to rasterize PDF pages."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = [int(page) for page in page_numbers if int(page) > 0]
    if not pages:
        raise ValueError("page_numbers must contain at least one positive page number.")

    if len(pages) == 1:
        prefix = out.with_suffix("")
        subprocess.run(
            [
                tool,
                "-png",
                "-singlefile",
                "-f",
                str(pages[0]),
                "-l",
                str(pages[0]),
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
        )
        return out

    rendered_pages: list[Path] = []
    try:
        for index, page in enumerate(pages, start=1):
            page_path = out.with_name(f"{out.stem}.page{index}.png")
            prefix = page_path.with_suffix("")
            subprocess.run(
                [
                    tool,
                    "-png",
                    "-singlefile",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
            )
            rendered_pages.append(page_path)

        images = [Image.open(path).convert("RGB") for path in rendered_pages]
        try:
            max_width = max(image.width for image in images)
            total_height = sum(image.height for image in images)
            contact_sheet = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
            y = 0
            for image in images:
                contact_sheet.paste(image, (0, y))
                y += image.height
            contact_sheet.save(out)
        finally:
            for image in images:
                image.close()
    finally:
        for path in rendered_pages:
            path.unlink(missing_ok=True)
    return out


def assert_png_not_blank(image_path: str | Path) -> None:
    """Reject screenshots that are visually blank placeholders."""
    with Image.open(image_path) as image:
        sample = image.convert("RGB")
        sample.thumbnail((256, 256))
        colors = sample.getcolors(maxcolors=sample.width * sample.height)
        if not colors:
            return

        dominant = max(count for count, _color in colors)
        total_pixels = sample.width * sample.height
        dominant_ratio = dominant / float(total_pixels)

        if len(colors) == 1 or dominant_ratio >= 0.995:
            raise RuntimeError(
                f"Rendered screenshot at {image_path} is blank or nearly blank and cannot count as a real GUI/page capture."
            )


def html_page(title: str, body: str) -> str:
    """Wrap body HTML in a stable page frame for screenshotting."""
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eff3f6;
      --panel: #ffffff;
      --line: #d0d7de;
      --muted: #57606a;
      --text: #24292f;
      --accent: #0969da;
      --success: #1a7f37;
      --danger: #cf222e;
      --shadow: 0 10px 30px rgba(31, 35, 40, 0.08);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top right, rgba(9, 105, 218, 0.08), transparent 28%),
        linear-gradient(180deg, #f6f8fa 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .frame {{
      width: 1360px;
      min-height: 960px;
      margin: 32px auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .topbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: #f6f8fa;
      font-size: 14px;
      color: var(--muted);
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: #d8dee4;
    }}
    .content {{
      padding: 24px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 12px;
      color: var(--muted);
    }}
    .pill-success {{ color: var(--success); }}
    .pill-danger {{ color: var(--danger); }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      box-shadow: 0 4px 16px rgba(31, 35, 40, 0.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f6f8fa;
      color: var(--muted);
      font-weight: 600;
    }}
    td code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="topbar">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      <span>{safe_title}</span>
    </div>
    <div class="content">{body}</div>
  </div>
</body>
</html>
"""


_XVFB_PID_FILE = Path("/tmp/asil_xvfb.pid")
_OPENBOX_PID_FILE = Path("/tmp/asil_openbox.pid")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except OSError:
        return False
    if len(fields) >= 3 and fields[2] == "Z":
        return False
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _lookup_user(run_as_user: str | None):
    if not run_as_user:
        return None
    try:
        return pwd.getpwnam(run_as_user)
    except KeyError:
        return None


def _ensure_runtime_dir(*, run_as_user: str | None = None) -> Path:
    suffix = f"_{run_as_user}" if run_as_user else ""
    runtime_dir = Path(f"/tmp/asil_xdg_runtime{suffix}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime_dir.chmod(0o700)
    except OSError:
        pass
    if os.geteuid() == 0:
        user = _lookup_user(run_as_user)
        if user is not None:
            try:
                os.chown(runtime_dir, user.pw_uid, user.pw_gid)
            except OSError:
                pass
    return runtime_dir


def ensure_virtual_display(
    display: str | None = None,
    *,
    screen: str = "1920x1080x24",
    run_as_user: str | None = None,
) -> dict[str, str]:
    """Ensure an Xvfb + Openbox session is available for GUI screenshots."""
    screen = os.environ.get("ASIL_XVFB_SCREEN", "").strip() or screen
    disp = display or os.environ.get("DISPLAY") or os.environ.get("ASIL_XVFB_DISPLAY", ":99")
    runtime_dir = _ensure_runtime_dir(run_as_user=run_as_user)
    user = _lookup_user(run_as_user)

    env = os.environ.copy()
    env["DISPLAY"] = disp
    env["ASIL_XVFB_SCREEN"] = screen
    if run_as_user:
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    else:
        env.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
    if user is not None:
        home = getattr(user, "pw_dir", None)
        name = getattr(user, "pw_name", run_as_user)
        if home:
            env["HOME"] = home
        if name:
            env["USER"] = name
            env["LOGNAME"] = name

    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is None:
        raise RuntimeError("xdpyinfo is required for GUI rendering but is not installed.")

    display_ready = False
    try:
        subprocess.run(
            [xdpyinfo, "-display", disp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        display_ready = True
    except subprocess.CalledProcessError:
        display_ready = False

    if not display_ready:
        xvfb = shutil.which("Xvfb")
        if xvfb is None:
            raise RuntimeError("Xvfb is required for GUI rendering but is not installed.")
        proc = subprocess.Popen(
            [xvfb, disp, "-screen", "0", screen, "-ac", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        _XVFB_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Xvfb exited before the virtual display became ready.")
            try:
                subprocess.run(
                    [xdpyinfo, "-display", disp],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
                break
            except subprocess.CalledProcessError:
                time.sleep(0.25)
        else:
            raise RuntimeError("Timed out waiting for Xvfb to become ready.")

    openbox = shutil.which("openbox")
    if openbox is not None:
        pid = _read_pid(_OPENBOX_PID_FILE)
        if pid is None or not _pid_is_alive(pid):
            proc = subprocess.Popen(
                [openbox],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            _OPENBOX_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
            time.sleep(1.0)

    return {
        "DISPLAY": disp,
        "XDG_RUNTIME_DIR": env["XDG_RUNTIME_DIR"],
        "ASIL_XVFB_SCREEN": screen,
    }


def stop_virtual_display() -> None:
    """Best-effort cleanup of the managed Xvfb/Openbox session."""
    for pid_file in (_OPENBOX_PID_FILE, _XVFB_PID_FILE):
        pid = _read_pid(pid_file)
        if pid is None:
            pid_file.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, 15)
        except OSError:
            pid_file.unlink(missing_ok=True)
            continue
        deadline = time.time() + 5.0
        while time.time() < deadline and _pid_is_alive(pid):
            time.sleep(0.1)
        if _pid_is_alive(pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        pid_file.unlink(missing_ok=True)


def launch_gui_process(
    command: list[str],
    *,
    display: str | None = None,
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
    run_as_user: str | None = None,
) -> subprocess.Popen[str]:
    """Launch a GUI process inside the managed virtual display."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display, run_as_user=run_as_user))
    if extra_env:
        gui_env.update(extra_env)
    final_command = list(command)
    user = _lookup_user(run_as_user)
    if user is not None:
        if not extra_env or "HOME" not in extra_env:
            gui_env["HOME"] = user.pw_dir
        gui_env["USER"] = run_as_user
        gui_env["LOGNAME"] = run_as_user
    if run_as_user and os.geteuid() == 0:
        if user is not None:
            runuser = shutil.which("runuser")
            if runuser is not None:
                final_command = [runuser, "--preserve-environment", "-u", run_as_user, "--", *final_command]
    return subprocess.Popen(
        final_command,
        cwd=str(cwd) if cwd is not None else None,
        env=gui_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )


def ensure_audio_backend(*, run_as_user: str | None = None) -> None:
    """Best-effort audio backend setup for desktop apps that assert without a playback device."""
    pulseaudio = shutil.which("pulseaudio")
    if pulseaudio is None:
        return

    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(run_as_user=run_as_user))
    user = _lookup_user(run_as_user)
    if user is not None:
        gui_env["HOME"] = user.pw_dir
        gui_env["USER"] = run_as_user
        gui_env["LOGNAME"] = run_as_user

    def _wrap(command: list[str]) -> list[str]:
        final_command = list(command)
        if run_as_user and os.geteuid() == 0:
            if user is None:
                return final_command
            runuser = shutil.which("runuser")
            if runuser is not None:
                return [runuser, "--preserve-environment", "-u", run_as_user, "--", *final_command]
        return final_command

    check = subprocess.run(
        _wrap([pulseaudio, "--check"]),
        capture_output=True,
        env=gui_env,
    )
    if check.returncode != 0:
        subprocess.run(
            _wrap([pulseaudio, "--start", "--exit-idle-time=-1"]),
            check=False,
            capture_output=True,
            env=gui_env,
        )
        time.sleep(1.0)

    pactl = shutil.which("pactl")
    if pactl is None:
        return
    sinks = subprocess.run(
        _wrap([pactl, "list", "short", "sinks"]),
        check=False,
        capture_output=True,
        text=True,
        env=gui_env,
    )
    if "asil-null" not in sinks.stdout:
        subprocess.run(
            _wrap([pactl, "load-module", "module-null-sink", "sink_name=asil-null"]),
            check=False,
            capture_output=True,
            env=gui_env,
        )
    subprocess.run(
        _wrap([pactl, "set-default-sink", "asil-null"]),
        check=False,
        capture_output=True,
        env=gui_env,
    )
    subprocess.run(
        _wrap([pactl, "set-default-source", "asil-null.monitor"]),
        check=False,
        capture_output=True,
        env=gui_env,
    )


def ensure_user_access(path: str | Path, *, run_as_user: str | None) -> None:
    """Best-effort ownership/permission fixup so GUI apps can read task assets."""
    target = Path(path)
    if not target.exists() or os.geteuid() != 0:
        return
    user = _lookup_user(run_as_user)
    if user is None:
        return

    targets = [target]
    for parent in target.parents:
        if parent == Path("/tmp") or parent == parent.parent:
            break
        targets.append(parent)
    if target.is_dir():
        targets.extend(sorted(target.rglob("*")))

    seen: set[Path] = set()
    for item in targets:
        if item in seen:
            continue
        seen.add(item)
        try:
            os.chown(item, user.pw_uid, user.pw_gid)
        except OSError:
            pass
        try:
            os.chmod(item, 0o755 if item.is_dir() else 0o644)
        except OSError:
            pass


def wait_for_window(
    title_pattern: str,
    *,
    window_class_pattern: str | None = None,
    display: str | None = None,
    timeout: float = 30.0,
    min_width: int = 0,
    min_height: int = 0,
) -> str:
    """Wait for a visible X11 window whose title matches the pattern."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))
    command_timeout = min(max(timeout, 5.0), 15.0)

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")

    deadline = time.time() + timeout
    while time.time() < deadline:
        search_args = [xdotool, "search", "--onlyvisible"]
        if window_class_pattern:
            search_args.extend(["--class", window_class_pattern])
        else:
            search_args.extend(["--name", title_pattern])
        result = subprocess.run(
            search_args,
            capture_output=True,
            text=True,
            env=gui_env,
            timeout=command_timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            window_ids = result.stdout.strip().splitlines()
            candidates: list[tuple[int, str]] = []
            for window_id in window_ids:
                if window_class_pattern:
                    try:
                        title = _window_title(window_id, display=display)
                    except Exception:
                        continue
                    if not re.search(title_pattern, title):
                        continue
                try:
                    _left, _top, width, height = _window_geometry(window_id, display=display)
                except Exception:
                    continue
                if width < min_width or height < min_height:
                    continue
                candidates.append((width * height, window_id))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for window matching {title_pattern!r}.")


def send_keys_to_window(
    title_pattern: str,
    keys: list[str],
    *,
    window_class_pattern: str | None = None,
    display: str | None = None,
    timeout: float = 30.0,
    min_width: int = 0,
    min_height: int = 0,
) -> str:
    """Focus a window and send key presses to it."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")

    window_id = wait_for_window(
        title_pattern,
        window_class_pattern=window_class_pattern,
        display=display,
        timeout=timeout,
        min_width=min_width,
        min_height=min_height,
    )
    subprocess.run(
        [xdotool, "windowactivate", "--sync", window_id, "key", "--window", window_id, *keys],
        check=True,
        capture_output=True,
        env=gui_env,
    )
    return window_id


def click_window_relative(
    title_pattern: str,
    x_offset: int,
    y_offset: int,
    *,
    window_class_pattern: str | None = None,
    display: str | None = None,
    timeout: float = 30.0,
    min_width: int = 0,
    min_height: int = 0,
) -> str:
    """Focus a window and click a position relative to its top-left corner."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")

    window_id = wait_for_window(
        title_pattern,
        window_class_pattern=window_class_pattern,
        display=display,
        timeout=timeout,
        min_width=min_width,
        min_height=min_height,
    )
    left, top, _width, _height = _window_geometry(window_id, display=display)
    subprocess.run(
        [xdotool, "windowactivate", "--sync", window_id],
        check=True,
        capture_output=True,
        env=gui_env,
    )
    subprocess.run(
        [xdotool, "mousemove", "--sync", str(left + x_offset), str(top + y_offset), "click", "1"],
        check=True,
        capture_output=True,
        env=gui_env,
    )
    return window_id


def type_text_to_window(
    title_pattern: str,
    text: str,
    *,
    window_class_pattern: str | None = None,
    display: str | None = None,
    timeout: float = 30.0,
    min_width: int = 0,
    min_height: int = 0,
) -> str:
    """Focus a window and type text into the active control."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")

    window_id = wait_for_window(
        title_pattern,
        window_class_pattern=window_class_pattern,
        display=display,
        timeout=timeout,
        min_width=min_width,
        min_height=min_height,
    )
    subprocess.run(
        [xdotool, "windowactivate", "--sync", window_id],
        check=True,
        capture_output=True,
        env=gui_env,
    )
    for index, line in enumerate(text.splitlines() or [""]):
        if line:
            subprocess.run(
                [xdotool, "type", "--window", window_id, "--delay", "0", line],
                check=True,
                capture_output=True,
                env=gui_env,
            )
        if index < len(text.splitlines()) - 1:
            subprocess.run(
                [xdotool, "key", "--window", window_id, "Return"],
                check=True,
                capture_output=True,
                env=gui_env,
            )
    return window_id


def activate_window(
    title_pattern: str,
    *,
    window_class_pattern: str | None = None,
    display: str | None = None,
    timeout: float = 30.0,
    min_width: int = 0,
    min_height: int = 0,
) -> str:
    """Focus a window and return its window id."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")

    window_id = wait_for_window(
        title_pattern,
        window_class_pattern=window_class_pattern,
        display=display,
        timeout=timeout,
        min_width=min_width,
        min_height=min_height,
    )
    subprocess.run(
        [xdotool, "windowactivate", "--sync", window_id],
        check=True,
        capture_output=True,
        env=gui_env,
    )
    return window_id


def active_window_id(*, display: str | None = None) -> str:
    """Return the currently focused X11 window id."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))
    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")
    result = subprocess.run(
        [xdotool, "getactivewindow"],
        check=True,
        capture_output=True,
        text=True,
        env=gui_env,
    )
    window_id = result.stdout.strip()
    if not window_id:
        raise RuntimeError("No active X11 window is available.")
    return window_id


def read_clipboard_text(*, display: str | None = None) -> str:
    """Read X11 clipboard text via xclip."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    tool = shutil.which("xclip")
    if tool is None:
        raise RuntimeError("xclip is required for GUI state synchronization but is not installed.")

    clipboard_error: subprocess.CalledProcessError | None = None
    for selection in ("clipboard", "primary"):
        try:
            result = subprocess.run(
                [tool, "-selection", selection, "-o"],
                check=True,
                capture_output=True,
                env=gui_env,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            clipboard_error = exc

    assert clipboard_error is not None
    raise clipboard_error


def _window_geometry(window_id: str, *, display: str | None = None) -> tuple[int, int, int, int]:
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xwininfo = shutil.which("xwininfo")
    if xwininfo is None:
        raise RuntimeError("xwininfo is required for GUI rendering but is not installed.")

    result = subprocess.run(
        [xwininfo, "-id", str(window_id)],
        check=True,
        capture_output=True,
        text=True,
        env=gui_env,
        timeout=15.0,
    )
    text = result.stdout
    left = int(re.search(r"Absolute upper-left X:\s+(-?\d+)", text).group(1))
    top = int(re.search(r"Absolute upper-left Y:\s+(-?\d+)", text).group(1))
    width = int(re.search(r"Width:\s+(\d+)", text).group(1))
    height = int(re.search(r"Height:\s+(\d+)", text).group(1))
    return left, top, width, height


def _window_title(window_id: str, *, display: str | None = None) -> str:
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        raise RuntimeError("xdotool is required for GUI rendering but is not installed.")

    result = subprocess.run(
        [xdotool, "getwindowname", str(window_id)],
        check=True,
        capture_output=True,
        text=True,
        env=gui_env,
        timeout=15.0,
    )
    return result.stdout.strip()


def _pointer_location(*, display: str | None = None) -> tuple[int, int] | None:
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None:
        return None

    try:
        result = subprocess.run(
            [xdotool, "getmouselocation", "--shell"],
            check=True,
            capture_output=True,
            text=True,
            env=gui_env,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    x_match = re.search(r"^X=(-?\d+)$", result.stdout, re.MULTILINE)
    y_match = re.search(r"^Y=(-?\d+)$", result.stdout, re.MULTILINE)
    if not x_match or not y_match:
        return None
    return int(x_match.group(1)), int(y_match.group(1))


def _cursor_icon() -> tuple[Image.Image, tuple[int, int]]:
    icon = Image.new("RGBA", (18, 26), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    arrow = [
        (0, 0),
        (0, 18),
        (4, 14),
        (8, 24),
        (12, 22),
        (8, 13),
        (15, 13),
    ]
    draw.polygon(arrow, fill=(0, 0, 0, 255))
    draw.line(arrow + [arrow[0]], fill=(255, 255, 255, 255), width=1)
    draw.line([(4, 14), (8, 24)], fill=(255, 255, 255, 255), width=1)
    return icon, (0, 0)


def _overlay_pointer_cursor(image: Image.Image, *, x: int, y: int) -> Image.Image:
    base = image.convert("RGBA")
    icon, hotspot = _cursor_icon()
    dest_x = x - hotspot[0]
    dest_y = y - hotspot[1]
    if dest_x >= base.width or dest_y >= base.height or dest_x + icon.width <= 0 or dest_y + icon.height <= 0:
        return image

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.alpha_composite(icon, dest=(dest_x, dest_y))
    composited = Image.alpha_composite(base, overlay)
    return composited.convert(image.mode) if image.mode != "RGBA" else composited


def _activate_window_id_for_capture(
    window_id: str,
    *,
    display: str | None = None,
    maximize: bool = False,
) -> None:
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))

    xdotool = shutil.which("xdotool")
    if xdotool is None or not window_id:
        return

    commands = [
        [xdotool, "windowactivate", "--sync", str(window_id)],
        [xdotool, "windowraise", str(window_id)],
    ]
    if maximize:
        commands.extend(
            [
                [xdotool, "windowmove", str(window_id), "0", "0"],
                [xdotool, "windowsize", str(window_id), "100%", "100%"],
            ]
        )
    for command in commands:
        try:
            subprocess.run(
                command,
                check=False,
                capture_output=True,
                env=gui_env,
                timeout=5.0,
            )
        except Exception:
            continue


def _find_capture_fallback_window(
    *,
    title_pattern: str,
    window_class_pattern: str | None,
    fallback_window_specs: Sequence[dict[str, Any]] | None,
    display: str | None,
    timeout: float,
    min_width: int,
    min_height: int,
) -> tuple[str, str, str]:
    """Return a main-window fallback for active-window captures."""

    specs = list(fallback_window_specs or ())
    if not specs:
        specs = [
            {
                "app": "",
                "title_pattern": title_pattern,
                "window_class_pattern": window_class_pattern,
                "min_width": min_width,
                "min_height": min_height,
            }
        ]

    wait_timeout = min(max(timeout, 1.0), 3.0)
    last_error = ""
    for spec in specs:
        pattern = str(spec.get("title_pattern") or title_pattern)
        class_pattern = spec.get("window_class_pattern", window_class_pattern)
        app_name = str(spec.get("app") or spec.get("name") or "")
        try:
            window_id = wait_for_window(
                pattern,
                window_class_pattern=str(class_pattern) if class_pattern else None,
                display=display,
                timeout=wait_timeout,
                min_width=int(spec.get("min_width") or min_width),
                min_height=int(spec.get("min_height") or min_height),
            )
            return window_id, app_name, ""
        except Exception as exc:
            last_error = str(exc)
            continue
    return "", "", last_error


def _active_window_matches_fallback_specs(
    window_id: str,
    *,
    fallback_window_specs: Sequence[dict[str, Any]] | None,
    display: str | None,
    prefer_first_fallback: bool = False,
) -> bool:
    specs = list(fallback_window_specs or ())
    if not specs:
        return True
    if prefer_first_fallback:
        specs = specs[:1]
    try:
        title = _window_title(window_id, display=display)
    except Exception:
        return False
    for spec in specs:
        pattern = str(spec.get("title_pattern") or "")
        if pattern and re.search(pattern, title):
            return True
    return False


def capture_window_to_png(
    output_path: str | Path,
    *,
    title_pattern: str,
    window_class_pattern: str | None = None,
    display: str | None = None,
    timeout: float = 30.0,
    margin: int = 0,
    settle_delay: float = 1.0,
    min_width: int = 0,
    min_height: int = 0,
    capture_metadata: dict[str, Any] | None = None,
    active_window: bool = False,
    fallback_window_specs: Sequence[dict[str, Any]] | None = None,
    retry_on_incomplete: bool = True,
    prefer_first_fallback: bool = False,
) -> Path:
    """Capture an X11 app window to a PNG by cropping the root display."""
    gui_env = os.environ.copy()
    gui_env.update(ensure_virtual_display(display))
    command_timeout = min(max(timeout, 5.0), 15.0)

    tool = shutil.which("import")
    if tool is None:
        raise RuntimeError("ImageMagick 'import' is required for GUI rendering but is not installed.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if capture_metadata is not None:
        capture_metadata.update(
            {
                "display": gui_env.get("DISPLAY", ""),
                "screen": gui_env.get("ASIL_XVFB_SCREEN", ""),
                "capture_complete": False,
                "active_window": bool(active_window),
                "active_window_too_small": False,
                "fallback_used": False,
                "fallback_reason": "",
                "fallback_window_id": "",
                "fallback_app": "",
                "fallback_maximize_used": False,
                "retry_capture_used": False,
                "root_capture_size": [],
                "cropped_size": [],
                "window_id": "",
                "window_geometry": {},
            }
        )

    last_geometry_error: Exception | None = None
    window_id = ""
    left = top = width = height = 0
    for _attempt in range(3):
        if active_window:
            window_id = active_window_id(display=display)
        else:
            window_id = wait_for_window(
                title_pattern,
                window_class_pattern=window_class_pattern,
                display=display,
                timeout=timeout,
                min_width=min_width,
                min_height=min_height,
            )
        if settle_delay > 0:
            time.sleep(settle_delay)
        try:
            left, top, width, height = _window_geometry(window_id, display=display)
            active_not_main = active_window and not _active_window_matches_fallback_specs(
                window_id,
                fallback_window_specs=fallback_window_specs,
                display=display,
                prefer_first_fallback=prefer_first_fallback,
            )
            if active_window and ((width < min_width or height < min_height) or active_not_main):
                fallback_reason = "active_window_not_main" if active_not_main else "active_window_too_small"
                if capture_metadata is not None:
                    capture_metadata.update(
                        {
                            "active_window_too_small": width < min_width or height < min_height,
                            "active_window_id": str(window_id),
                            "active_window_geometry": {
                                "left": left,
                                "top": top,
                                "width": width,
                                "height": height,
                            },
                            "fallback_reason": fallback_reason,
                        }
                    )
                fallback_window_id, fallback_app, fallback_error = _find_capture_fallback_window(
                    title_pattern=title_pattern,
                    window_class_pattern=window_class_pattern,
                    fallback_window_specs=fallback_window_specs,
                    display=display,
                    timeout=timeout,
                    min_width=min_width,
                    min_height=min_height,
                )
                if fallback_window_id:
                    window_id = fallback_window_id
                    _activate_window_id_for_capture(window_id, display=display, maximize=True)
                    if settle_delay > 0:
                        time.sleep(min(settle_delay, 0.5))
                    left, top, width, height = _window_geometry(window_id, display=display)
                    if capture_metadata is not None:
                        capture_metadata.update(
                            {
                                "fallback_used": True,
                                "fallback_window_id": str(window_id),
                                "fallback_app": fallback_app,
                                "fallback_maximize_used": True,
                            }
                        )
                else:
                    raise RuntimeError(
                        f"Active window is not suitable for capture ({fallback_reason}: {width}x{height}, "
                        f"expected at least {min_width}x{min_height}); no main-window fallback found. {fallback_error}"
                    )
            last_geometry_error = None
            break
        except Exception as exc:
            last_geometry_error = exc
            time.sleep(0.5)
    if last_geometry_error is not None:
        raise last_geometry_error

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        root_capture = Path(handle.name)

    def _capture_root() -> None:
        subprocess.run(
            [tool, "-display", gui_env["DISPLAY"], "-window", "root", str(root_capture)],
            check=True,
            capture_output=True,
            env=gui_env,
            timeout=command_timeout,
        )

    def _save_crop() -> bool:
        nonlocal left, top, width, height
        with Image.open(root_capture) as image:
            strict_window_visible = left >= 0 and top >= 0 and (left + width) <= image.width and (top + height) <= image.height
            edge_tolerance = 24
            x0 = max(left - margin, 0)
            y0 = max(top - margin, 0)
            x1 = min(left + width + margin, image.width)
            y1 = min(top + height + margin, image.height)
            cropped = image.crop((x0, y0, x1, y1))
            crop_meets_expectation = cropped.width >= min_width and cropped.height >= min_height
            tolerated_window_visible = (
                min_width > 0
                and min_height > 0
                and left >= -edge_tolerance
                and top >= -edge_tolerance
                and (left + width) <= image.width + edge_tolerance
                and (top + height) <= image.height + edge_tolerance
                and crop_meets_expectation
            )
            window_visible = strict_window_visible or tolerated_window_visible
            capture_complete = window_visible and crop_meets_expectation
            pointer = _pointer_location(display=display)
            if pointer is not None:
                pointer_x, pointer_y = pointer
                if x0 <= pointer_x < x1 and y0 <= pointer_y < y1:
                    cropped = _overlay_pointer_cursor(cropped, x=pointer_x - x0, y=pointer_y - y0)
            cropped.save(out)
            if capture_metadata is not None:
                capture_metadata.update(
                    {
                        "root_capture_size": [image.width, image.height],
                        "cropped_size": [cropped.width, cropped.height],
                        "capture_complete": capture_complete,
                        "window_visible": window_visible,
                        "window_visible_strict": strict_window_visible,
                        "window_visible_tolerance_px": edge_tolerance,
                        "crop_meets_expectation": crop_meets_expectation,
                        "window_id": str(window_id),
                        "window_geometry": {
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height,
                        },
                    }
                )
            return capture_complete

    try:
        _capture_root()
        window_visible = _save_crop()
        if retry_on_incomplete and not window_visible:
            if capture_metadata is not None:
                capture_metadata["retry_capture_used"] = True
                capture_metadata["fallback_reason"] = capture_metadata.get("fallback_reason") or "incomplete_window_crop"
            _activate_window_id_for_capture(window_id, display=display, maximize=True)
            time.sleep(0.35)
            left, top, width, height = _window_geometry(window_id, display=display)
            _capture_root()
            _save_crop()
    finally:
        root_capture.unlink(missing_ok=True)

    assert_png_not_blank(out)
    return out


def terminate_process(proc: subprocess.Popen[str], *, timeout: float = 10.0) -> None:
    """Stop a GUI process without leaking background apps."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=timeout)
