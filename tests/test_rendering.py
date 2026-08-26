from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw


def test_assert_png_not_blank_rejects_single_color_image(tmp_path: Path):
    from asil.rendering import assert_png_not_blank

    image_path = tmp_path / "blank.png"
    Image.new("RGBA", (240, 160), "white").save(image_path)

    try:
        assert_png_not_blank(image_path)
    except RuntimeError as exc:
        assert "blank" in str(exc).lower()
    else:
        raise AssertionError("Expected blank image validation to fail")


def test_assert_png_not_blank_accepts_image_with_visible_content(tmp_path: Path):
    from asil.rendering import assert_png_not_blank

    image_path = tmp_path / "content.png"
    image = Image.new("RGBA", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 200, 120), fill="black")
    image.save(image_path)

    assert_png_not_blank(image_path)


def test_capture_window_to_png_reports_incomplete_when_window_extends_offscreen(tmp_path: Path, monkeypatch):
    from asil.rendering import capture_window_to_png

    output = tmp_path / "window.png"
    capture_metadata: dict[str, bool] = {}

    monkeypatch.setattr("asil.rendering.ensure_virtual_display", lambda display=None: {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime"})
    monkeypatch.setattr("asil.rendering.wait_for_window", lambda *args, **kwargs: "0x123")
    monkeypatch.setattr("asil.rendering._window_geometry", lambda *args, **kwargs: (-20, 10, 640, 360))
    monkeypatch.setattr("asil.rendering._pointer_location", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.rendering.assert_png_not_blank", lambda path: None)
    monkeypatch.setattr("asil.rendering.shutil.which", lambda name: "/usr/bin/import" if name == "import" else f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/import":
            Image.new("RGB", (800, 600), "white").save(Path(cmd[-1]))
            return None
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)

    result = capture_window_to_png(
        output,
        title_pattern="Test",
        capture_metadata=capture_metadata,
        retry_on_incomplete=False,
    )

    assert result == output
    assert capture_metadata["capture_complete"] is False
    assert capture_metadata["root_capture_size"] == [800, 600]
    assert capture_metadata["cropped_size"] == [620, 360]


def test_capture_window_to_png_falls_back_when_active_window_is_too_small(tmp_path: Path, monkeypatch):
    from asil.rendering import capture_window_to_png

    output = tmp_path / "window.png"
    capture_metadata: dict[str, object] = {}

    monkeypatch.setattr(
        "asil.rendering.ensure_virtual_display",
        lambda display=None: {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime", "ASIL_XVFB_SCREEN": "1920x1080x24"},
    )
    monkeypatch.setattr("asil.rendering.active_window_id", lambda *args, **kwargs: "0xtiny")
    monkeypatch.setattr("asil.rendering.wait_for_window", lambda *args, **kwargs: "0xmain")
    monkeypatch.setattr(
        "asil.rendering._window_geometry",
        lambda window_id, **kwargs: (10, 10, 180, 120) if window_id == "0xtiny" else (20, 30, 1000, 700),
    )
    monkeypatch.setattr("asil.rendering._pointer_location", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.rendering.assert_png_not_blank", lambda path: None)
    monkeypatch.setattr("asil.rendering.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/import":
            Image.new("RGB", (1920, 1080), "white").save(Path(cmd[-1]))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)

    capture_window_to_png(
        output,
        title_pattern="Main",
        active_window=True,
        min_width=800,
        min_height=600,
        capture_metadata=capture_metadata,
        fallback_window_specs=[
            {
                "app": "audacity",
                "title_pattern": "Audacity",
                "window_class_pattern": "audacity",
                "min_width": 800,
                "min_height": 600,
            }
        ],
    )

    with Image.open(output) as image:
        assert image.size == (1000, 700)
    assert capture_metadata["active_window_too_small"] is True
    assert capture_metadata["fallback_used"] is True
    assert capture_metadata["fallback_maximize_used"] is True
    assert capture_metadata["fallback_window_id"] == "0xmain"
    assert capture_metadata["fallback_app"] == "audacity"
    assert capture_metadata["window_id"] == "0xmain"
    assert capture_metadata["capture_complete"] is True


def test_capture_window_to_png_retries_incomplete_crop_after_raising_window(tmp_path: Path, monkeypatch):
    from asil.rendering import capture_window_to_png

    output = tmp_path / "window.png"
    capture_metadata: dict[str, object] = {}
    geometries = [(-40, 10, 640, 360), (0, 0, 800, 600)]
    import_calls = 0

    monkeypatch.setattr(
        "asil.rendering.ensure_virtual_display",
        lambda display=None: {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime", "ASIL_XVFB_SCREEN": "1920x1080x24"},
    )
    monkeypatch.setattr("asil.rendering.wait_for_window", lambda *args, **kwargs: "0x123")
    monkeypatch.setattr("asil.rendering._window_geometry", lambda *args, **kwargs: geometries.pop(0) if geometries else (0, 0, 800, 600))
    monkeypatch.setattr("asil.rendering._pointer_location", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.rendering.assert_png_not_blank", lambda path: None)
    monkeypatch.setattr("asil.rendering.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        nonlocal import_calls
        if cmd[0] == "/usr/bin/import":
            import_calls += 1
            Image.new("RGB", (800, 600), "white").save(Path(cmd[-1]))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)

    capture_window_to_png(output, title_pattern="Test", capture_metadata=capture_metadata)

    assert import_calls == 2
    assert capture_metadata["retry_capture_used"] is True
    assert capture_metadata["capture_complete"] is True
    assert capture_metadata["cropped_size"] == [800, 600]


def test_capture_window_to_png_accepts_small_window_manager_edge_overflow(tmp_path: Path, monkeypatch):
    from asil.rendering import capture_window_to_png

    output = tmp_path / "window.png"
    capture_metadata: dict[str, object] = {}

    monkeypatch.setattr(
        "asil.rendering.ensure_virtual_display",
        lambda display=None: {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime", "ASIL_XVFB_SCREEN": "1920x1080x24"},
    )
    monkeypatch.setattr("asil.rendering.wait_for_window", lambda *args, **kwargs: "0x123")
    monkeypatch.setattr("asil.rendering._window_geometry", lambda *args, **kwargs: (1, 20, 1920, 1080))
    monkeypatch.setattr("asil.rendering._pointer_location", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.rendering.assert_png_not_blank", lambda path: None)
    monkeypatch.setattr("asil.rendering.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/import":
            Image.new("RGB", (1920, 1080), "white").save(Path(cmd[-1]))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)

    capture_window_to_png(
        output,
        title_pattern="Test",
        capture_metadata=capture_metadata,
        min_width=1200,
        min_height=700,
        retry_on_incomplete=False,
    )

    assert capture_metadata["capture_complete"] is True
    assert capture_metadata["window_visible"] is True
    assert capture_metadata["window_visible_strict"] is False
    assert capture_metadata["cropped_size"] == [1919, 1060]


def test_capture_window_to_png_overlays_pointer_when_inside_window(tmp_path: Path, monkeypatch):
    from asil.rendering import capture_window_to_png

    output = tmp_path / "window.png"

    monkeypatch.setattr(
        "asil.rendering.ensure_virtual_display",
        lambda display=None: {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime"},
    )
    monkeypatch.setattr("asil.rendering.wait_for_window", lambda *args, **kwargs: "0x123")
    monkeypatch.setattr("asil.rendering._window_geometry", lambda *args, **kwargs: (100, 50, 240, 160))
    monkeypatch.setattr("asil.rendering.assert_png_not_blank", lambda path: None)
    monkeypatch.setattr(
        "asil.rendering.shutil.which",
        lambda name: "/usr/bin/import" if name == "import" else "/usr/bin/xdotool" if name == "xdotool" else f"/usr/bin/{name}",
    )

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/import":
            Image.new("RGB", (800, 600), "white").save(Path(cmd[-1]))
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[:2] == ["/usr/bin/xdotool", "getmouselocation"]:
            return SimpleNamespace(stdout="X=150\nY=90\nSCREEN=0\nWINDOW=0x123\n", stderr="", returncode=0)
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)

    capture_window_to_png(output, title_pattern="Test")

    with Image.open(output) as image:
        cursor_region = image.crop((45, 35, 75, 75))
        diff = ImageChops.difference(cursor_region.convert("RGB"), Image.new("RGB", cursor_region.size, "white"))
        assert diff.getbbox() is not None


def test_capture_window_to_png_omits_pointer_when_outside_window(tmp_path: Path, monkeypatch):
    from asil.rendering import capture_window_to_png

    output = tmp_path / "window.png"

    monkeypatch.setattr(
        "asil.rendering.ensure_virtual_display",
        lambda display=None: {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime"},
    )
    monkeypatch.setattr("asil.rendering.wait_for_window", lambda *args, **kwargs: "0x123")
    monkeypatch.setattr("asil.rendering._window_geometry", lambda *args, **kwargs: (100, 50, 240, 160))
    monkeypatch.setattr("asil.rendering.assert_png_not_blank", lambda path: None)
    monkeypatch.setattr(
        "asil.rendering.shutil.which",
        lambda name: "/usr/bin/import" if name == "import" else "/usr/bin/xdotool" if name == "xdotool" else f"/usr/bin/{name}",
    )

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/import":
            Image.new("RGB", (800, 600), "white").save(Path(cmd[-1]))
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[:2] == ["/usr/bin/xdotool", "getmouselocation"]:
            return SimpleNamespace(stdout="X=20\nY=20\nSCREEN=0\nWINDOW=0x123\n", stderr="", returncode=0)
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)

    capture_window_to_png(output, title_pattern="Test")

    with Image.open(output) as image:
        diff = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white"))
        assert diff.getbbox() is None


def test_ensure_virtual_display_defaults_to_1080p_and_supports_env_override(tmp_path: Path, monkeypatch):
    import inspect

    from asil import rendering
    from asil.rendering import ensure_virtual_display

    commands: list[list[str]] = []
    run_calls = 0

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

    def fake_run(cmd, **kwargs):
        nonlocal run_calls
        commands.append(cmd)
        if cmd[0] == "/usr/bin/xdpyinfo":
            run_calls += 1
            if run_calls == 1:
                raise subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=0)

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return FakeProcess()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("ASIL_XVFB_SCREEN", "1366x768x24")
    monkeypatch.setattr(rendering, "_XVFB_PID_FILE", tmp_path / "xvfb.pid")
    monkeypatch.setattr(rendering, "_OPENBOX_PID_FILE", tmp_path / "openbox.pid")
    monkeypatch.setattr("asil.rendering.shutil.which", lambda name: f"/usr/bin/{name}" if name in {"xdpyinfo", "Xvfb"} else None)
    monkeypatch.setattr("asil.rendering.subprocess.run", fake_run)
    monkeypatch.setattr("asil.rendering.subprocess.Popen", fake_popen)
    monkeypatch.setattr("asil.rendering.time.sleep", lambda *_args, **_kwargs: None)

    env = ensure_virtual_display()

    assert inspect.signature(ensure_virtual_display).parameters["screen"].default == "1920x1080x24"
    assert env["ASIL_XVFB_SCREEN"] == "1366x768x24"
    assert ["/usr/bin/Xvfb", ":99", "-screen", "0", "1366x768x24", "-ac", "-nolisten", "tcp"] in commands


@patch("asil.rendering.subprocess.Popen")
@patch("asil.rendering.ensure_virtual_display", return_value={"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime"})
@patch("asil.rendering.shutil.which", return_value="/usr/sbin/runuser")
@patch("asil.rendering.pwd.getpwnam")
@patch("asil.rendering.os.geteuid", return_value=0)
def test_launch_gui_process_runs_as_named_user_when_root(
    mock_geteuid,
    mock_getpwnam,
    mock_which,
    mock_display,
    mock_popen,
):
    from asil.rendering import launch_gui_process

    mock_getpwnam.return_value = SimpleNamespace(pw_dir="/home/asilgui", pw_uid=1234, pw_gid=1234)

    launch_gui_process(["vlc", "preview.wav"], run_as_user="asilgui")

    mock_popen.assert_called_once()
    command = mock_popen.call_args.args[0]
    assert command == ["/usr/sbin/runuser", "--preserve-environment", "-u", "asilgui", "--", "vlc", "preview.wav"]


@patch("asil.rendering.subprocess.Popen")
@patch("asil.rendering.ensure_virtual_display", return_value={"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime"})
@patch("asil.rendering.shutil.which", return_value="/usr/sbin/runuser")
@patch("asil.rendering.pwd.getpwnam")
@patch("asil.rendering.os.geteuid", return_value=0)
def test_launch_gui_process_preserves_extra_environment_when_switching_user(
    mock_geteuid,
    mock_getpwnam,
    mock_which,
    mock_display,
    mock_popen,
):
    from asil.rendering import launch_gui_process

    mock_getpwnam.return_value = SimpleNamespace(pw_dir="/home/asilgui", pw_uid=1234, pw_gid=1234)

    launch_gui_process(
        ["audacity"],
        run_as_user="asilgui",
        extra_env={"HOME": "/tmp/gui-home", "XDG_CONFIG_HOME": "/tmp/gui-config"},
    )

    mock_popen.assert_called_once()
    command = mock_popen.call_args.args[0]
    assert command == [
        "/usr/sbin/runuser",
        "--preserve-environment",
        "-u",
        "asilgui",
        "--",
        "audacity",
    ]


@patch("asil.rendering.subprocess.Popen")
@patch("asil.rendering.ensure_virtual_display", return_value={"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime"})
@patch("asil.rendering.shutil.which", return_value="/usr/sbin/runuser")
@patch("asil.rendering.pwd.getpwnam")
@patch("asil.rendering.os.geteuid", return_value=0)
def test_launch_gui_process_sets_home_to_target_user_when_not_overridden(
    mock_geteuid,
    mock_getpwnam,
    mock_which,
    mock_display,
    mock_popen,
):
    from asil.rendering import launch_gui_process

    mock_getpwnam.return_value = SimpleNamespace(pw_dir="/home/asilgui", pw_uid=1234, pw_gid=1234)

    launch_gui_process(["audacity"], run_as_user="asilgui")

    mock_popen.assert_called_once()
    env = mock_popen.call_args.kwargs["env"]
    assert env["HOME"] == "/home/asilgui"


@patch("asil.rendering.subprocess.run")
@patch("asil.rendering.time.sleep")
@patch("asil.rendering.shutil.which", side_effect=lambda tool: "/usr/bin/xdpyinfo" if tool == "xdpyinfo" else None)
@patch("asil.rendering.os.chown")
@patch("asil.rendering.os.geteuid", return_value=0)
@patch("asil.rendering.pwd.getpwnam", return_value=SimpleNamespace(pw_uid=1234, pw_gid=1234))
def test_ensure_virtual_display_uses_user_owned_runtime_dir_when_requested(
    mock_getpwnam,
    mock_geteuid,
    mock_chown,
    mock_which,
    mock_sleep,
    mock_run,
):
    from asil.rendering import ensure_virtual_display

    mock_run.return_value = SimpleNamespace(returncode=0)

    env = ensure_virtual_display(run_as_user="asilgui")

    assert env["XDG_RUNTIME_DIR"].endswith("asilgui")
    mock_chown.assert_called_once()


@patch("asil.rendering.ensure_virtual_display", return_value={"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime-asilgui"})
@patch("asil.rendering.subprocess.run")
@patch("asil.rendering.time.sleep")
@patch("asil.rendering.shutil.which")
@patch("asil.rendering.os.geteuid", return_value=0)
@patch("asil.rendering.pwd.getpwnam", return_value=SimpleNamespace(pw_uid=1234, pw_gid=1234, pw_dir="/home/asilgui"))
def test_ensure_audio_backend_sets_default_sink_and_source_for_target_user(
    mock_getpwnam,
    mock_geteuid,
    mock_which,
    mock_sleep,
    mock_run,
    mock_display,
):
    from asil.rendering import ensure_audio_backend

    def which(tool: str) -> str | None:
        mapping = {
            "pulseaudio": "/usr/bin/pulseaudio",
            "pactl": "/usr/bin/pactl",
            "runuser": "/usr/sbin/runuser",
        }
        return mapping.get(tool)

    mock_which.side_effect = which
    mock_run.side_effect = [
        SimpleNamespace(returncode=1, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="0\talsa_output\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]

    ensure_audio_backend(run_as_user="asilgui")

    commands = [call.args[0] for call in mock_run.call_args_list]
    envs = [call.kwargs["env"] for call in mock_run.call_args_list]
    assert all(env["HOME"] == "/home/asilgui" for env in envs)
    assert all(env["USER"] == "asilgui" for env in envs)
    assert all(env["LOGNAME"] == "asilgui" for env in envs)
    assert [
        "/usr/sbin/runuser",
        "--preserve-environment",
        "-u",
        "asilgui",
        "--",
        "/usr/bin/pactl",
        "set-default-sink",
        "asil-null",
    ] in commands
    assert [
        "/usr/sbin/runuser",
        "--preserve-environment",
        "-u",
        "asilgui",
        "--",
        "/usr/bin/pactl",
        "set-default-source",
        "asil-null.monitor",
    ] in commands


@patch("asil.rendering.ensure_virtual_display", return_value={"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/tmp/runtime-asilgui"})
@patch("asil.rendering.subprocess.run")
@patch("asil.rendering.shutil.which")
@patch("asil.rendering.os.geteuid", return_value=0)
@patch("asil.rendering.pwd.getpwnam", return_value=SimpleNamespace(pw_uid=1234, pw_gid=1234, pw_dir="/home/asilgui"))
def test_ensure_audio_backend_sets_default_sink_even_when_null_sink_exists(
    mock_getpwnam,
    mock_geteuid,
    mock_which,
    mock_run,
    mock_display,
):
    from asil.rendering import ensure_audio_backend

    def which(tool: str) -> str | None:
        mapping = {
            "pulseaudio": "/usr/bin/pulseaudio",
            "pactl": "/usr/bin/pactl",
            "runuser": "/usr/sbin/runuser",
        }
        return mapping.get(tool)

    mock_which.side_effect = which
    mock_run.side_effect = [
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="1\tasil-null\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]

    ensure_audio_backend(run_as_user="asilgui")

    commands = [call.args[0] for call in mock_run.call_args_list]
    assert [
        "/usr/sbin/runuser",
        "--preserve-environment",
        "-u",
        "asilgui",
        "--",
        "/usr/bin/pactl",
        "set-default-sink",
        "asil-null",
    ] in commands
    assert [
        "/usr/sbin/runuser",
        "--preserve-environment",
        "-u",
        "asilgui",
        "--",
        "/usr/bin/pactl",
        "set-default-source",
        "asil-null.monitor",
    ] in commands


@patch("asil.rendering.os.geteuid", return_value=0)
@patch("asil.rendering.os.chmod")
@patch("asil.rendering.os.chown")
@patch("asil.rendering.pwd.getpwnam", return_value=SimpleNamespace(pw_uid=1234, pw_gid=4321))
def test_ensure_user_access_makes_tree_readable_by_gui_user(
    mock_getpwnam,
    mock_chown,
    mock_chmod,
    mock_geteuid,
    tmp_path: Path,
):
    from asil.rendering import ensure_user_access

    root = tmp_path / "project"
    nested = root / "nested"
    nested.mkdir(parents=True)
    file_path = nested / "asset.wav"
    file_path.write_text("data", encoding="utf-8")
    parent_dir = root.parent

    ensure_user_access(root, run_as_user="asilgui")

    chowned_paths = [Path(call.args[0]) for call in mock_chown.call_args_list]
    chmodded_paths = [Path(call.args[0]) for call in mock_chmod.call_args_list]
    assert parent_dir in chowned_paths
    assert root in chowned_paths
    assert nested in chowned_paths
    assert file_path in chowned_paths
    assert Path("/tmp") not in chowned_paths
    assert parent_dir in chmodded_paths
    assert root in chmodded_paths
    assert nested in chmodded_paths
    assert file_path in chmodded_paths
    assert Path("/tmp") not in chmodded_paths
