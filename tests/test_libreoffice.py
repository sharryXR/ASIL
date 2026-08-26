from pathlib import Path

from asil.adapters.libreoffice import LibreOfficeAdapter
from asil.protocol import Action


def test_observe_extracts_cells(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    obs = adapter.observe()

    assert obs.meta.app_name == "LibreOffice Calc"
    assert obs.meta.observation_source == "file_parse"
    # sample_ods has: Sheet1 with 4 cells (Revenue, 100, Cost, 60)
    assert len(obs.interactive_elements) >= 4


def test_observe_cell_values(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    obs = adapter.observe()

    cells = {e.id: e for e in obs.interactive_elements}
    assert "Sheet1!A1" in cells
    assert cells["Sheet1!A1"].value == "Revenue"
    assert "Sheet1!B1" in cells
    assert cells["Sheet1!B1"].value == "100"


def test_observe_shows_sheets(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    obs = adapter.observe()
    assert "Sheet1" in obs.data_summary


def test_execute_set_cell_value(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    action = Action(
        action_type="modify_file",
        target=str(sample_ods),
        params={
            "operations": [
                {"sheet": "Sheet1", "cell": "A1", "value": "Income", "value_type": "string"}
            ]
        },
    )
    obs = adapter.execute(action)
    cells = {e.id: e for e in obs.interactive_elements}
    assert cells["Sheet1!A1"].value == "Income"


def test_execute_set_numeric_value(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    action = Action(
        action_type="modify_file",
        target=str(sample_ods),
        params={
            "operations": [
                {"sheet": "Sheet1", "cell": "B1", "value": "250", "value_type": "float"}
            ]
        },
    )
    obs = adapter.execute(action)
    cells = {e.id: e for e in obs.interactive_elements}
    assert cells["Sheet1!B1"].value == "250"


def test_validate_action(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    good = Action(action_type="modify_file", target="x.ods", params={})
    bad = Action(action_type="api_call", target="obs", params={})
    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad)


def test_export_to_pdf_raises_without_libreoffice(sample_ods: Path, monkeypatch):
    """export_to_pdf should raise RuntimeError when LibreOffice is not installed."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    try:
        adapter.export_to_pdf()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "LibreOffice is not installed" in str(e)


def test_export_to_pdf_calls_libreoffice(sample_ods: Path, monkeypatch, tmp_path):
    """export_to_pdf should call LibreOffice CLI with correct args."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/libreoffice")
    called = {}

    def mock_run(cmd, **kwargs):
        called["cmd"] = cmd
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.run", mock_run)
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    out = adapter.export_to_pdf()
    assert out.suffix == ".pdf"
    assert called["cmd"][0] == "/usr/bin/libreoffice"
    assert "--headless" in called["cmd"]
    assert "--convert-to" in called["cmd"]


def test_describe_rendering_reports_real_window_capture(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "calc window" in artifact.description.lower()


def test_gui_session_spec_exposes_explicit_startup_and_readiness_probes(sample_ods: Path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "desktop"
    assert spec.backend_ready_probe is None
    assert spec.ui_ready_probe is not None
    assert spec.extra_env["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_render_to_png_uses_real_calc_window_capture(sample_ods: Path, monkeypatch, tmp_path):
    adapter = LibreOfficeAdapter(ods_path=sample_ods)
    calls = {}
    lock_path = sample_ods.parent / f".~lock.{sample_ods.name}#"
    lock_path.write_text("stale-lock", encoding="utf-8")

    def fake_launch(command, **kwargs):
        calls["command"] = list(command)
        calls["launch_kwargs"] = dict(kwargs)
        return object()

    def fake_capture(output_path, **kwargs):
        calls["capture_output_path"] = Path(output_path)
        calls["capture_kwargs"] = dict(kwargs)
        kwargs["capture_metadata"]["capture_complete"] = False
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    def fake_terminate(proc, **kwargs):
        calls["terminated"] = proc
        calls["terminate_kwargs"] = dict(kwargs)

    def fake_send_keys(title_pattern, keys, **kwargs):
        calls.setdefault("keys", []).append((title_pattern, list(keys), dict(kwargs)))
        return "window-id"

    monkeypatch.setattr("asil.adapters.libreoffice.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("asil.adapters.libreoffice.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.libreoffice.capture_window_to_png", fake_capture)
    monkeypatch.setattr("asil.adapters.libreoffice.send_keys_to_window", fake_send_keys)
    monkeypatch.setattr(
        "asil.adapters.libreoffice.ensure_user_access",
        lambda path, **kwargs: calls.setdefault("access", (Path(path), kwargs)),
    )
    monkeypatch.setattr("asil.adapters.libreoffice.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "calc.png")

    assert out == tmp_path / "calc.png"
    assert calls["command"][0] == "/usr/bin/libreoffice"
    assert calls["command"][1] == "--calc"
    assert calls["command"][-1] == str(sample_ods)
    assert calls["launch_kwargs"]["run_as_user"] == "asilgui"
    assert calls["capture_output_path"] == tmp_path / "calc.png"
    assert calls["capture_kwargs"]["title_pattern"] == r".*LibreOffice Calc|.* - LibreOffice Calc"
    assert calls["capture_kwargs"]["timeout"] == 60.0
    assert calls["capture_kwargs"]["min_width"] == 900
    assert calls["capture_kwargs"]["min_height"] == 700
    assert calls["keys"] == [("Tip of the Day", ["Escape"], {"timeout": 5.0})]
    assert adapter._last_capture_complete is False
    assert not lock_path.exists()
    assert calls["terminated"] is not None
