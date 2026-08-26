from pathlib import Path

from asil.adapters.libreoffice_writer import LibreOfficeWriterAdapter
from asil.protocol import Action


def _make_sample_odt(tmp_path: Path) -> Path:
    adapter = LibreOfficeWriterAdapter.from_evaluation_context(tmp_path)
    return adapter.source_path


def test_observe_extracts_headings_and_paragraphs(tmp_path: Path):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    obs = adapter.observe()

    assert obs.meta.app_name == "LibreOffice Writer"
    assert obs.meta.observation_source == "file_parse"
    elements = {e.id: e for e in obs.interactive_elements}
    assert "heading:1" in elements
    assert "paragraph:1" in elements
    assert elements["heading:1"].value["text_content"] == "Project Brief"


def test_execute_set_paragraph_text(tmp_path: Path):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    action = Action(
        action_type="modify_file",
        target=str(odt),
        params={"operations": [{"action": "set_paragraph_text", "index": 1, "text": "Updated planning note."}]},
    )
    obs = adapter.execute(action)

    elements = {e.id: e for e in obs.interactive_elements}
    assert elements["paragraph:1"].value["text_content"] == "Updated planning note."


def test_execute_add_heading_and_paragraph(tmp_path: Path):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    action = Action(
        action_type="modify_file",
        target=str(odt),
        params={
            "operations": [
                {"action": "add_heading", "text": "Timeline", "level": 2},
                {"action": "add_paragraph", "text": "Draft the next milestone list."},
            ]
        },
    )
    obs = adapter.execute(action)
    labels = [e.label for e in obs.interactive_elements]

    assert "Timeline" in labels
    assert "Draft the next milestone list." in labels


def test_validate_action(tmp_path: Path):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    assert adapter.validate_action(Action(action_type="modify_file", target=str(odt), params={}))
    assert not adapter.validate_action(Action(action_type="api_call", target="writer", params={}))


def test_export_to_pdf_raises_without_libreoffice(tmp_path: Path, monkeypatch):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    monkeypatch.setattr("shutil.which", lambda _: None)
    try:
        adapter.export_to_pdf()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as exc:
        assert "LibreOffice is not installed" in str(exc)


def test_export_to_pdf_raises_when_conversion_produces_no_pdf(tmp_path: Path, monkeypatch):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/libreoffice")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    try:
        adapter.export_to_pdf(tmp_path / "writer.pdf")
        assert False, "Should have raised RuntimeError when PDF output is missing"
    except RuntimeError as exc:
        assert "did not produce a PDF" in str(exc)


def test_describe_rendering_reports_real_window_capture(tmp_path: Path):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "writer window" in artifact.description.lower()


def test_gui_session_spec_exposes_startup_and_readiness_probes(tmp_path: Path):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "desktop"
    assert spec.ui_ready_probe is not None
    assert spec.extra_env["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_render_to_png_uses_real_writer_window_capture(tmp_path: Path, monkeypatch):
    odt = _make_sample_odt(tmp_path)
    adapter = LibreOfficeWriterAdapter(odt)
    calls = {}
    lock_path = odt.parent / f".~lock.{odt.name}#"
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

    monkeypatch.setattr("asil.adapters.libreoffice_writer.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("asil.adapters.libreoffice_writer.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.libreoffice_writer.capture_window_to_png", fake_capture)
    monkeypatch.setattr("asil.adapters.libreoffice_writer.send_keys_to_window", fake_send_keys)
    monkeypatch.setattr("asil.adapters.libreoffice_writer.ensure_user_access", lambda path, **kwargs: calls.setdefault("access", (Path(path), kwargs)))
    monkeypatch.setattr("asil.adapters.libreoffice_writer.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "writer.png")

    assert out == tmp_path / "writer.png"
    assert calls["command"][0] == "/usr/bin/libreoffice"
    assert calls["command"][1] == "--writer"
    assert calls["command"][-1] == str(odt)
    assert calls["launch_kwargs"]["run_as_user"] == "asilgui"
    assert calls["capture_output_path"] == tmp_path / "writer.png"
    assert calls["capture_kwargs"]["title_pattern"] == r".*LibreOffice Writer|.* - LibreOffice Writer"
    assert calls["capture_kwargs"]["timeout"] == 60.0
    assert calls["capture_kwargs"]["min_width"] == 900
    assert calls["capture_kwargs"]["min_height"] == 700
    assert calls["keys"] == [("Tip of the Day", ["Escape"], {"timeout": 5.0})]
    assert adapter._last_capture_complete is False
    assert not lock_path.exists()
    assert calls["terminated"] is not None
