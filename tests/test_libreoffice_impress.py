from pathlib import Path

from asil.adapters.libreoffice_impress import LibreOfficeImpressAdapter
from asil.rendering import rasterize_pdf_pages
from asil.protocol import Action


def _make_sample_odp(tmp_path: Path) -> Path:
    adapter = LibreOfficeImpressAdapter.from_evaluation_context(tmp_path)
    return adapter.source_path


def test_observe_extracts_slide_titles_and_bodies(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    obs = adapter.observe()

    assert obs.meta.app_name == "LibreOffice Impress"
    elements = {e.id: e for e in obs.interactive_elements}
    assert elements["slide:1:title"].value["text_content"] == "Roadmap Overview"
    assert elements["slide:2:body:1"].value["text_content"] == "Confirm owners for launch tasks."


def test_execute_set_slide_title_and_body(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    action = Action(
        action_type="modify_file",
        target=str(odp),
        params={
            "operations": [
                {"action": "set_slide_title", "slide_index": 1, "text": "Launch Readiness"},
                {"action": "set_slide_body", "slide_index": 1, "body_index": 1, "text": "Finalize training materials."},
            ]
        },
    )
    obs = adapter.execute(action)
    elements = {e.id: e for e in obs.interactive_elements}

    assert elements["slide:1:title"].value["text_content"] == "Launch Readiness"
    assert elements["slide:1:body:1"].value["text_content"] == "Finalize training materials."


def test_execute_add_slide_and_append_body(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    action = Action(
        action_type="modify_file",
        target=str(odp),
        params={
            "operations": [
                {"action": "append_slide_body", "slide_index": 2, "text": "Share the checklist with stakeholders."},
                {
                    "action": "add_slide",
                    "title": "Executive Recap",
                    "body": ["Revenue plan approved.", "Hiring review pending."],
                },
            ]
        },
    )
    obs = adapter.execute(action)
    elements = {e.id: e for e in obs.interactive_elements}

    assert elements["slide:2:body:2"].value["text_content"] == "Share the checklist with stakeholders."
    assert elements["slide:3:title"].value["text_content"] == "Executive Recap"
    assert elements["slide:3:body:2"].value["text_content"] == "Hiring review pending."


def test_execute_accepts_common_text_aliases(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    action = Action(
        action_type="modify_file",
        target=str(odp),
        params={
            "operations": [
                {"action": "set_slide_title", "slide_index": 1, "title": "Quarterly Plan"},
                {"action": "set_slide_body", "slide_index": 1, "body_index": 1, "value": "Confirm team assignments this week."},
                {"action": "append_slide_body", "slide_index": 1, "body": ["Share the checklist with stakeholders."]},
            ]
        },
    )

    obs = adapter.execute(action)
    elements = {e.id: e for e in obs.interactive_elements}

    assert elements["slide:1:title"].value["text_content"] == "Quarterly Plan"
    assert elements["slide:1:body:1"].value["text_content"] == "Confirm team assignments this week."
    assert elements["slide:1:body:2"].value["text_content"] == "Share the checklist with stakeholders."


def test_validate_action(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    assert adapter.validate_action(Action(action_type="modify_file", target=str(odp), params={}))
    assert not adapter.validate_action(Action(action_type="api_call", target="impress", params={}))


def test_export_to_pdf_raises_without_libreoffice(tmp_path: Path, monkeypatch):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    monkeypatch.setattr("shutil.which", lambda _: None)
    try:
        adapter.export_to_pdf()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as exc:
        assert "LibreOffice is not installed" in str(exc)


def test_export_to_pdf_raises_when_conversion_produces_no_pdf(tmp_path: Path, monkeypatch):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/libreoffice")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    try:
        adapter.export_to_pdf(tmp_path / "slides.pdf")
        assert False, "Should have raised RuntimeError when PDF output is missing"
    except RuntimeError as exc:
        assert "did not produce a PDF" in str(exc)


def test_describe_rendering_reports_real_window_capture(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "impress window" in artifact.description.lower()


def test_gui_session_spec_exposes_startup_and_readiness_probes(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "desktop"
    assert spec.ui_ready_probe is not None
    assert spec.extra_env["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_render_to_png_uses_real_impress_window_capture(tmp_path: Path, monkeypatch):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)
    calls = {}
    lock_path = odp.parent / f".~lock.{odp.name}#"
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

    def fake_send_keys(title_pattern, keys, **kwargs):
        calls.setdefault("keys", []).append((title_pattern, list(keys), dict(kwargs)))
        return "window-id"

    def fake_terminate(proc, **kwargs):
        calls["terminated"] = proc
        calls["terminate_kwargs"] = dict(kwargs)

    monkeypatch.setattr("asil.adapters.libreoffice_impress.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("asil.adapters.libreoffice_impress.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.libreoffice_impress.capture_window_to_png", fake_capture)
    monkeypatch.setattr("asil.adapters.libreoffice_impress.send_keys_to_window", fake_send_keys)
    monkeypatch.setattr("asil.adapters.libreoffice_impress.ensure_user_access", lambda path, **kwargs: calls.setdefault("access", (Path(path), kwargs)))
    monkeypatch.setattr("asil.adapters.libreoffice_impress.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "slides.png")

    assert out == tmp_path / "slides.png"
    assert calls["command"][0] == "/usr/bin/libreoffice"
    assert calls["command"][1] == "--impress"
    assert calls["command"][-1] == str(odp)
    assert calls["launch_kwargs"]["run_as_user"] == "asilgui"
    assert calls["capture_output_path"] == tmp_path / "slides.png"
    assert calls["capture_kwargs"]["title_pattern"] == r".*LibreOffice Impress|.* - LibreOffice Impress"
    assert calls["capture_kwargs"]["timeout"] == 60.0
    assert calls["capture_kwargs"]["min_width"] == 1000
    assert calls["capture_kwargs"]["min_height"] == 720
    assert calls.get("keys", []) == [("Tip of the Day", ["Escape"], {"timeout": 5.0})]
    assert adapter._last_capture_complete is False
    assert not lock_path.exists()
    assert calls["terminated"] is not None


def test_render_target_updates_selected_slide_indices(tmp_path: Path):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)

    adapter.set_render_target({"slide_indices": [2, 3, 2, 0, "4"]})

    assert adapter._render_slide_indices == [2, 3, 4]
    artifact = adapter.describe_rendering()
    assert "2, 3, 4" in artifact.description


def test_render_to_png_selects_requested_slide_in_editor(tmp_path: Path, monkeypatch):
    odp = _make_sample_odp(tmp_path)
    adapter = LibreOfficeImpressAdapter(odp)
    adapter.set_render_target({"slide_indices": [3, 4]})
    calls = []

    monkeypatch.setattr("asil.adapters.libreoffice_impress.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("asil.adapters.libreoffice_impress.launch_gui_process", lambda *args, **kwargs: object())
    monkeypatch.setattr("asil.adapters.libreoffice_impress.capture_window_to_png", lambda output_path, **kwargs: Path(output_path))
    monkeypatch.setattr("asil.adapters.libreoffice_impress.ensure_user_access", lambda *args, **kwargs: None)
    monkeypatch.setattr("asil.adapters.libreoffice_impress.terminate_process", lambda *args, **kwargs: None)

    def fake_send_keys(title_pattern, keys, **kwargs):
        calls.append((title_pattern, list(keys), dict(kwargs)))
        return "window-id"

    monkeypatch.setattr("asil.adapters.libreoffice_impress.send_keys_to_window", fake_send_keys)

    adapter.render_to_png(tmp_path / "slides.png")

    assert calls == [
        (
            "Tip of the Day",
            ["Escape"],
            {"timeout": 5.0},
        ),
        (
            r".*LibreOffice Impress|.* - LibreOffice Impress",
            ["Home"],
            {"timeout": 30.0, "min_width": 1000, "min_height": 720},
        ),
        (
            r".*LibreOffice Impress|.* - LibreOffice Impress",
            ["Next"],
            {"timeout": 30.0, "min_width": 1000, "min_height": 720},
        ),
    ]


def test_rasterize_pdf_pages_rejects_empty_page_list(tmp_path: Path):
    pdf = tmp_path / "slides.pdf"
    pdf.write_text("pdf")

    try:
        rasterize_pdf_pages(pdf, tmp_path / "slides.png", [])
        assert False, "Expected ValueError for empty page list"
    except ValueError as exc:
        assert "at least one positive page number" in str(exc)
