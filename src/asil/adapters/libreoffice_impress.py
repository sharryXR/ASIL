"""ASIL adapter for LibreOffice Impress — Pattern A (ODP slide manipulation)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    capture_window_to_png,
    ensure_user_access,
    launch_gui_process,
    send_keys_to_window,
    terminate_process,
)

OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
DRAW_NS = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
PRESENTATION_NS = "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
SVG_NS = "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
NS = {
    "office": OFFICE_NS,
    "text": TEXT_NS,
    "draw": DRAW_NS,
    "presentation": PRESENTATION_NS,
    "svg": SVG_NS,
}

ODP_MIMETYPE = "application/vnd.oasis.opendocument.presentation"
_IMPRESS_WINDOW_PATTERN = r".*LibreOffice Impress|.* - LibreOffice Impress"
ODP_MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"'
    ' manifest:version="1.2">'
    f'<manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="{ODP_MIMETYPE}"/>'
    '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
    "</manifest:manifest>"
)


def _default_content_xml() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content'
        ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
        ' xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"'
        ' xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
        ' xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"'
        ' office:version="1.2">'
        "<office:body><office:presentation>"
        '<draw:page draw:name="slide1">'
        '<draw:frame presentation:class="title" svg:x="2cm" svg:y="1.5cm" svg:width="22cm" svg:height="3cm">'
        "<draw:text-box><text:p>Roadmap Overview</text:p></draw:text-box>"
        "</draw:frame>"
        '<draw:frame presentation:class="outline" svg:x="2cm" svg:y="5cm" svg:width="22cm" svg:height="8cm">'
        "<draw:text-box><text:p>Kickoff planning is in progress.</text:p></draw:text-box>"
        "</draw:frame>"
        "</draw:page>"
        '<draw:page draw:name="slide2">'
        '<draw:frame presentation:class="title" svg:x="2cm" svg:y="1.5cm" svg:width="22cm" svg:height="3cm">'
        "<draw:text-box><text:p>Next Steps</text:p></draw:text-box>"
        "</draw:frame>"
        '<draw:frame presentation:class="outline" svg:x="2cm" svg:y="5cm" svg:width="22cm" svg:height="8cm">'
        "<draw:text-box><text:p>Confirm owners for launch tasks.</text:p></draw:text-box>"
        "</draw:frame>"
        "</draw:page>"
        "</office:presentation></office:body></office:document-content>"
    ).encode("utf-8")


def _frame_text_box(frame: etree._Element) -> etree._Element:
    text_box = frame.find("draw:text-box", namespaces=NS)
    if text_box is None:
        text_box = etree.SubElement(frame, f"{{{DRAW_NS}}}text-box")
    return text_box


def _paragraphs_from_frame(frame: etree._Element) -> list[etree._Element]:
    text_box = _frame_text_box(frame)
    paragraphs = text_box.findall("text:p", namespaces=NS)
    if not paragraphs:
        paragraphs = [etree.SubElement(text_box, f"{{{TEXT_NS}}}p")]
    return paragraphs


def _set_frame_paragraphs(frame: etree._Element, lines: list[str]) -> None:
    text_box = _frame_text_box(frame)
    for child in list(text_box):
        text_box.remove(child)
    if not lines:
        lines = [""]
    for line in lines:
        paragraph = etree.SubElement(text_box, f"{{{TEXT_NS}}}p")
        paragraph.text = line


def _make_frame(frame_class: str, x: str, y: str, width: str, height: str, lines: list[str]) -> etree._Element:
    frame = etree.Element(f"{{{DRAW_NS}}}frame")
    frame.set(f"{{{PRESENTATION_NS}}}class", frame_class)
    frame.set(f"{{{SVG_NS}}}x", x)
    frame.set(f"{{{SVG_NS}}}y", y)
    frame.set(f"{{{SVG_NS}}}width", width)
    frame.set(f"{{{SVG_NS}}}height", height)
    _set_frame_paragraphs(frame, lines)
    return frame


class LibreOfficeImpressAdapter(ASILAdapter):
    app_name = "LibreOffice Impress"
    supported_action_types = ["modify_file", "set_value"]

    def __init__(self, odp_path: str | Path) -> None:
        self.odp_path = Path(odp_path)
        self._render_slide_indices: list[int] = [1]

    @classmethod
    def from_evaluation_context(
        cls, tmp: Path, sandbox=None, mock: bool = False
    ) -> "LibreOfficeImpressAdapter":
        odp = tmp / "slides.odp"
        with zipfile.ZipFile(odp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mimetype", ODP_MIMETYPE, compress_type=zipfile.ZIP_STORED)
            zf.writestr("META-INF/manifest.xml", ODP_MANIFEST)
            zf.writestr("content.xml", _default_content_xml())
        return cls(odp)

    @property
    def source_path(self) -> Path:
        return self.odp_path

    def clone(self, new_path: Path) -> "LibreOfficeImpressAdapter":
        shutil.copy2(self.odp_path, new_path)
        return LibreOfficeImpressAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        return {"odp_path": str(self.odp_path)}

    def _read_content_xml(self) -> bytes:
        with zipfile.ZipFile(self.odp_path) as zf:
            return zf.read("content.xml")

    def _write_content_xml(self, content: bytes) -> None:
        tmp = self.odp_path.with_suffix(".odp.tmp")
        with zipfile.ZipFile(self.odp_path, "r") as zin:
            with zipfile.ZipFile(tmp, "w") as zout:
                for item in zin.infolist():
                    if item.filename == "content.xml":
                        zout.writestr(item, content)
                    else:
                        zout.writestr(item, zin.read(item.filename))
        tmp.replace(self.odp_path)

    def _presentation_root(self) -> etree._Element:
        root = etree.fromstring(self._read_content_xml())
        presentation_root = root.find(".//office:presentation", namespaces=NS)
        if presentation_root is None:
            raise RuntimeError("ODP content.xml missing office:presentation root")
        return presentation_root

    def _slide_pages(self, presentation_root: etree._Element) -> list[etree._Element]:
        return presentation_root.findall("draw:page", namespaces=NS)

    def _frame_by_class(self, slide: etree._Element, frame_class: str) -> etree._Element | None:
        return slide.find(f"draw:frame[@presentation:class='{frame_class}']", namespaces=NS)

    def observe(self) -> Observation:
        root = etree.fromstring(self._read_content_xml())
        presentation_root = root.find(".//office:presentation", namespaces=NS)
        elements: list[Element] = []
        slide_count = 0
        if presentation_root is not None:
            slides = presentation_root.findall("draw:page", namespaces=NS)
            slide_count = len(slides)
            for slide_index, slide in enumerate(slides, start=1):
                title_frame = self._frame_by_class(slide, "title")
                if title_frame is not None:
                    title_text = "\n".join("".join(p.itertext()).strip() for p in _paragraphs_from_frame(title_frame)).strip()
                    elements.append(
                        Element(
                            id=f"slide:{slide_index}:title",
                            type="slide_title",
                            label=title_text or f"Slide {slide_index} Title",
                            value={"text_content": title_text},
                            editable=True,
                            actions=["set_text"],
                            metadata={"slide_index": slide_index, "slide_name": slide.get(f'{{{DRAW_NS}}}name', f'slide{slide_index}')},
                        )
                    )
                body_frame = self._frame_by_class(slide, "outline")
                if body_frame is not None:
                    for body_index, paragraph in enumerate(_paragraphs_from_frame(body_frame), start=1):
                        text_content = "".join(paragraph.itertext()).strip()
                        elements.append(
                            Element(
                                id=f"slide:{slide_index}:body:{body_index}",
                                type="slide_body",
                                label=text_content[:80] or f"Slide {slide_index} Body {body_index}",
                                value={"text_content": text_content},
                                editable=True,
                                actions=["set_text", "append_text"],
                                metadata={"slide_index": slide_index, "body_index": body_index},
                            )
                        )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "slide_canvas",
                "active_document": self.odp_path.name,
                "document_path": str(self.odp_path),
            },
            data_summary=f"Presentation with {slide_count} slides and {len(elements)} visible text elements",
        )

    def execute(self, action: Action) -> Observation:
        root = etree.fromstring(self._read_content_xml())
        presentation_root = root.find(".//office:presentation", namespaces=NS)
        if presentation_root is None:
            raise RuntimeError("ODP content.xml missing office:presentation root")

        slides = self._slide_pages(presentation_root)
        for op in action.params.get("operations", []):
            op_action = op.get("action", "set_slide_title")
            slide_index = int(op.get("slide_index", 1))
            while len(slides) < slide_index:
                slides.append(self._append_slide(presentation_root, f"Slide {len(slides) + 1}", ["Add content here."]))

            slide = slides[slide_index - 1]
            if op_action == "set_slide_title":
                title_frame = self._frame_by_class(slide, "title")
                if title_frame is None:
                    title_frame = _make_frame("title", "2cm", "1.5cm", "22cm", "3cm", [""])
                    slide.insert(0, title_frame)
                _set_frame_paragraphs(title_frame, [self._operation_text(op)])
            elif op_action == "set_slide_body":
                body_index = int(op.get("body_index", 1))
                body_frame = self._frame_by_class(slide, "outline")
                if body_frame is None:
                    body_frame = _make_frame("outline", "2cm", "5cm", "22cm", "8cm", [""])
                    slide.append(body_frame)
                lines = ["".join(p.itertext()).strip() for p in _paragraphs_from_frame(body_frame)]
                while len(lines) < body_index:
                    lines.append("")
                lines[body_index - 1] = self._operation_text(op)
                _set_frame_paragraphs(body_frame, lines)
            elif op_action == "append_slide_body":
                body_frame = self._frame_by_class(slide, "outline")
                if body_frame is None:
                    body_frame = _make_frame("outline", "2cm", "5cm", "22cm", "8cm", [])
                    slide.append(body_frame)
                lines = ["".join(p.itertext()).strip() for p in _paragraphs_from_frame(body_frame)]
                if lines == [""]:
                    lines = []
                lines.append(self._operation_text(op))
                _set_frame_paragraphs(body_frame, lines)
            elif op_action == "add_slide":
                title = str(op["title"])
                body = [str(line) for line in op.get("body", [])]
                slides.append(self._append_slide(presentation_root, title, body))
            else:
                raise ValueError(f"Unsupported Impress operation: {op_action}")

        content = etree.tostring(root, xml_declaration=True, encoding="utf-8")
        self._write_content_xml(content)
        return self.observe()

    def _append_slide(self, presentation_root: etree._Element, title: str, body: list[str]) -> etree._Element:
        slide_index = len(self._slide_pages(presentation_root)) + 1
        slide = etree.SubElement(presentation_root, f"{{{DRAW_NS}}}page")
        slide.set(f"{{{DRAW_NS}}}name", f"slide{slide_index}")
        slide.append(_make_frame("title", "2cm", "1.5cm", "22cm", "3cm", [title]))
        slide.append(_make_frame("outline", "2cm", "5cm", "22cm", "8cm", body or [""]))
        return slide

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def set_render_target(self, render_target: dict[str, object] | None) -> None:
        """Select which slide(s) should be rendered for GUI artifacts."""
        slide_indices = None
        if isinstance(render_target, dict):
            slide_indices = render_target.get("slide_indices")
        if isinstance(slide_indices, list):
            cleaned = []
            for slide_index in slide_indices:
                try:
                    page = int(slide_index)
                except (TypeError, ValueError):
                    continue
                if page > 0 and page not in cleaned:
                    cleaned.append(page)
            if cleaned:
                self._render_slide_indices = cleaned
                return
        self._render_slide_indices = [1]

    @staticmethod
    def _operation_text(op: dict[str, object]) -> str:
        """Accept a few common LLM aliases for visible slide text payloads."""
        if "text" in op and op["text"] is not None:
            return str(op["text"])
        if "value" in op and op["value"] is not None:
            return str(op["value"])
        if "title" in op and op["title"] is not None:
            return str(op["title"])
        body = op.get("body")
        if isinstance(body, list) and body:
            return str(body[0])
        if isinstance(body, str):
            return body
        raise KeyError("text")

    def describe_rendering(self) -> RenderArtifact:
        if len(self._render_slide_indices) == 1:
            description = f"Screenshot of the real LibreOffice Impress window focused on slide {self._render_slide_indices[0]}"
        else:
            description = (
                "Screenshot of the real LibreOffice Impress window with requested slide focus "
                + ", ".join(str(slide) for slide in self._render_slide_indices)
            )
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description=description,
        )

    def _dismiss_tip_of_the_day(self) -> None:
        try:
            send_keys_to_window("Tip of the Day", ["Escape"], timeout=5.0)
        except Exception:
            pass

    def _dismiss_document_recovery(self) -> None:
        for pattern in ("LibreOffice .*Document Recovery", "Document Recovery"):
            try:
                send_keys_to_window(pattern, ["Escape"], timeout=3.0)
                break
            except Exception:
                continue

    def _dismiss_startup_dialogs(self) -> None:
        self._dismiss_tip_of_the_day()
        self._dismiss_document_recovery()

    def _gui_home(self) -> Path:
        home_path = self.odp_path.parent / "_impress_gui_home"
        shutil.rmtree(home_path, ignore_errors=True)
        for subdir in ("home", "config", "data", "cache"):
            (home_path / subdir).mkdir(parents=True, exist_ok=True)
        ensure_user_access(home_path, run_as_user="asilgui")
        return home_path

    def get_gui_session_spec(self) -> GUISessionSpec:
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to launch Impress.")
        self._clear_stale_lock()
        ensure_user_access(self.odp_path.parent, run_as_user="asilgui")
        home_path = self._gui_home()
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(lo, "--impress", str(self.odp_path)),
            window_title_pattern=_IMPRESS_WINDOW_PATTERN,
            window_class_pattern=r"libreoffice|soffice|LibreOffice",
            run_as_user="asilgui",
            startup_timeout_s=60.0,
            post_launch_delay_s=6.0,
            post_launch_callback=lambda: (self._dismiss_startup_dialogs(), self._focus_render_slide()),
            ui_ready_probe=self._probe_gui_ready,
            close_callback=self._clear_stale_lock,
            min_width=1000,
            min_height=720,
            persist_shortcuts=("ctrl+s",),
            extra_env={
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "HOME": str(home_path / "home"),
                "XDG_CONFIG_HOME": str(home_path / "config"),
                "XDG_DATA_HOME": str(home_path / "data"),
                "XDG_CACHE_HOME": str(home_path / "cache"),
            },
        )

    def _probe_gui_ready(self, _session=None) -> None:
        from asil.gui_agent.session import GUISessionStartupError

        capture_metadata = {"capture_complete": True}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            probe_path = Path(handle.name)
        try:
            capture_window_to_png(
                probe_path,
                title_pattern=_IMPRESS_WINDOW_PATTERN,
                window_class_pattern=r"libreoffice|soffice|LibreOffice",
                timeout=30.0,
                settle_delay=2.0,
                min_width=1000,
                min_height=720,
                capture_metadata=capture_metadata,
            )
            if not capture_metadata.get("capture_complete", True):
                raise GUISessionStartupError(
                    "startup_overlay_blocked",
                    "LibreOffice Impress window is present but not fully visible or ready for interaction.",
                )
        finally:
            probe_path.unlink(missing_ok=True)

    def export_to_pdf(self, output_path: str | Path | None = None) -> Path:
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to use export_to_pdf().")

        out = Path(output_path) if output_path else self.odp_path.with_suffix(".pdf")
        subprocess.run(
            [lo, "--headless", "--convert-to", "pdf", "--outdir", str(out.parent), str(self.odp_path)],
            check=True,
            capture_output=True,
        )
        lo_output = out.parent / (self.odp_path.stem + ".pdf")
        if lo_output != out and lo_output.exists():
            shutil.move(str(lo_output), str(out))
        if not out.exists():
            raise RuntimeError(
                f"LibreOffice Impress conversion did not produce a PDF at {out}. "
                "Ensure libreoffice-impress is installed in the runtime image."
            )
        return out

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        out = Path(output_path) if output_path else self.odp_path.with_suffix(".png")
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to capture Impress window screenshots.")

        self._clear_stale_lock()
        ensure_user_access(self.odp_path.parent, run_as_user="asilgui")
        with tempfile.TemporaryDirectory(prefix="asil_impress_home_") as home_root:
            home_path = Path(home_root)
            for subdir in ("home", "config", "data", "cache"):
                (home_path / subdir).mkdir(parents=True, exist_ok=True)
            ensure_user_access(home_path, run_as_user="asilgui")

            proc = launch_gui_process(
                [lo, "--impress", str(self.odp_path)],
                extra_env={
                    "LIBGL_ALWAYS_SOFTWARE": "1",
                    "HOME": str(home_path / "home"),
                    "XDG_CONFIG_HOME": str(home_path / "config"),
                    "XDG_DATA_HOME": str(home_path / "data"),
                    "XDG_CACHE_HOME": str(home_path / "cache"),
                },
                run_as_user="asilgui",
            )
            try:
                self._dismiss_tip_of_the_day()
                self._focus_render_slide()
                capture_metadata = {"capture_complete": True}
                capture_window_to_png(
                    out,
                    title_pattern=_IMPRESS_WINDOW_PATTERN,
                    timeout=60.0,
                    margin=12,
                    settle_delay=6.0,
                    min_width=1000,
                    min_height=720,
                    capture_metadata=capture_metadata,
                )
                self._last_capture_complete = bool(capture_metadata.get("capture_complete", True))
            finally:
                terminate_process(proc)
                self._clear_stale_lock()
        return out

    def _focus_render_slide(self) -> None:
        max_slide = max(1, len(self._slide_pages(self._presentation_root())))
        target_slide = min(max(int(self._render_slide_indices[0]), 1), max_slide)
        if target_slide <= 1:
            return
        send_keys_to_window(
            _IMPRESS_WINDOW_PATTERN,
            ["Home"],
            timeout=30.0,
            min_width=1000,
            min_height=720,
        )
        send_keys_to_window(
            _IMPRESS_WINDOW_PATTERN,
            ["Next"] * (target_slide - 1),
            timeout=30.0,
            min_width=1000,
            min_height=720,
        )

    def _lock_file_path(self) -> Path:
        return self.odp_path.parent / f".~lock.{self.odp_path.name}#"

    def _clear_stale_lock(self) -> None:
        self._lock_file_path().unlink(missing_ok=True)
