"""ASIL adapter for LibreOffice Writer — Pattern A (ODF text document manipulation)."""

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
TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
NS = {"office": OFFICE_NS, "text": TEXT_NS, "table": TABLE_NS}

ODT_MIMETYPE = "application/vnd.oasis.opendocument.text"
_WRITER_WINDOW_PATTERN = r".*LibreOffice Writer|.* - LibreOffice Writer"
ODT_MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"'
    ' manifest:version="1.2">'
    f'<manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="{ODT_MIMETYPE}"/>'
    '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
    "</manifest:manifest>"
)


def _default_content_xml() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content'
        ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
        ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
        ' office:version="1.2">'
        "<office:body><office:text>"
        '<text:h text:outline-level="1">Project Brief</text:h>'
        "<text:p>Quarterly planning draft.</text:p>"
        "<text:p>Action items are pending review.</text:p>"
        "</office:text></office:body></office:document-content>"
    ).encode("utf-8")


class LibreOfficeWriterAdapter(ASILAdapter):
    app_name = "LibreOffice Writer"
    supported_action_types = ["modify_file", "set_value"]

    def __init__(self, odt_path: str | Path) -> None:
        self.odt_path = Path(odt_path)

    @classmethod
    def from_evaluation_context(cls, tmp: Path, sandbox=None, mock: bool = False) -> "LibreOfficeWriterAdapter":
        odt = tmp / "writer.odt"
        with zipfile.ZipFile(odt, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mimetype", ODT_MIMETYPE, compress_type=zipfile.ZIP_STORED)
            zf.writestr("META-INF/manifest.xml", ODT_MANIFEST)
            zf.writestr("content.xml", _default_content_xml())
        return cls(odt)

    @property
    def source_path(self) -> Path:
        return self.odt_path

    def clone(self, new_path: Path) -> "LibreOfficeWriterAdapter":
        shutil.copy2(self.odt_path, new_path)
        return LibreOfficeWriterAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        return {"odt_path": str(self.odt_path)}

    def _read_content_xml(self) -> bytes:
        with zipfile.ZipFile(self.odt_path) as zf:
            return zf.read("content.xml")

    def _write_content_xml(self, content: bytes) -> None:
        tmp = self.odt_path.with_suffix(".odt.tmp")
        with zipfile.ZipFile(self.odt_path, "r") as zin:
            with zipfile.ZipFile(tmp, "w") as zout:
                for item in zin.infolist():
                    if item.filename == "content.xml":
                        zout.writestr(item, content)
                    else:
                        zout.writestr(item, zin.read(item.filename))
        tmp.replace(self.odt_path)

    def _text_root(self) -> etree._Element:
        root = etree.fromstring(self._read_content_xml())
        text_root = root.find(".//office:text", namespaces=NS)
        if text_root is None:
            raise RuntimeError("ODT content.xml missing office:text root")
        return text_root

    def observe(self) -> Observation:
        root = etree.fromstring(self._read_content_xml())
        text_root = root.find(".//office:text", namespaces=NS)
        elements: list[Element] = []
        paragraphs = 0
        headings = 0
        if text_root is not None:
            for idx, node in enumerate(text_root.xpath("./text:h | ./text:p", namespaces=NS), start=1):
                tag = etree.QName(node).localname
                text_content = "".join(node.itertext()).strip()
                style_name = node.get(f"{{{TEXT_NS}}}style-name", "")
                if tag == "h":
                    headings += 1
                    elem_id = f"heading:{headings}"
                    elem_type = "heading"
                    metadata = {"outline_level": int(node.get(f'{{{TEXT_NS}}}outline-level', '1'))}
                else:
                    paragraphs += 1
                    elem_id = f"paragraph:{paragraphs}"
                    elem_type = "paragraph"
                    metadata = {}
                elements.append(
                    Element(
                        id=elem_id,
                        type=elem_type,
                        label=text_content[:80] or elem_id,
                        value={"text_content": text_content, "style_name": style_name},
                        editable=True,
                        actions=["set_text", "set_style", "delete"],
                        metadata=metadata,
                    )
                )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "document_page",
                "active_document": self.odt_path.name,
                "document_path": str(self.odt_path),
            },
            data_summary=f"Writer document with {headings} headings and {paragraphs} paragraphs",
        )

    def execute(self, action: Action) -> Observation:
        root = etree.fromstring(self._read_content_xml())
        text_root = root.find(".//office:text", namespaces=NS)
        if text_root is None:
            raise RuntimeError("ODT content.xml missing office:text root")

        for op in action.params.get("operations", []):
            op_action = op.get("action", "set_paragraph_text")
            if op_action == "set_paragraph_text":
                index = int(op["index"])
                text = str(op["text"])
                paragraphs = text_root.xpath("./text:p", namespaces=NS)
                if 1 <= index <= len(paragraphs):
                    node = paragraphs[index - 1]
                    for child in list(node):
                        node.remove(child)
                    node.text = text
            elif op_action == "set_heading_text":
                index = int(op["index"])
                text = str(op["text"])
                headings = text_root.xpath("./text:h", namespaces=NS)
                if 1 <= index <= len(headings):
                    node = headings[index - 1]
                    for child in list(node):
                        node.remove(child)
                    node.text = text
            elif op_action == "add_paragraph":
                node = etree.SubElement(text_root, f"{{{TEXT_NS}}}p")
                node.text = str(op["text"])
                style_name = op.get("style_name")
                if style_name:
                    node.set(f"{{{TEXT_NS}}}style-name", str(style_name))
            elif op_action == "add_heading":
                node = etree.SubElement(text_root, f"{{{TEXT_NS}}}h")
                node.set(f"{{{TEXT_NS}}}outline-level", str(op.get("level", 1)))
                node.text = str(op["text"])
            else:
                raise ValueError(f"Unsupported Writer operation: {op_action}")

        content = etree.tostring(root, xml_declaration=True, encoding="utf-8")
        self._write_content_xml(content)
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def _lock_file_path(self) -> Path:
        return self.odt_path.parent / f".~lock.{self.odt_path.name}#"

    def _clear_stale_lock(self) -> None:
        self._lock_file_path().unlink(missing_ok=True)

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
        home_path = self.odt_path.parent / "_writer_gui_home"
        shutil.rmtree(home_path, ignore_errors=True)
        for subdir in ("home", "config", "data", "cache"):
            (home_path / subdir).mkdir(parents=True, exist_ok=True)
        ensure_user_access(home_path, run_as_user="asilgui")
        return home_path

    def get_gui_session_spec(self) -> GUISessionSpec:
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to launch Writer.")
        self._clear_stale_lock()
        ensure_user_access(self.odt_path.parent, run_as_user="asilgui")
        home_path = self._gui_home()
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(lo, "--writer", str(self.odt_path)),
            window_title_pattern=_WRITER_WINDOW_PATTERN,
            window_class_pattern=r"libreoffice|soffice|LibreOffice",
            run_as_user="asilgui",
            startup_timeout_s=60.0,
            post_launch_delay_s=6.0,
            post_launch_callback=self._dismiss_startup_dialogs,
            ui_ready_probe=self._probe_gui_ready,
            close_callback=self._clear_stale_lock,
            min_width=900,
            min_height=700,
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
                title_pattern=_WRITER_WINDOW_PATTERN,
                window_class_pattern=r"libreoffice|soffice|LibreOffice",
                timeout=30.0,
                settle_delay=2.0,
                min_width=900,
                min_height=700,
                capture_metadata=capture_metadata,
            )
            if not capture_metadata.get("capture_complete", True):
                raise GUISessionStartupError(
                    "startup_overlay_blocked",
                    "LibreOffice Writer window is present but not fully visible or ready for interaction.",
                )
        finally:
            probe_path.unlink(missing_ok=True)

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real LibreOffice Writer window with document chrome",
        )

    def export_to_pdf(self, output_path: str | Path | None = None) -> Path:
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to use export_to_pdf().")

        out = Path(output_path) if output_path else self.odt_path.with_suffix(".pdf")
        subprocess.run(
            [lo, "--headless", "--convert-to", "pdf", "--outdir", str(out.parent), str(self.odt_path)],
            check=True,
            capture_output=True,
        )
        lo_output = out.parent / (self.odt_path.stem + ".pdf")
        if lo_output != out and lo_output.exists():
            shutil.move(str(lo_output), str(out))
        if not out.exists():
            raise RuntimeError(
                f"LibreOffice Writer conversion did not produce a PDF at {out}. "
                "Ensure libreoffice-writer is installed in the runtime image."
            )
        return out

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        out = Path(output_path) if output_path else self.odt_path.with_suffix(".png")
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to capture Writer window screenshots.")

        self._clear_stale_lock()
        ensure_user_access(self.odt_path.parent, run_as_user="asilgui")
        with tempfile.TemporaryDirectory(prefix="asil_writer_home_") as home_root:
            home_path = Path(home_root)
            for subdir in ("home", "config", "data", "cache"):
                (home_path / subdir).mkdir(parents=True, exist_ok=True)
            ensure_user_access(home_path, run_as_user="asilgui")

            proc = launch_gui_process(
                [lo, "--writer", str(self.odt_path)],
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
                capture_metadata = {"capture_complete": True}
                capture_window_to_png(
                    out,
                    title_pattern=_WRITER_WINDOW_PATTERN,
                    timeout=60.0,
                    margin=12,
                    settle_delay=6.0,
                    min_width=900,
                    min_height=700,
                    capture_metadata=capture_metadata,
                )
                self._last_capture_complete = bool(capture_metadata.get("capture_complete", True))
            finally:
                terminate_process(proc)
                self._clear_stale_lock()
        return out
