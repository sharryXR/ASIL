"""ASIL adapter for LibreOffice Calc — Pattern A (ODF file manipulation)."""

from __future__ import annotations
import re
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
TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
NS = {"office": OFFICE_NS, "table": TABLE_NS, "text": TEXT_NS}
_CALC_WINDOW_PATTERN = r".*LibreOffice Calc|.* - LibreOffice Calc"


def _col_letter(index: int) -> str:
    """0 → A, 1 → B, ... 25 → Z, 26 → AA."""
    result = ""
    i = index
    while True:
        result = chr(65 + i % 26) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def _parse_cell_ref(ref: str) -> tuple[int, int]:
    """'B3' → (col=1, row=2)."""
    m = re.match(r"([A-Z]+)(\d+)", ref)
    if not m:
        raise ValueError(f"Invalid cell ref: {ref}")
    col_str, row_str = m.group(1), m.group(2)
    col = 0
    for ch in col_str:
        col = col * 26 + (ord(ch) - 64)
    return col - 1, int(row_str) - 1


class LibreOfficeAdapter(ASILAdapter):
    app_name = "LibreOffice Calc"
    supported_action_types = ["modify_file", "set_value"]

    def __init__(self, ods_path: str | Path) -> None:
        self.ods_path = Path(ods_path)

    @property
    def source_path(self) -> Path:
        return self.ods_path

    def clone(self, new_path: Path) -> "LibreOfficeAdapter":
        shutil.copy2(self.ods_path, new_path)
        return LibreOfficeAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        return {"ods_path": str(self.ods_path)}

    def _read_content_xml(self) -> bytes:
        with zipfile.ZipFile(self.ods_path) as zf:
            return zf.read("content.xml")

    def _write_content_xml(self, content: bytes) -> None:
        tmp = self.ods_path.with_suffix(".ods.tmp")
        with zipfile.ZipFile(self.ods_path, "r") as zin:
            with zipfile.ZipFile(tmp, "w") as zout:
                for item in zin.infolist():
                    if item.filename == "content.xml":
                        zout.writestr(item, content)
                    else:
                        zout.writestr(item, zin.read(item.filename))
        tmp.replace(self.ods_path)

    def observe(self) -> Observation:
        root = etree.fromstring(self._read_content_xml())
        elements: list[Element] = []
        sheet_names: list[str] = []

        for sheet in root.xpath(".//table:table", namespaces=NS):
            sname = sheet.get(f"{{{TABLE_NS}}}name", "Sheet")
            sheet_names.append(sname)
            for row_idx, row in enumerate(sheet.xpath("table:table-row", namespaces=NS)):
                col_idx = 0
                for cell in row.xpath("table:table-cell", namespaces=NS):
                    repeat = int(cell.get(f"{{{TABLE_NS}}}number-columns-repeated", "1"))
                    vtype = cell.get(f"{{{OFFICE_NS}}}value-type", "")
                    val = cell.get(f"{{{OFFICE_NS}}}value", "")
                    formula = cell.get(f"{{{TABLE_NS}}}formula", "")
                    text_parts = cell.xpath(".//text:p/text()", namespaces=NS)
                    text = "".join(text_parts)

                    if vtype or text or formula:
                        cell_id = f"{sname}!{_col_letter(col_idx)}{row_idx + 1}"
                        elements.append(Element(
                            id=cell_id,
                            type="cell",
                            label=cell_id,
                            value=val or text,
                            editable=True,
                            data_type=vtype or "string",
                            actions=["set_value", "set_formula", "format", "clear"],
                            metadata={"formula": formula} if formula else {},
                        ))
                    col_idx += repeat

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "spreadsheet",
                "active_document": self.ods_path.name,
                "document_path": str(self.ods_path),
            },
            data_summary=f"Spreadsheet with {len(sheet_names)} sheets: {sheet_names}, {len(elements)} cells",
        )

    def execute(self, action: Action) -> Observation:
        root = etree.fromstring(self._read_content_xml())

        for op in action.params.get("operations", []):
            sheet_name = op.get("sheet", "Sheet1")
            cell_ref = op.get("cell", "A1")
            new_value = op.get("value", "")
            vtype = op.get("value_type", "string")

            col, row = _parse_cell_ref(cell_ref)

            sheets = root.xpath(
                f".//table:table[@table:name='{sheet_name}']", namespaces=NS
            )
            if not sheets:
                continue
            sheet_elem = sheets[0]

            rows = sheet_elem.xpath("table:table-row", namespaces=NS)
            while len(rows) <= row:
                etree.SubElement(sheet_elem, f"{{{TABLE_NS}}}table-row")
                rows = sheet_elem.xpath("table:table-row", namespaces=NS)

            target_row = rows[row]
            cells = target_row.xpath("table:table-cell", namespaces=NS)
            while len(cells) <= col:
                etree.SubElement(target_row, f"{{{TABLE_NS}}}table-cell")
                cells = target_row.xpath("table:table-cell", namespaces=NS)

            cell_elem = cells[col]
            for child in list(cell_elem):
                cell_elem.remove(child)
            for attr in list(cell_elem.attrib):
                if "value" in attr or "formula" in attr:
                    del cell_elem.attrib[attr]

            cell_elem.set(f"{{{OFFICE_NS}}}value-type", vtype)
            if vtype == "float":
                cell_elem.set(f"{{{OFFICE_NS}}}value", new_value)
            tp = etree.SubElement(cell_elem, f"{{{TEXT_NS}}}p")
            tp.text = new_value

        content = etree.tostring(root, xml_declaration=True, encoding="utf-8")
        self._write_content_xml(content)
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real LibreOffice Calc window with spreadsheet chrome",
        )

    def _lock_file_path(self) -> Path:
        return self.ods_path.parent / f".~lock.{self.ods_path.name}#"

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
        home_path = self.ods_path.parent / "_calc_gui_home"
        shutil.rmtree(home_path, ignore_errors=True)
        for subdir in ("home", "config", "data", "cache"):
            (home_path / subdir).mkdir(parents=True, exist_ok=True)
        ensure_user_access(home_path, run_as_user="asilgui")
        return home_path

    def get_gui_session_spec(self) -> GUISessionSpec:
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to launch Calc.")
        self._clear_stale_lock()
        ensure_user_access(self.ods_path.parent, run_as_user="asilgui")
        home_path = self._gui_home()
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(lo, "--calc", str(self.ods_path)),
            window_title_pattern=_CALC_WINDOW_PATTERN,
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
                title_pattern=_CALC_WINDOW_PATTERN,
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
                    "LibreOffice Calc window is present but not fully visible or ready for interaction.",
                )
        finally:
            probe_path.unlink(missing_ok=True)

    def export_to_pdf(self, output_path: str | Path | None = None) -> Path:
        """Export the current ODS to PDF using LibreOffice CLI.

        Raises RuntimeError if LibreOffice is not installed.
        Returns the output PDF path.
        """
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError(
                "LibreOffice is not installed. Install it to use export_to_pdf()."
            )

        out = Path(output_path) if output_path else self.ods_path.with_suffix(".pdf")

        subprocess.run(
            [
                lo, "--headless", "--convert-to", "pdf",
                "--outdir", str(out.parent),
                str(self.ods_path),
            ],
            check=True,
            capture_output=True,
        )

        # LibreOffice outputs {input_stem}.pdf in outdir; rename if needed
        lo_output = out.parent / (self.ods_path.stem + ".pdf")
        if lo_output != out and lo_output.exists():
            shutil.move(str(lo_output), str(out))

        return out

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        """Capture the real LibreOffice Calc window for the current spreadsheet."""
        out = Path(output_path) if output_path else self.ods_path.with_suffix(".png")
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if lo is None:
            raise RuntimeError("LibreOffice is not installed. Install it to capture Calc window screenshots.")

        self._clear_stale_lock()
        ensure_user_access(self.ods_path.parent, run_as_user="asilgui")
        with tempfile.TemporaryDirectory(prefix="asil_calc_home_") as home_root:
            home_path = Path(home_root)
            for subdir in ("home", "config", "data", "cache"):
                (home_path / subdir).mkdir(parents=True, exist_ok=True)
            ensure_user_access(home_path, run_as_user="asilgui")

            proc = launch_gui_process(
                [lo, "--calc", str(self.ods_path)],
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
                    title_pattern=_CALC_WINDOW_PATTERN,
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
