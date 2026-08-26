#!/usr/bin/env python3
"""UNO worker — the *native LibreOffice programmatic interface* execution backend
for the matched-baseline experiment (#3, reviewer WgDr).

Runs under the interpreter that ``python3-uno`` targets (system python3.10 on the
Ubuntu 22.04 eval image), connects to a headless ``soffice`` UNO socket, and holds
a single live document. It speaks newline-delimited JSON over stdin/stdout so the
py3.11 orchestrator (which imports the ASIL package + calls the model) can drive it
without importing ``uno`` itself (PyUNO is built against system python, not the 3.11
venv).

Protocol (one JSON object per stdin line -> one JSON object per stdout line):
  {"cmd":"load","url":"file:///abs/path.ods"}          -> {"ok":true}
  {"cmd":"observe"}                                     -> {"ok":true,"text":"<native state>"}
  {"cmd":"exec","code":"document...","save":true}       -> {"ok":true,"stdout":"...","error":null}
  {"cmd":"save"}                                        -> {"ok":true}
  {"cmd":"quit"}                                        -> (exits)

The model-authored code is exec()'d with the live UNO objects in scope
(``document``, ``desktop``, ``smgr``, ``ctx``, ``uno``) exactly like a UNO macro
gets ``ThisComponent`` — this is the app's own automation API with full coverage,
which is the whole point of the comparison against ASIL's curated contract.
"""
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

import uno
from com.sun.star.beans import PropertyValue


def _pv(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


_FILTER_BY_EXT = {
    ".ods": "calc8",
    ".odt": "writer8",
    ".odp": "impress8",
}


class Worker:
    def __init__(self, port: int) -> None:
        local_ctx = uno.getComponentContext()
        resolver = local_ctx.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_ctx
        )
        ctx = resolver.resolve(
            f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
        )
        self.ctx = ctx
        self.smgr = ctx.ServiceManager
        self.desktop = self.smgr.createInstanceWithContext(
            "com.sun.star.frame.Desktop", ctx
        )
        self.doc = None
        self.url = None

    # ---- document lifecycle -------------------------------------------------
    def load(self, url: str):
        if self.doc is not None:
            try:
                self.doc.close(False)
            except Exception:
                pass
            self.doc = None
        # Hidden, editable load of the task's initial artifact.
        self.doc = self.desktop.loadComponentFromURL(
            url, "_blank", 0, (_pv("Hidden", True),)
        )
        self.url = url
        return {"ok": True}

    def _filter_for(self, url: str) -> str:
        for ext, flt in _FILTER_BY_EXT.items():
            if url.lower().endswith(ext):
                return flt
        return "calc8"

    def save(self):
        if self.doc is None or self.url is None:
            return {"ok": False, "error": "no document loaded"}
        flt = self._filter_for(self.url)
        self.doc.storeToURL(self.url, (_pv("FilterName", flt),))
        return {"ok": True}

    # ---- native observation -------------------------------------------------
    def observe(self) -> dict:
        if self.doc is None:
            return {"ok": False, "error": "no document loaded"}
        try:
            if self.doc.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
                return {"ok": True, "text": self._observe_calc()}
            if self.doc.supportsService("com.sun.star.text.TextDocument"):
                return {"ok": True, "text": self._observe_writer()}
            if self.doc.supportsService("com.sun.star.presentation.PresentationDocument") or \
               self.doc.supportsService("com.sun.star.drawing.DrawingDocument"):
                return {"ok": True, "text": self._observe_impress()}
        except Exception as exc:  # observation must never kill the loop
            return {"ok": True, "text": f"(observation error: {exc})"}
        return {"ok": True, "text": "(unknown document type)"}

    def _observe_calc(self) -> str:
        lines = []
        sheets = self.doc.Sheets
        for si in range(sheets.Count):
            sheet = sheets.getByIndex(si)
            name = sheet.Name
            cursor = sheet.createCursor()
            cursor.gotoEndOfUsedArea(False)
            last_col = cursor.RangeAddress.EndColumn
            last_row = cursor.RangeAddress.EndRow
            cells = []
            for r in range(last_row + 1):
                for c in range(last_col + 1):
                    cell = sheet.getCellByPosition(c, r)
                    s = cell.getString()
                    if s == "":
                        continue
                    col = _col_letter(c)
                    formula = cell.getFormula()
                    if formula and formula.startswith("="):
                        cells.append(f"{col}{r + 1}={s} (formula {formula})")
                    else:
                        cells.append(f"{col}{r + 1}={s}")
            body = "; ".join(cells) if cells else "(empty)"
            lines.append(f"Sheet '{name}' [used {last_col + 1}x{last_row + 1}]: {body}")
        return "\n".join(lines)

    def _observe_writer(self) -> str:
        lines = []
        enum = self.doc.Text.createEnumeration()
        i = 0
        while enum.hasMoreElements():
            el = enum.nextElement()
            if el.supportsService("com.sun.star.text.Paragraph"):
                i += 1
                lines.append(f"P{i}: {el.getString()}")
        return "\n".join(lines) if lines else "(empty document)"

    def _observe_impress(self) -> str:
        lines = []
        slides = self.doc.DrawPages
        for si in range(slides.Count):
            slide = slides.getByIndex(si)
            parts = []
            for shi in range(slide.Count):
                shape = slide.getByIndex(shi)
                try:
                    txt = shape.getString()
                except Exception:
                    txt = ""
                if txt:
                    parts.append(txt.replace("\n", " / "))
            body = " | ".join(parts) if parts else "(no text)"
            lines.append(f"Slide {si + 1}: {body}")
        return "\n".join(lines)

    # ---- native action execution -------------------------------------------
    def exec_code(self, code: str, do_save: bool = True) -> dict:
        if self.doc is None:
            return {"ok": False, "error": "no document loaded"}
        g = {
            "uno": uno,
            "PropertyValue": PropertyValue,
            "document": self.doc,
            "ThisComponent": self.doc,
            "desktop": self.desktop,
            "smgr": self.smgr,
            "ctx": self.ctx,
        }
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, g)  # noqa: S102 — this is the baseline's action channel
        except Exception:
            return {"ok": False, "stdout": buf.getvalue(), "error": traceback.format_exc()}
        if do_save:
            try:
                self.save()
            except Exception:
                return {"ok": False, "stdout": buf.getvalue(),
                        "error": "exec ok but save failed:\n" + traceback.format_exc()}
        return {"ok": True, "stdout": buf.getvalue(), "error": None}


def _col_letter(index: int) -> str:
    result = ""
    i = index
    while True:
        result = chr(65 + i % 26) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 2002
    worker = Worker(port)
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad json: {exc}"}) + "\n")
            sys.stdout.flush()
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        try:
            if cmd == "load":
                resp = worker.load(req["url"])
            elif cmd == "observe":
                resp = worker.observe()
            elif cmd == "exec":
                resp = worker.exec_code(req["code"], req.get("save", True))
            elif cmd == "save":
                resp = worker.save()
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as exc:
            resp = {"ok": False, "error": f"{exc}\n{traceback.format_exc()}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
