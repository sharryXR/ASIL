"""ASIL adapter for draw.io — lightweight diagram state with webpage-style rendering."""

from __future__ import annotations

import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.protocol import Action, Element, Observation
from asil.rendering import RenderArtifact, capture_html_to_png, capture_url_to_png, html_page


_LIVE_EDITOR_XML_SCRIPT = r"""
async () => {
  const isEditorUi = (candidate) => Boolean(
    candidate && candidate.editor && candidate.editor.graph
  );
  let editorUi = [
    window.__ASIL_DRAWIO_UI__,
    window.editorUi,
    window.ui,
    window.app,
  ].find(isEditorUi) || null;

  // diagrams.net keeps its App instance in App.main's local scope. Its
  // resize handler resolves windowResized on the instance at event time, so
  // a short-lived prototype hook exposes that live instance without changing
  // the document or relying on the immutable launch URL.
  if (!editorUi && window.EditorUi && window.EditorUi.prototype) {
    const prototype = window.EditorUi.prototype;
    const originalWindowResized = prototype.windowResized;

    if (typeof originalWindowResized === 'function') {
      prototype.windowResized = function(...args) {
        if (isEditorUi(this)) {
          editorUi = this;
        }
        return originalWindowResized.apply(this, args);
      };

      try {
        window.dispatchEvent(new Event('resize'));
        const deadline = Date.now() + 1000;

        while (!editorUi && Date.now() < deadline) {
          await new Promise((resolve) => window.setTimeout(resolve, 50));
        }
      } finally {
        prototype.windowResized = originalWindowResized;
      }
    }
  }

  if (!editorUi) {
    return null;
  }
  if (typeof editorUi.getFileData === 'function') {
    const xml = editorUi.getFileData(true);

    if (typeof xml === 'string' && xml.length > 0) {
      return xml;
    }
  }
  if (editorUi.editor && typeof editorUi.editor.getGraphXml === 'function' &&
      window.mxUtils && typeof window.mxUtils.getXml === 'function') {
    return window.mxUtils.getXml(editorUi.editor.getGraphXml());
  }
  return null;
}
"""


def _default_state() -> dict[str, Any]:
    return {
        "canvas": {"width": 960, "height": 640, "background": "#f8fafc"},
        "shapes": [
            {
                "id": "start",
                "label": "Start",
                "shape_kind": "ellipse",
                "x": 80,
                "y": 140,
                "width": 120,
                "height": 70,
                "fill": "#dbeafe",
                "stroke": "#2563eb",
            },
            {
                "id": "review",
                "label": "Review",
                "shape_kind": "rectangle",
                "x": 300,
                "y": 140,
                "width": 150,
                "height": 80,
                "fill": "#fef3c7",
                "stroke": "#d97706",
            },
        ],
        "connectors": [
            {
                "id": "flow_1",
                "source": "start",
                "target": "review",
                "label": "submit",
                "stroke": "#64748b",
            }
        ],
    }


class DrawioAdapter(ASILAdapter):
    app_name = "draw.io"
    supported_action_types = ["modify_file"]

    def __init__(self, diagram_path: str | Path, base_url: str = "") -> None:
        self.diagram_path = Path(diagram_path)
        self.base_url = base_url.rstrip("/")
        self.clear_gui_shadow_state()
        if not self.diagram_path.exists():
            self.setup_state("default")
        elif not self.diagram_xml_path.exists():
            self._write_drawio_xml(self._read_state())

    def gui_eval_mode(self) -> str:
        return "live_shadow_required"

    @classmethod
    def from_evaluation_context(cls, tmp: Path, sandbox=None, mock: bool = False) -> "DrawioAdapter":
        del sandbox, mock
        return cls(tmp / "diagram.drawio.json", base_url=os.environ.get("DRAWIO_URL", "").strip())

    @property
    def source_path(self) -> Path:
        return self.diagram_path

    @property
    def diagram_xml_path(self) -> Path:
        return self.diagram_path.with_suffix("")

    def clone(self, new_path: Path) -> "DrawioAdapter":
        shutil.copy2(self.diagram_path, new_path)
        return DrawioAdapter(new_path, base_url=self.base_url)

    def get_context(self) -> dict[str, str]:
        return {
            "diagram_path": str(self.diagram_path),
            "drawio_xml_path": str(self.diagram_xml_path),
        }

    def reset_state(self) -> None:
        self.setup_state("default")

    def setup_state(self, initial_state: str) -> None:
        state = _default_state()
        if initial_state == "blank":
            state["shapes"] = []
            state["connectors"] = []
        self._write_state(state)
        self.clear_gui_shadow_state()

    def prepare_task(self, task: Any) -> None:
        """Seed generated alias ids before deterministic rollout.

        Task generation often renames existing draw.io node ids in the action
        stream without also materializing those ids in the initial diagram. For
        rollout we preserve the generated surface by cloning the source element
        to the generated alias, then the ground-truth action can update it.
        """
        self.setup_state(getattr(task, "initial_state", "default") or "default")
        taskgen = getattr(task, "_taskgen", {}) or {}
        replacements = taskgen.get("replacements") if isinstance(taskgen, dict) else {}
        if not isinstance(replacements, dict):
            return

        state = self._read_state()
        shapes_by_id = {str(shape.get("id")): shape for shape in state["shapes"]}
        connectors_by_id = {str(connector.get("id")): connector for connector in state["connectors"]}
        changed = False

        for old, new in replacements.items():
            old_id = str(old)
            new_id = str(new)
            if old_id == new_id or not new_id:
                continue
            if old_id in shapes_by_id and new_id not in shapes_by_id:
                cloned = deepcopy(shapes_by_id[old_id])
                cloned["id"] = new_id
                state["shapes"].append(cloned)
                shapes_by_id[new_id] = cloned
                changed = True

        for old, new in replacements.items():
            old_id = str(old)
            new_id = str(new)
            if old_id == new_id or not new_id:
                continue
            if old_id in connectors_by_id and new_id not in connectors_by_id:
                cloned = deepcopy(connectors_by_id[old_id])
                cloned["id"] = new_id
                cloned["source"] = str(replacements.get(cloned.get("source"), cloned.get("source", "")))
                cloned["target"] = str(replacements.get(cloned.get("target"), cloned.get("target", "")))
                state["connectors"].append(cloned)
                connectors_by_id[new_id] = cloned
                changed = True

        if changed:
            self._write_state(state)

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types and action.target == "diagram"

    def get_gui_session_spec(self) -> GUISessionSpec | None:
        if not self.base_url:
            return None
        return GUISessionSpec(
            surface_type="browser",
            browser_url=self._live_editor_url(),
            browser_navigation_mode="current_page",
            window_title_pattern=r".*",
            window_class_pattern=r"chromium|Chromium|chrome|Google-chrome",
            min_width=1000,
            min_height=700,
            post_launch_delay_s=5.0,
            backend_ready_probe=self._probe_backend_ready,
        )

    def _probe_backend_ready(self) -> None:
        from asil.gui_agent.session import GUISessionStartupError

        try:
            response = requests.get(self.base_url, timeout=10)
        except requests.RequestException as exc:
            raise GUISessionStartupError("backend_unready", f"draw.io backend is not reachable at {self.base_url}.") from exc
        if response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"draw.io backend returned HTTP {response.status_code} for {self.base_url}.",
            )
        shell_url = self._editor_shell_url()
        try:
            shell_response = requests.get(shell_url, timeout=10)
        except requests.RequestException as exc:
            raise GUISessionStartupError(
                "backend_unready",
                f"draw.io shell page is not reachable at {shell_url}.",
            ) from exc
        if shell_response.status_code >= 500:
            raise GUISessionStartupError(
                "backend_unready",
                f"draw.io shell page returned HTTP {shell_response.status_code} for {shell_url}.",
            )

    def _prime_browser_session(self, session) -> None:
        from asil.gui_agent.session import _assert_browser_page_ready, navigate_browser_target

        page = session.browser_page
        target_url = self._live_editor_url()
        if not str(page.url).startswith(target_url):
            navigate_browser_target(session, target_url, timeout_ms=60_000)
            page = session.browser_page
        page.wait_for_load_state("domcontentloaded", timeout=30_000)
        _assert_browser_page_ready(
            session,
            required_selectors=(".geMenubarContainer",),
            app_name="draw.io",
            timeout_ms=90_000,
        )
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        _assert_browser_page_ready(
            session,
            required_selectors=(".geMenubarContainer",),
            app_name="draw.io",
            timeout_ms=90_000,
        )

    def _probe_ui_ready(self, session) -> None:
        from asil.gui_agent.session import _assert_browser_page_ready

        _assert_browser_page_ready(
            session,
            required_selectors=(
                ".geMenubarContainer",
                ".geToolbarContainer",
                ".geDiagramContainer",
            ),
            app_name="draw.io",
            timeout_ms=90_000,
        )

    def observe(self) -> Observation:
        state = self._shadow_or_state()
        shapes_by_id = {shape["id"]: shape for shape in state["shapes"]}
        elements: list[Element] = [
            Element(
                id="canvas",
                type="canvas",
                label="Diagram Canvas",
                value=state["canvas"],
                editable=True,
                actions=["resize_canvas"],
            )
        ]
        for shape in state["shapes"]:
            elements.append(
                Element(
                    id=f"shape:{shape['id']}",
                    type="shape",
                    label=shape["label"],
                    value=shape,
                    editable=True,
                    actions=["update_shape", "delete_element"],
                )
            )
        for connector in state["connectors"]:
            source_shape = shapes_by_id.get(connector["source"], {})
            target_shape = shapes_by_id.get(connector["target"], {})
            connector_value = dict(connector)
            connector_value["source_label"] = source_shape.get("label", connector["source"])
            connector_value["target_label"] = target_shape.get("label", connector["target"])
            elements.append(
                Element(
                    id=f"connector:{connector['id']}",
                    type="connector",
                    label=connector.get("label", connector["id"]),
                    value=connector_value,
                    editable=True,
                    actions=["update_connector", "delete_element"],
                )
            )

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "diagram_canvas",
                "active_document": self.diagram_path.name,
                "document_path": str(self.diagram_path),
            },
            data_summary=f"Diagram with {len(state['shapes'])} shapes and {len(state['connectors'])} connectors",
        )

    def execute(self, action: Action) -> Observation:
        if not self.validate_action(action):
            raise ValueError(f"Unsupported draw.io action: {action}")

        state = self._read_state()
        for operation in action.params.get("operations", []):
            op = operation.get("action")
            if op == "add_shape":
                state["shapes"].append(
                    {
                        "id": operation["id"],
                        "label": operation["label"],
                        "shape_kind": operation.get("shape_kind", "rectangle"),
                        "x": int(operation.get("x", 0)),
                        "y": int(operation.get("y", 0)),
                        "width": int(operation.get("width", 120)),
                        "height": int(operation.get("height", 70)),
                        "fill": operation.get("fill", "#ffffff"),
                        "stroke": operation.get("stroke", "#334155"),
                    }
                )
            elif op == "update_shape":
                shape = self._find_by_id(state["shapes"], operation["id"])
                shape.update(operation.get("changes", {}))
            elif op == "add_connector":
                state["connectors"].append(
                    {
                        "id": operation["id"],
                        "source": operation["source"],
                        "target": operation["target"],
                        "label": operation.get("label", ""),
                        "stroke": operation.get("stroke", "#64748b"),
                    }
                )
            elif op == "update_connector":
                connector = self._find_by_id(state["connectors"], operation["id"])
                connector.update(operation.get("changes", {}))
            elif op == "delete_element":
                state["shapes"] = [shape for shape in state["shapes"] if shape["id"] != operation["id"]]
                state["connectors"] = [connector for connector in state["connectors"] if connector["id"] != operation["id"]]
            elif op == "resize_canvas":
                state["canvas"]["width"] = int(operation["width"])
                state["canvas"]["height"] = int(operation["height"])
            else:
                raise ValueError(f"Unsupported draw.io operation: {op}")

        self._write_state(state)
        return self.observe()

    def describe_rendering(self) -> RenderArtifact:
        if self.base_url:
            return RenderArtifact(
                filename="",
                kind="web_page_capture",
                backend="playwright+chromium",
                actual_page=True,
                description="Live draw.io webpage capture",
            )
        return RenderArtifact(
            filename="",
            kind="state_render",
            backend="wkhtmltoimage+html",
            actual_page=False,
            description="Synthetic draw.io diagram render",
        )

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        output = Path(output_path) if output_path else self.diagram_path.with_suffix(".png")
        if self.base_url:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    return capture_url_to_png(
                        self._live_editor_url(),
                        output,
                        backend="playwright",
                        full_page=False,
                        initial_wait_ms=12_000,
                        timeout_ms=90_000,
                        wait_for_selectors=[
                            ".geMenubarContainer",
                            ".geToolbarContainer",
                            ".geDiagramContainer",
                        ],
                    )
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(8)
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
        return capture_html_to_png(self._diagram_html(), output)

    def _live_editor_url(self) -> str:
        xml_payload = quote(self._state_to_drawio_xml(self._read_state()), safe="")
        return f"{self.base_url}/?offline=1&stealth=1#R{xml_payload}"

    def _editor_shell_url(self) -> str:
        return f"{self.base_url}/?offline=1&stealth=1"

    def _diagram_html(self) -> str:
        state = self._read_state()
        canvas = state["canvas"]
        shape_html = []
        connector_html = []
        centers = {
            shape["id"]: (shape["x"] + shape["width"] / 2, shape["y"] + shape["height"] / 2)
            for shape in state["shapes"]
        }
        for connector in state["connectors"]:
            src = centers.get(connector["source"])
            dst = centers.get(connector["target"])
            if not src or not dst:
                continue
            left = min(src[0], dst[0])
            top = min(src[1], dst[1])
            width = max(abs(dst[0] - src[0]), 2)
            connector_html.append(
                f"<div style='position:absolute;left:{left}px;top:{top}px;width:{width}px;height:2px;background:{connector['stroke']};'></div>"
            )
        for shape in state["shapes"]:
            border_radius = "999px" if shape["shape_kind"] == "ellipse" else "12px"
            shape_html.append(
                "<div style='position:absolute;"
                f"left:{shape['x']}px;top:{shape['y']}px;width:{shape['width']}px;height:{shape['height']}px;"
                f"background:{shape['fill']};border:2px solid {shape['stroke']};border-radius:{border_radius};"
                "display:flex;align-items:center;justify-content:center;font-weight:600;color:#0f172a;'>"
                f"{shape['label']}</div>"
            )
        body = (
            "<h1>draw.io Diagram</h1>"
            "<p>Workflow canvas preview</p>"
            f"<div class='panel' style='position:relative;width:{canvas['width']}px;height:{canvas['height']}px;"
            f"background:{canvas['background']};overflow:hidden;padding:0;'>"
            + "".join(connector_html)
            + "".join(shape_html)
            + "</div>"
        )
        return html_page("draw.io Diagram", body)

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.diagram_path.read_text(encoding="utf-8"))

    def _shadow_or_state(self) -> dict[str, Any]:
        shadow = self._get_gui_shadow_state()
        if shadow and {"canvas", "shapes", "connectors"} <= set(shadow):
            return shadow
        return self._read_state()

    def _write_state(self, state: dict[str, Any]) -> None:
        self.diagram_path.parent.mkdir(parents=True, exist_ok=True)
        self.diagram_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self._write_drawio_xml(state)

    def sync_from_gui(self, session=None) -> None:
        page = getattr(session, "browser_page", None) if session is not None else None
        live_xml = self._read_live_editor_xml(page)
        state = self._state_from_drawio_xml(live_xml) if live_xml is not None else None
        live_url = str(getattr(page, "url", "") or "")
        if state is None:
            state = self._state_from_live_editor_url(live_url)
        if state is not None:
            self._set_gui_shadow_state(state)

    @staticmethod
    def _read_live_editor_xml(page: Any | None) -> str | None:
        evaluate = getattr(page, "evaluate", None)
        if not callable(evaluate):
            return None
        try:
            xml_payload = evaluate(_LIVE_EDITOR_XML_SCRIPT)
        except Exception:
            return None
        if isinstance(xml_payload, str) and xml_payload.strip():
            return xml_payload
        return None

    def _write_drawio_xml(self, state: dict[str, Any]) -> None:
        self.diagram_xml_path.write_text(self._state_to_drawio_xml(state), encoding="utf-8")

    def _state_to_drawio_xml(self, state: dict[str, Any]) -> str:
        cells = [
            '<mxCell id="0"/>',
            '<mxCell id="1" parent="0"/>',
        ]

        for shape in state["shapes"]:
            style_parts = [
                "whiteSpace=wrap",
                "html=1",
                f"fillColor={shape.get('fill', '#ffffff')}",
                f"strokeColor={shape.get('stroke', '#334155')}",
            ]
            if shape.get("shape_kind") == "ellipse":
                style_parts.append("ellipse")
            else:
                style_parts.append("rounded=1")
            value = escape(shape.get("label", ""))
            cells.append(
                f'<mxCell id="{escape(shape["id"])}" value="{value}" style="{";".join(style_parts)};" '
                'vertex="1" parent="1">'
                f'<mxGeometry x="{int(shape.get("x", 0))}" y="{int(shape.get("y", 0))}" '
                f'width="{int(shape.get("width", 120))}" height="{int(shape.get("height", 70))}" as="geometry"/>'
                "</mxCell>"
            )

        for connector in state["connectors"]:
            value = escape(connector.get("label", ""))
            cells.append(
                f'<mxCell id="{escape(connector["id"])}" value="{value}" '
                f'style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={connector.get("stroke", "#64748b")};" '
                f'edge="1" parent="1" source="{escape(connector["source"])}" target="{escape(connector["target"])}">'
                '<mxGeometry relative="1" as="geometry"/>'
                "</mxCell>"
            )

        return (
            '<mxfile host="app.diagrams.net" version="24.7.17">'
            '<diagram id="asil-page-1" name="Page-1">'
            '<mxGraphModel dx="1280" dy="720" grid="1" gridSize="10" guides="1" tooltips="1" '
            'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" '
            'math="0" shadow="0"><root>'
            + "".join(cells)
            + "</root></mxGraphModel></diagram></mxfile>"
        )

    def _state_from_live_editor_url(self, url: str) -> dict[str, Any] | None:
        if "#R" not in url:
            return None
        try:
            xml_payload = unquote(url.split("#R", 1)[1])
        except Exception:
            return None
        return self._state_from_drawio_xml(xml_payload)

    def _state_from_drawio_xml(self, xml_payload: str) -> dict[str, Any] | None:
        try:
            root = ET.fromstring(xml_payload)
        except Exception:
            return None

        mx_root = root.find(".//root")
        if mx_root is None:
            return None
        state = {
            "canvas": {"width": 960, "height": 640, "background": "#f8fafc"},
            "shapes": [],
            "connectors": [],
        }
        for cell in mx_root.findall("mxCell"):
            cell_id = cell.attrib.get("id", "")
            if cell_id in {"0", "1"}:
                continue
            if cell.attrib.get("vertex") == "1":
                geometry = cell.find("mxGeometry")
                style = cell.attrib.get("style", "")
                shape_kind = "ellipse" if "ellipse" in style else "rectangle"
                fill = self._style_value(style, "fillColor", "#ffffff")
                stroke = self._style_value(style, "strokeColor", "#334155")
                state["shapes"].append(
                    {
                        "id": cell_id,
                        "label": cell.attrib.get("value", ""),
                        "shape_kind": shape_kind,
                        "x": int(float(geometry.attrib.get("x", "0"))) if geometry is not None else 0,
                        "y": int(float(geometry.attrib.get("y", "0"))) if geometry is not None else 0,
                        "width": int(float(geometry.attrib.get("width", "120"))) if geometry is not None else 120,
                        "height": int(float(geometry.attrib.get("height", "70"))) if geometry is not None else 70,
                        "fill": fill,
                        "stroke": stroke,
                    }
                )
            elif cell.attrib.get("edge") == "1":
                style = cell.attrib.get("style", "")
                state["connectors"].append(
                    {
                        "id": cell_id,
                        "source": cell.attrib.get("source", ""),
                        "target": cell.attrib.get("target", ""),
                        "label": cell.attrib.get("value", ""),
                        "stroke": self._style_value(style, "strokeColor", "#64748b"),
                    }
                )
        return state

    @staticmethod
    def _style_value(style: str, key: str, default: str) -> str:
        prefix = f"{key}="
        for fragment in style.split(";"):
            if fragment.startswith(prefix):
                return fragment.split("=", 1)[1] or default
        return default

    @staticmethod
    def _find_by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
        for item in items:
            if item["id"] == item_id:
                return item
        raise KeyError(item_id)
