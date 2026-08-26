"""ASIL adapter for Inkscape — Pattern A (SVG file manipulation)."""

from __future__ import annotations
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from lxml import etree

from asil.adapter import ASILAdapter
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    capture_window_to_png,
    ensure_user_access,
    launch_gui_process,
    terminate_process,
)

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
XLINK_NS = "http://www.w3.org/1999/xlink"
NS = {"svg": SVG_NS, "inkscape": INK_NS}

_SHAPE_TAGS = frozenset(
    f"{{{SVG_NS}}}{t}"
    for t in ("rect", "circle", "ellipse", "line", "path", "polygon", "polyline", "text", "image")
)

_RECT_ATTRS = ("x", "y", "width", "height", "rx", "ry")
_CIRCLE_ATTRS = ("cx", "cy", "r")
_ELLIPSE_ATTRS = ("cx", "cy", "rx", "ry")
_LINE_ATTRS = ("x1", "y1", "x2", "y2")
_COMMON_ATTRS = ("style", "transform")
_INKSCAPE_WINDOW_PATTERN = r".*Inkscape|.* - Inkscape"


class InkscapeAdapter(ASILAdapter):
    app_name = "Inkscape"
    supported_action_types = ["modify_file", "set_value"]

    def __init__(self, svg_path: str | Path) -> None:
        self.svg_path = Path(svg_path)
        self._tree: etree._ElementTree | None = None

    @property
    def source_path(self) -> Path:
        return self.svg_path

    def clone(self, new_path: Path) -> "InkscapeAdapter":
        shutil.copy2(self.svg_path, new_path)
        return InkscapeAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        return {"svg_path": str(self.svg_path)}

    def prepare_task(self, task: Any) -> None:
        replacements = (getattr(task, "_taskgen", {}) or {}).get("replacements") or {}
        if not isinstance(replacements, dict):
            return
        root = self._load()
        changed = False
        for old, new in replacements.items():
            old_id = str(old)
            new_id = str(new)
            if not old_id or not new_id or old_id == new_id or any(ch.isspace() for ch in new_id):
                continue
            if root.xpath(f"//*[@id={old_id!r}]", namespaces=NS) and not root.xpath(f"//*[@id={new_id!r}]", namespaces=NS):
                source = root.xpath(f"//*[@id={old_id!r}]", namespaces=NS)[0]
                cloned = deepcopy(source)
                cloned.set("id", new_id)
                source.getparent().append(cloned)
                changed = True
        if changed:
            self._save()

    def _load(self) -> etree._Element:
        self._tree = etree.parse(str(self.svg_path))
        return self._tree.getroot()

    def _save(self) -> None:
        assert self._tree is not None
        self._tree.write(str(self.svg_path), xml_declaration=True, encoding="utf-8")

    @staticmethod
    def _local_tag(elem: etree._Element) -> str:
        return etree.QName(elem.tag).localname

    def _extract_element(self, elem: etree._Element) -> Element | None:
        tag = self._local_tag(elem)
        elem_id = elem.get("id")
        if not elem_id:
            return None

        value: dict = {}
        metadata: dict = {}
        if tag == "rect":
            for a in _RECT_ATTRS:
                v = elem.get(a)
                if v is not None:
                    value[a] = v
            # SVG uses the specified radius for both axes when exactly one of
            # rx/ry is present. Inkscape commonly serializes that compact form
            # even though both radius controls show the same effective value.
            if "rx" in value and "ry" not in value:
                value["ry"] = value["rx"]
            elif "ry" in value and "rx" not in value:
                value["rx"] = value["ry"]
        elif tag == "circle":
            for a in _CIRCLE_ATTRS:
                v = elem.get(a)
                if v is not None:
                    value[a] = v
        elif tag == "ellipse":
            for a in _ELLIPSE_ATTRS:
                v = elem.get(a)
                if v is not None:
                    value[a] = v
        elif tag == "line":
            for a in _LINE_ATTRS:
                v = elem.get(a)
                if v is not None:
                    value[a] = v
        elif tag == "text":
            value["text_content"] = "".join(elem.itertext())
            for a in ("x", "y"):
                v = elem.get(a)
                if v is not None:
                    value[a] = v
        elif tag == "path":
            value["d"] = elem.get("d", "")
        elif tag == "g":
            child_ids = [child.get("id") for child in elem if child.get("id")]
            value["child_count"] = len(child_ids)
            metadata["child_ids"] = child_ids
        elif tag == "image":
            value["href"] = elem.get(f"{{{XLINK_NS}}}href", elem.get("href", ""))

        for a in _COMMON_ATTRS:
            v = elem.get(a)
            if v:
                value[a] = v

        parent = elem.getparent()
        if parent is not None and parent.get("id"):
            metadata["parent_id"] = parent.get("id")

        label = elem.get(f"{{{INK_NS}}}label", tag)

        elem_type = "group" if tag == "g" else tag

        return Element(
            id=elem_id,
            type=elem_type,
            label=label,
            value=value,
            editable=True,
            actions=["set_attribute", "move", "resize", "delete", "set_style"],
            metadata=metadata,
        )

    def observe(self) -> Observation:
        root = self._load()
        elements: list[Element] = []

        for elem in root.iter():
            local_tag = self._local_tag(elem)
            is_non_layer_group = (
                local_tag == "g"
                and elem.get(f"{{{INK_NS}}}groupmode") != "layer"
            )
            if elem.tag in _SHAPE_TAGS or is_non_layer_group:
                parsed = self._extract_element(elem)
                if parsed:
                    elements.append(parsed)

        canvas_w = root.get("width", "0").replace("px", "")
        canvas_h = root.get("height", "0").replace("px", "")

        layers = root.xpath("//svg:g[@inkscape:groupmode='layer']", namespaces=NS)
        layer_names = [l.get(f"{{{INK_NS}}}label", l.get("id", "")) for l in layers]

        # Expose layers as elements so element_exists/element_value checks work
        for layer in layers:
            layer_id = layer.get("id")
            if layer_id:
                elements.append(Element(
                    id=layer_id,
                    type="layer",
                    label=layer.get(f"{{{INK_NS}}}label", layer_id),
                    value={"label": layer.get(f"{{{INK_NS}}}label", layer_id)},
                    editable=True,
                    actions=["delete"],
                ))

        return self._build_observation(
            source="file_parse",
            elements=elements,
            app_state={
                "current_view": "canvas",
                "active_document": self.svg_path.name,
                "document_path": str(self.svg_path),
            },
            environment={
                "system": {
                    "canvas_width": float(canvas_w or 0),
                    "canvas_height": float(canvas_h or 0),
                },
            },
            data_summary=f"SVG with {len(elements)} elements across {len(layer_names)} layers: {layer_names}",
        )

    # Prefix → namespace URI map for attribute expansion
    _ATTR_NS = {
        "inkscape": INK_NS,
        "xlink": XLINK_NS,
        "xml": "http://www.w3.org/XML/1998/namespace",
        "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd",
        "dc": "http://purl.org/dc/elements/1.1/",
        "cc": "http://creativecommons.org/ns#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }

    @staticmethod
    def _expand_attr(name: str) -> str:
        """Expand 'prefix:local' attribute names to '{uri}local' for lxml."""
        if ":" in name:
            prefix, local = name.split(":", 1)
            uri = InkscapeAdapter._ATTR_NS.get(prefix)
            if uri:
                return f"{{{uri}}}{local}"
        return name

    def execute(self, action: Action) -> Observation:
        root = self._load()

        operations = action.params.get("operations", [])
        for op in operations:
            op_action = op.get("action", "set_attribute")

            if op_action == "add_element":
                parents = root.xpath(op["parent_xpath"], namespaces=NS)
                if parents:
                    tag = op["tag"]
                    raw_attrs = dict(op.get("attributes", {}))
                    # Pull out text_content before passing to lxml (it's not an XML attribute)
                    text_content = raw_attrs.pop("text_content", None)
                    # Expand namespace-prefixed attribute names
                    attrs = {self._expand_attr(k): str(v) for k, v in raw_attrs.items()}
                    new_elem = etree.SubElement(parents[0], f"{{{SVG_NS}}}{tag}", attrs)
                    if text_content is not None:
                        new_elem.text = str(text_content)

            elif op_action == "delete":
                targets = root.xpath(op["xpath"], namespaces=NS)
                for t in targets:
                    t.getparent().remove(t)

            else:
                # default: set_attribute
                xpath = op.get("xpath", "")
                attribute = op.get("attribute", "")
                value = op.get("value", "")
                if xpath and attribute:
                    targets = root.xpath(xpath, namespaces=NS)
                    if attribute == "text_content":
                        # Set SVG text node content
                        for t in targets:
                            t.text = str(value)
                    else:
                        attr_key = self._expand_attr(attribute)
                        for t in targets:
                            t.set(attr_key, str(value))

        self._save()
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real Inkscape window showing the current document",
        )

    def render_to_png(self, output_path: str | Path | None = None, dpi: int = 96) -> Path:
        """Capture the real Inkscape window showing the current SVG document.

        Raises RuntimeError if Inkscape is not installed.
        Returns the output PNG path.
        """
        inkscape = shutil.which("inkscape")
        if inkscape is None:
            raise RuntimeError(
                "Inkscape is not installed. Install it to use render_to_png()."
            )

        out = Path(output_path) if output_path else self.svg_path.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)

        ensure_user_access(self.svg_path.parent, run_as_user="asilgui")
        proc = launch_gui_process(
            [inkscape, str(self.svg_path)],
            extra_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
            run_as_user="asilgui",
        )
        try:
            capture_metadata = {"capture_complete": True}
            capture_window_to_png(
                out,
                title_pattern=_INKSCAPE_WINDOW_PATTERN,
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
        return out
