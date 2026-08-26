"""ASIL adapter for GIMP — lightweight PNG-backed mock document."""

from __future__ import annotations

import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, PngImagePlugin

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

_STATE_KEY = "asil_gimp_state"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _XcfLayerSnapshot:
    """One layer read from the XCF that the GUI actually edited."""

    name: str
    pixels: Image.Image
    offset_x: int
    offset_y: int
    visible: bool


class GimpAdapter(ASILAdapter):
    app_name = "GIMP"
    supported_action_types = ["invoke_function"]

    def __init__(self, image_path: str | Path) -> None:
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            self.setup_state("default")

    @property
    def gui_project_path(self) -> Path:
        return self.image_path.with_suffix(".xcf")

    @property
    def gui_composite_path(self) -> Path:
        return self.image_path.with_name(f"{self.image_path.stem}.gui.png")

    @classmethod
    def from_evaluation_context(
        cls,
        tmp: str | Path,
        sandbox=None,
        mock: bool = False,
    ) -> "GimpAdapter":
        del sandbox, mock
        return cls(Path(tmp) / "gimp_document.png")

    @property
    def source_path(self) -> Path:
        return self.image_path

    def clone(self, new_path: Path) -> "GimpAdapter":
        shutil.copy2(self.image_path, new_path)
        return GimpAdapter(new_path)

    def get_context(self) -> dict[str, str]:
        image = str(self.image_path)
        return {
            "image_path": image,
            "document_path": image,
        }

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real GIMP window showing the current document",
        )

    def get_gui_session_spec(self) -> GUISessionSpec:
        state = self._load_state()
        self._write_xcf_project(state)
        ensure_user_access(self.gui_project_path.parent, run_as_user="asilgui")
        return GUISessionSpec(
            surface_type="desktop",
            launch_command=(
                "gimp",
                "--new-instance",
                "--no-splash",
                str(self.gui_project_path),
            ),
            window_title_pattern="GIMP|GNU Image Manipulation Program",
            run_as_user="asilgui",
            persist_shortcuts=("ctrl+s",),
        )

    def reset_state(self) -> None:
        self.setup_state("default")

    def setup_state(self, initial_state: str) -> None:
        self.clear_gui_shadow_state()
        self._write_state(self._initial_state(initial_state or "default"))

    def prepare_task(self, task: Any) -> None:
        self.setup_state(getattr(task, "initial_state", "default") or "default")
        replacements = (getattr(task, "_taskgen", {}) or {}).get("replacements") or {}
        if not isinstance(replacements, dict):
            return
        state = self._load_state()
        layers_by_id = {layer["id"]: layer for layer in state.get("layers", [])}
        changed = False
        for old, new in replacements.items():
            if not isinstance(old, str) or not isinstance(new, str):
                continue
            if old not in layers_by_id or new in layers_by_id:
                continue
            if any(ch.isspace() for ch in new) or "/" in new:
                continue
            cloned = dict(layers_by_id[old])
            cloned["id"] = new
            cloned["label"] = new.replace("_", " ").title()
            state["layers"].append(cloned)
            layers_by_id[new] = cloned
            changed = True
        if changed:
            self._write_state(state)

    def validate_action(self, action: Action) -> bool:
        return (
            action.action_type in self.supported_action_types
            and action.target == "gimp"
            and isinstance(action.params.get("operations"), list)
        )

    def observe(self) -> Observation:
        shadow_state = self._get_gui_shadow_state()
        state = shadow_state if shadow_state is not None else self._load_state()
        active_path = self.gui_project_path if shadow_state is not None else self.image_path
        document_path = self.gui_composite_path if shadow_state is not None else self.image_path
        canvas = state["canvas"]
        layers = state["layers"]
        elements = [
            Element(
                id="image",
                type="document",
                label=active_path.name,
                value={
                    "width": canvas["width"],
                    "height": canvas["height"],
                    "background": canvas["background"],
                    "layer_count": len(layers),
                },
                editable=True,
                actions=["crop_image", "resize_image", "export_png"],
            )
        ]
        for layer in layers:
            elements.append(
                Element(
                    id=layer["id"],
                    type="layer",
                    label=layer.get("label", layer["id"]),
                    value={
                        key: value
                        for key, value in layer.items()
                        if key not in {"id", "label"}
                    },
                    editable=True,
                    actions=[
                        "add_image_layer",
                        "update_layer",
                        "apply_filter",
                        "delete_layer",
                        "reorder_layer",
                    ],
                )
            )

        return self._build_observation(
            source="xcf_live" if shadow_state is not None else "png_metadata",
            elements=elements,
            app_state={
                "current_view": "canvas",
                "active_document": active_path.name,
                "document_path": str(document_path),
            },
            environment={
                "system": {
                    "canvas_width": canvas["width"],
                    "canvas_height": canvas["height"],
                }
            },
            data_summary=f"Raster image with {len(layers)} layers on a {canvas['width']}x{canvas['height']} canvas",
        )

    def sync_from_gui(self, session=None) -> None:
        del session
        canvas_size, snapshots = self._read_xcf_project()
        identity_by_name = {
            self._normalize_gui_layer_name(layer.get("label", layer["id"])): layer["id"]
            for layer in self._load_state().get("layers", [])
        }
        text_layer_names = {
            self._normalize_gui_layer_name(layer.get("label", layer["id"]))
            for layer in self._load_state().get("layers", [])
            if layer.get("kind") == "text"
        }

        layers: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for snapshot in snapshots:
            normalized_name = self._normalize_gui_layer_name(snapshot.name)
            layer_id = identity_by_name.get(normalized_name, normalized_name or "layer")
            if layer_id in used_ids:
                suffix = 2
                while f"{layer_id}_{suffix}" in used_ids:
                    suffix += 1
                layer_id = f"{layer_id}_{suffix}"
            used_ids.add(layer_id)
            layers.append(
                self._state_from_xcf_layer(
                    snapshot,
                    layer_id=layer_id,
                    text_candidate=(
                        normalized_name in text_layer_names
                        or any(
                            token in normalized_name.split("_")
                            for token in ("text", "title", "label", "watermark")
                        )
                    ),
                )
            )

        background = "#ffffff"
        canvas_width, canvas_height = canvas_size
        for layer in layers:
            if (
                layer.get("visible", True)
                and layer.get("x") == 0
                and layer.get("y") == 0
                and layer.get("width") == canvas_width
                and layer.get("height") == canvas_height
                and isinstance(layer.get("fill"), str)
            ):
                background = layer["fill"]
                break

        self._write_gui_composite((canvas_width, canvas_height), snapshots)
        self._set_gui_shadow_state(
            {
                "canvas": {
                    "width": canvas_width,
                    "height": canvas_height,
                    "background": background,
                },
                "layers": layers,
            }
        )

    def _write_gui_composite(
        self,
        canvas_size: tuple[int, int],
        snapshots: list[_XcfLayerSnapshot],
    ) -> None:
        """Persist a raster evaluation view composed only from live XCF pixels."""
        composite = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        for snapshot in reversed(snapshots):
            if not snapshot.visible:
                continue
            composite.alpha_composite(
                snapshot.pixels.convert("RGBA"),
                dest=(snapshot.offset_x, snapshot.offset_y),
            )
        self.gui_composite_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.gui_composite_path.with_name(f".{self.gui_composite_path.name}.tmp")
        composite.save(temporary, format="PNG")
        temporary.replace(self.gui_composite_path)

    def execute(self, action: Action) -> Observation:
        if not self.validate_action(action):
            raise ValueError(f"Unsupported GIMP action: {action}")

        state = self._load_state()
        for operation in action.params.get("operations", []):
            op_name = operation.get("action")
            if op_name == "add_layer":
                state["layers"].append(self._normalize_shape_layer(operation))
            elif op_name == "add_image_layer":
                state["layers"].append(self._normalize_image_layer(operation))
            elif op_name == "add_text_layer":
                state["layers"].append(self._normalize_text_layer(operation))
            elif op_name == "update_layer":
                layer = self._layer_by_id(state, operation["id"])
                layer.update(operation.get("changes", {}))
            elif op_name == "apply_filter":
                layer = self._layer_by_id(state, operation["id"])
                if layer.get("kind") != "image":
                    raise ValueError("apply_filter requires an image layer.")
                filters = dict(layer.get("filters") or {})
                filters.update(operation.get("filters", {}))
                layer["filters"] = filters
            elif op_name == "delete_layer":
                state["layers"] = [layer for layer in state["layers"] if layer["id"] != operation["id"]]
            elif op_name == "crop_image":
                self._crop_image(state, operation)
            elif op_name == "resize_image":
                self._resize_image(state, operation)
            elif op_name == "reorder_layer":
                self._reorder_layer(state, operation["id"], int(operation["index"]))
            else:
                raise ValueError(f"Unknown GIMP operation: {op_name}")

        self._write_state(state)
        return self.observe()

    def render_to_png(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        gimp_bin = shutil.which("gimp")
        if gimp_bin is None:
            raise RuntimeError("gimp is not installed.")

        ensure_user_access(self.image_path.parent, run_as_user="asilgui")
        proc = launch_gui_process(
            [gimp_bin, "--new-instance", "--no-splash", str(self.image_path)],
            extra_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
            run_as_user="asilgui",
        )
        try:
            send_keys_to_window(
                "GIMP|GNU Image Manipulation Program",
                ["Escape"],
                timeout=60.0,
                min_width=800,
                min_height=600,
            )
            capture_window_to_png(
                out,
                title_pattern="GIMP|GNU Image Manipulation Program",
                timeout=60.0,
                margin=12,
                settle_delay=6.0,
                min_width=800,
                min_height=600,
            )
        finally:
            terminate_process(proc)
        return out

    def _write_xcf_project(self, state: dict[str, Any]) -> None:
        gimp_console = shutil.which("gimp-console") or shutil.which("gimp")
        if gimp_console is None:
            raise RuntimeError("gimp-console is required to create the editable GIMP project.")

        canvas = state["canvas"]
        width = int(canvas["width"])
        height = int(canvas["height"])
        project_layers = list(state.get("layers", []))
        if not project_layers:
            project_layers = [
                {
                    "id": "background",
                    "label": "Background",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "fill": canvas.get("background", "#ffffff"),
                    "opacity": 1.0,
                    "visible": True,
                }
            ]

        self.gui_project_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".asil-gimp-layers-",
            dir=self.gui_project_path.parent,
        ) as temporary_dir:
            layer_paths: list[tuple[dict[str, Any], Path]] = []
            for index, layer in enumerate(project_layers):
                layer_path = Path(temporary_dir) / f"layer_{index:04d}.png"
                self._render_xcf_layer(layer, (width, height)).save(layer_path, format="PNG")
                layer_paths.append((layer, layer_path))

            commands = [
                f"(let* ((image (car (gimp-image-new {width} {height} RGB))) (layer -1))"
            ]
            for layer, layer_path in layer_paths:
                commands.append(
                    " (set! layer (car (gimp-file-load-layer RUN-NONINTERACTIVE image "
                    f"{self._scheme_string(layer_path)})))"
                )
                commands.append(
                    f" (gimp-item-set-name layer {self._scheme_string(layer.get('label', layer['id']))})"
                )
                commands.append(" (gimp-image-insert-layer image layer 0 0)")
                commands.append(" (gimp-layer-set-mode layer NORMAL-MODE)")
                if not layer.get("visible", True):
                    commands.append(" (gimp-item-set-visible layer FALSE)")
            commands.extend(
                [
                    " (gimp-file-save RUN-NONINTERACTIVE image layer "
                    f"{self._scheme_string(self.gui_project_path)} "
                    f"{self._scheme_string(self.gui_project_path)})",
                    " (gimp-image-delete image))",
                ]
            )
            completed = subprocess.run(
                [
                    gimp_console,
                    "-idf",
                    "-b",
                    "".join(commands),
                    "-b",
                    "(gimp-quit 0)",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            if completed.returncode != 0 or not self.gui_project_path.exists():
                detail = (completed.stderr or completed.stdout or "unknown GIMP error").strip()
                raise RuntimeError(f"Could not create editable GIMP XCF project: {detail}")

    def _read_xcf_project(self) -> tuple[tuple[int, int], list[_XcfLayerSnapshot]]:
        identify = shutil.which("identify")
        convert = shutil.which("convert")
        if identify is None or convert is None:
            raise RuntimeError("ImageMagick identify and convert are required to inspect the edited XCF.")
        self._wait_for_stable_xcf()
        canvas_size = self._read_xcf_canvas_size()
        completed = subprocess.run(
            [
                identify,
                "-format",
                "%s\t%l\t%w\t%h\t%X\t%Y\t%[compose]\n",
                str(self.gui_project_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )

        snapshots: list[_XcfLayerSnapshot] = []
        for line in completed.stdout.splitlines():
            fields = line.split("\t", 6)
            if len(fields) != 7:
                raise RuntimeError(f"Could not parse XCF layer metadata: {line!r}")
            scene, name, _width, _height, offset_x, offset_y, compose = fields
            rendered = subprocess.run(
                [convert, f"{self.gui_project_path}[{int(scene)}]", "png:-"],
                check=True,
                capture_output=True,
                timeout=20,
            )
            with Image.open(io.BytesIO(rendered.stdout)) as opened:
                pixels = opened.convert("RGBA").copy()
            snapshots.append(
                _XcfLayerSnapshot(
                    name=name or f"Layer {scene}",
                    pixels=pixels,
                    offset_x=int(offset_x),
                    offset_y=int(offset_y),
                    visible=compose.strip().casefold() != "none",
                )
            )
        if not snapshots:
            raise RuntimeError("The edited XCF contains no readable layers.")
        return canvas_size, snapshots

    def _wait_for_stable_xcf(self, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        previous: tuple[int, int] | None = None
        while time.monotonic() < deadline:
            try:
                stat = self.gui_project_path.stat()
            except FileNotFoundError:
                time.sleep(0.1)
                continue
            current = (stat.st_size, stat.st_mtime_ns)
            if stat.st_size > 0 and current == previous:
                return
            previous = current
            time.sleep(0.1)
        raise RuntimeError(f"GIMP XCF did not become stable: {self.gui_project_path}")

    def _read_xcf_canvas_size(self) -> tuple[int, int]:
        with self.gui_project_path.open("rb") as handle:
            header = handle.read(64)
        if not header.startswith(b"gimp xcf "):
            raise RuntimeError(f"Not a GIMP XCF project: {self.gui_project_path}")
        try:
            size_offset = header.index(b"\0", 9) + 1
            width, height = struct.unpack(">II", header[size_offset : size_offset + 8])
        except (ValueError, struct.error) as exc:
            raise RuntimeError(f"Could not read XCF canvas size: {self.gui_project_path}") from exc
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid XCF canvas size: {width}x{height}")
        return width, height

    def _state_from_xcf_layer(
        self,
        snapshot: _XcfLayerSnapshot,
        *,
        layer_id: str,
        text_candidate: bool,
    ) -> dict[str, Any]:
        pixels = snapshot.pixels.convert("RGBA")
        alpha = pixels.getchannel("A")
        local_bbox = alpha.getbbox()
        base: dict[str, Any] = {
            "id": layer_id,
            "label": snapshot.name,
            "visible": snapshot.visible,
            "blend_mode": "normal",
        }
        if local_bbox is None:
            base.update(
                {
                    "kind": "empty",
                    "x": snapshot.offset_x,
                    "y": snapshot.offset_y,
                    "width": 0,
                    "height": 0,
                    "opacity": 0.0,
                }
            )
            return base

        left, top, right, bottom = local_bbox
        cropped = pixels.crop(local_bbox)
        rgba_pixels = list(cropped.getdata())
        visible_pixels = [pixel for pixel in rgba_pixels if pixel[3] >= 16]
        pixel_count = max(1, cropped.width * cropped.height)
        coverage = len(visible_pixels) / pixel_count
        dominant = Counter(pixel[:3] for pixel in visible_pixels).most_common(1)
        dominant_color = (
            "#{:02x}{:02x}{:02x}".format(*dominant[0][0]) if dominant else "#000000"
        )
        max_alpha = max((pixel[3] for pixel in rgba_pixels), default=0)
        text = self._ocr_layer_text(pixels) if text_candidate else ""

        if text:
            kind = "text"
        elif coverage >= 0.94:
            kind = "rectangle"
        elif 0.65 <= coverage < 0.94:
            kind = "ellipse"
        else:
            kind = "image"

        base.update(
            {
                "kind": kind,
                "x": snapshot.offset_x + left,
                "y": snapshot.offset_y + top,
                "width": right - left,
                "height": bottom - top,
                "opacity": round(max_alpha / 255.0, 4),
            }
        )
        if kind in {"rectangle", "ellipse"}:
            base["fill"] = dominant_color
        elif kind == "text":
            base["text"] = text
            base["color"] = dominant_color
            base["font_size"] = max(8, int(round((bottom - top) * 0.86)))
        else:
            base["dominant_color"] = dominant_color
        return base

    @staticmethod
    def _normalize_gui_layer_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")

    @staticmethod
    def _scheme_string(value: str | Path) -> str:
        return json.dumps(str(value), ensure_ascii=True)

    def _ocr_layer_text(self, pixels: Image.Image) -> str:
        tesseract = shutil.which("tesseract")
        if tesseract is None:
            return ""
        rgba = pixels.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox is None:
            return ""
        cropped = rgba.crop(bbox)
        dominant = Counter(
            pixel[:3] for pixel in cropped.getdata() if pixel[3] >= 16
        ).most_common(1)
        if not dominant:
            return ""
        red, green, blue = dominant[0][0]
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        background = (0, 0, 0) if luminance >= 128 else (255, 255, 255)
        flattened = Image.new("RGB", cropped.size, background)
        flattened.paste(cropped.convert("RGB"), mask=cropped.getchannel("A"))
        enlarged = flattened.resize(
            (max(1, flattened.width * 4), max(1, flattened.height * 4)),
            Image.Resampling.LANCZOS,
        )
        thresholded = ImageOps.grayscale(enlarged).point(lambda value: 255 if value >= 128 else 0)
        buffer = io.BytesIO()
        thresholded.save(buffer, format="PNG")
        try:
            completed = subprocess.run(
                [tesseract, "stdin", "stdout", "--psm", "7"],
                input=buffer.getvalue(),
                check=False,
                capture_output=True,
                text=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode != 0:
            return ""
        return " ".join(completed.stdout.decode("utf-8", errors="replace").replace("\x0c", " ").split())

    def _render_xcf_layer(
        self,
        layer: dict[str, Any],
        canvas_size: tuple[int, int],
    ) -> Image.Image:
        if layer["kind"] == "image":
            return self._render_image_layer(layer, canvas_size)
        rendered = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(rendered)
        if layer["kind"] == "rectangle":
            draw.rectangle(
                [
                    int(layer["x"]),
                    int(layer["y"]),
                    int(layer["x"] + max(1, int(layer["width"])) - 1),
                    int(layer["y"] + max(1, int(layer["height"])) - 1),
                ],
                fill=self._hex_to_rgba(layer["fill"], layer.get("opacity", 1.0)),
            )
        elif layer["kind"] == "ellipse":
            draw.ellipse(
                [
                    int(layer["x"]),
                    int(layer["y"]),
                    int(layer["x"] + max(1, int(layer["width"])) - 1),
                    int(layer["y"] + max(1, int(layer["height"])) - 1),
                ],
                fill=self._hex_to_rgba(layer["fill"], layer.get("opacity", 1.0)),
            )
        elif layer["kind"] == "text":
            draw.text(
                (int(layer["x"]), int(layer["y"])),
                layer["text"],
                fill=self._hex_to_rgba(layer["color"], layer.get("opacity", 1.0)),
                font=self._font(int(layer.get("font_size", 24))),
            )
        else:
            raise ValueError(f"Unsupported GIMP XCF layer kind: {layer['kind']}")
        return rendered

    def _initial_state(self, initial_state: str) -> dict[str, Any]:
        if initial_state == "blank":
            return {
                "canvas": {"width": 800, "height": 600, "background": "#ffffff"},
                "layers": [],
            }

        return {
            "canvas": {"width": 800, "height": 600, "background": "#f5f1e8"},
            "layers": [
                {
                    "id": "hero_bg",
                    "label": "Hero Background",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 800,
                    "height": 600,
                    "fill": "#f5f1e8",
                    "opacity": 1.0,
                    "visible": True,
                    "blend_mode": "normal",
                },
                {
                    "id": "logo_block",
                    "label": "Logo Block",
                    "kind": "rectangle",
                    "x": 40,
                    "y": 40,
                    "width": 140,
                    "height": 90,
                    "fill": "#cc4444",
                    "opacity": 1.0,
                    "visible": True,
                    "blend_mode": "normal",
                },
                {
                    "id": "accent_circle",
                    "label": "Accent Circle",
                    "kind": "ellipse",
                    "x": 620,
                    "y": 80,
                    "width": 120,
                    "height": 120,
                    "fill": "#4da3ff",
                    "opacity": 0.9,
                    "visible": True,
                    "blend_mode": "normal",
                },
                {
                    "id": "watermark_text",
                    "label": "Watermark",
                    "kind": "text",
                    "x": 560,
                    "y": 500,
                    "text": "DRAFT",
                    "font_size": 48,
                    "color": "#b8b8b8",
                    "opacity": 0.45,
                    "visible": True,
                    "blend_mode": "normal",
                },
            ],
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.image_path.exists():
            self.setup_state("default")

        with Image.open(self.image_path) as image:
            payload = image.text.get(_STATE_KEY) or image.info.get(_STATE_KEY)
            if payload:
                return json.loads(payload)
            width, height = image.size
        return {
            "canvas": {"width": width, "height": height, "background": "#ffffff"},
            "layers": [],
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self._render_state(state)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(_STATE_KEY, json.dumps(state, ensure_ascii=True, sort_keys=True))
        rendered.save(self.image_path, format="PNG", pnginfo=pnginfo)

    def _render_state(self, state: dict[str, Any]) -> Image.Image:
        canvas = state["canvas"]
        image = Image.new(
            "RGBA",
            (int(canvas["width"]), int(canvas["height"])),
            self._hex_to_rgba(canvas["background"], 1.0),
        )
        for layer in state["layers"]:
            if not layer.get("visible", True):
                continue
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            if layer["kind"] == "rectangle":
                draw.rectangle(
                    [
                        int(layer["x"]),
                        int(layer["y"]),
                        int(layer["x"] + layer["width"]),
                        int(layer["y"] + layer["height"]),
                    ],
                    fill=self._hex_to_rgba(layer["fill"], layer.get("opacity", 1.0)),
                )
            elif layer["kind"] == "ellipse":
                draw.ellipse(
                    [
                        int(layer["x"]),
                        int(layer["y"]),
                        int(layer["x"] + layer["width"]),
                        int(layer["y"] + layer["height"]),
                    ],
                    fill=self._hex_to_rgba(layer["fill"], layer.get("opacity", 1.0)),
                )
            elif layer["kind"] == "text":
                draw.text(
                    (int(layer["x"]), int(layer["y"])),
                    layer["text"],
                    fill=self._hex_to_rgba(layer["color"], layer.get("opacity", 1.0)),
                    font=self._font(int(layer.get("font_size", 24))),
                )
            elif layer["kind"] == "image":
                overlay = self._render_image_layer(layer, image.size)
            image = Image.alpha_composite(image, overlay)
        return image.convert("RGBA")

    @classmethod
    def _render_image_layer(cls, layer: dict[str, Any], canvas_size: tuple[int, int]) -> Image.Image:
        with Image.open(cls._resolve_asset_path(layer["asset_path"])) as opened:
            source = opened.convert("RGBA")
        crop = layer.get("crop")
        if crop:
            left = int(crop.get("x", 0))
            top = int(crop.get("y", 0))
            right = left + int(crop.get("width", source.width - left))
            bottom = top + int(crop.get("height", source.height - top))
            source = source.crop((left, top, right, bottom))

        width = int(layer.get("width", source.width))
        height = int(layer.get("height", source.height))
        source = source.resize((width, height), Image.Resampling.LANCZOS)
        source = cls._apply_image_filters(source, layer.get("filters") or {})

        opacity = float(layer.get("opacity", 1.0))
        if opacity < 1.0:
            alpha = source.getchannel("A").point(lambda px: int(px * max(0.0, min(1.0, opacity))))
            source.putalpha(alpha)

        overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        overlay.alpha_composite(source, (int(layer.get("x", 0)), int(layer.get("y", 0))))
        return overlay

    @staticmethod
    def _apply_image_filters(image: Image.Image, filters: dict[str, Any]) -> Image.Image:
        result = image
        if filters.get("grayscale"):
            result = ImageOps.grayscale(result).convert("RGBA")
        if "brightness" in filters:
            result = ImageEnhance.Brightness(result).enhance(float(filters["brightness"]))
        if "contrast" in filters:
            result = ImageEnhance.Contrast(result).enhance(float(filters["contrast"]))
        if "saturation" in filters:
            result = ImageEnhance.Color(result).enhance(float(filters["saturation"]))
        if "blur_radius" in filters:
            result = result.filter(ImageFilter.GaussianBlur(float(filters["blur_radius"])))
        if filters.get("sharpen"):
            result = result.filter(ImageFilter.SHARPEN)
        return result.convert("RGBA")

    @staticmethod
    def _resolve_asset_path(asset_path: str) -> Path:
        path = Path(asset_path)
        if path.is_absolute():
            return path
        candidates = [
            _PROJECT_ROOT / path,
            _PROJECT_ROOT / "evaluation_examples" / "assets" / "realwork_images" / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"GIMP image asset not found: {asset_path}")

    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", max(8, size))
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def _hex_to_rgba(color: str, opacity: float) -> tuple[int, int, int, int]:
        value = color.lstrip("#")
        if len(value) != 6:
            raise ValueError(f"Expected 6-digit hex color, got: {color}")
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
        alpha = max(0, min(255, round(float(opacity) * 255)))
        return (red, green, blue, alpha)

    @staticmethod
    def _normalize_shape_layer(spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": spec["id"],
            "label": spec.get("label", spec["id"]),
            "kind": spec.get("kind", "rectangle"),
            "x": int(spec.get("x", 0)),
            "y": int(spec.get("y", 0)),
            "width": int(spec.get("width", 100)),
            "height": int(spec.get("height", 100)),
            "fill": spec.get("fill", "#000000"),
            "opacity": float(spec.get("opacity", 1.0)),
            "visible": bool(spec.get("visible", True)),
            "blend_mode": spec.get("blend_mode", "normal"),
        }

    @staticmethod
    def _normalize_image_layer(spec: dict[str, Any]) -> dict[str, Any]:
        raw_asset_path = spec.get("asset_path") or spec.get("path") or spec.get("asset")
        if not raw_asset_path:
            raise ValueError("add_image_layer requires asset_path.")
        asset_path = str(raw_asset_path)
        return {
            "id": spec["id"],
            "label": spec.get("label", spec["id"]),
            "kind": "image",
            "asset_path": asset_path,
            "x": int(spec.get("x", 0)),
            "y": int(spec.get("y", 0)),
            "width": int(spec.get("width", 400)),
            "height": int(spec.get("height", 300)),
            "crop": spec.get("crop"),
            "filters": spec.get("filters", {}),
            "opacity": float(spec.get("opacity", 1.0)),
            "visible": bool(spec.get("visible", True)),
            "blend_mode": spec.get("blend_mode", "normal"),
        }

    @staticmethod
    def _normalize_text_layer(spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": spec["id"],
            "label": spec.get("label", spec["id"]),
            "kind": "text",
            "x": int(spec.get("x", 0)),
            "y": int(spec.get("y", 0)),
            "text": spec.get("text", ""),
            "font_size": int(spec.get("font_size", 24)),
            "color": spec.get("color", "#000000"),
            "opacity": float(spec.get("opacity", 1.0)),
            "visible": bool(spec.get("visible", True)),
            "blend_mode": spec.get("blend_mode", "normal"),
        }

    @staticmethod
    def _layer_by_id(state: dict[str, Any], layer_id: str) -> dict[str, Any]:
        for layer in state["layers"]:
            if layer["id"] == layer_id:
                return layer
        raise KeyError(f"Unknown layer id: {layer_id}")

    @staticmethod
    def _crop_image(state: dict[str, Any], spec: dict[str, Any]) -> None:
        crop_x = int(spec.get("x", 0))
        crop_y = int(spec.get("y", 0))
        state["canvas"]["width"] = int(spec["width"])
        state["canvas"]["height"] = int(spec["height"])
        for layer in state["layers"]:
            layer["x"] = int(layer.get("x", 0)) - crop_x
            layer["y"] = int(layer.get("y", 0)) - crop_y

    @staticmethod
    def _resize_image(state: dict[str, Any], spec: dict[str, Any]) -> None:
        old_width = int(state["canvas"]["width"])
        old_height = int(state["canvas"]["height"])
        new_width = int(spec["width"])
        new_height = int(spec["height"])
        scale_x = new_width / old_width
        scale_y = new_height / old_height
        state["canvas"]["width"] = new_width
        state["canvas"]["height"] = new_height

        for layer in state["layers"]:
            for key, scale in (("x", scale_x), ("y", scale_y), ("width", scale_x), ("height", scale_y)):
                if key in layer:
                    layer[key] = int(round(layer[key] * scale))
            if "font_size" in layer:
                layer["font_size"] = max(8, int(round(layer["font_size"] * min(scale_x, scale_y))))

    @staticmethod
    def _reorder_layer(state: dict[str, Any], layer_id: str, index: int) -> None:
        layers = state["layers"]
        for current_index, layer in enumerate(layers):
            if layer["id"] == layer_id:
                moved = layers.pop(current_index)
                layers.insert(max(0, min(index, len(layers))), moved)
                return
        raise KeyError(f"Unknown layer id: {layer_id}")
