"""ASIL adapter for Blender — Pattern B (bpy script generation + execution)."""

from __future__ import annotations
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

from asil.adapter import ASILAdapter
from asil.protocol import Action, Element, Observation
from asil.rendering import (
    RenderArtifact,
    capture_window_to_png,
    launch_gui_process,
    send_keys_to_window,
    terminate_process,
)


def generate_observe_script(output_json: str) -> str:
    """Generate a Python script that Blender runs to dump scene state as JSON."""
    return textwrap.dedent(f"""\
        import bpy
        import json

        scene = bpy.context.scene
        elements = []

        def _material_color(mat):
            if mat is None:
                return None
            try:
                if mat.use_nodes and mat.node_tree:
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf is not None:
                        return list(bsdf.inputs["Base Color"].default_value)
            except Exception:
                pass
            try:
                return list(mat.diffuse_color)
            except Exception:
                return None

        for obj in scene.objects:
            elem = {{
                "id": obj.name,
                "type": obj.type.lower(),
                "label": obj.name,
                "value": {{
                    "location": list(obj.location),
                    "rotation_euler": list(obj.rotation_euler),
                    "scale": list(obj.scale),
                    "dimensions": list(obj.dimensions),
                    "visible": obj.visible_get(),
                }},
                "editable": True,
                "actions": ["set_location", "set_rotation", "set_scale", "delete",
                           "set_material", "add_modifier", "set_keyframe"],
                "metadata": {{}},
            }}
            if obj.type == "LIGHT" and obj.data:
                elem["value"]["light_type"] = getattr(obj.data, "type", "")
                elem["value"]["energy"] = getattr(obj.data, "energy", 0.0)
            if obj.data and hasattr(obj.data, "materials"):
                elem["metadata"]["materials"] = [
                    {{"name": m.name, "color": _material_color(m)}} for m in obj.data.materials if m
                ]
            if obj.modifiers:
                elem["metadata"]["modifiers"] = [
                    {{
                        "name": m.name,
                        "type": m.type,
                        "levels": getattr(m, "levels", None),
                        "render_levels": getattr(m, "render_levels", None),
                    }} for m in obj.modifiers
                ]
            if obj.animation_data:
                elem["metadata"]["has_animation_data"] = True
                if obj.animation_data.action:
                    elem["metadata"]["animation_fcurves"] = len(obj.animation_data.action.fcurves)
                    keyframe_frames = sorted(
                        {{
                            int(point.co.x)
                            for curve in obj.animation_data.action.fcurves
                            for point in curve.keyframe_points
                        }}
                    )
                    elem["metadata"]["keyframe_frames"] = keyframe_frames
            elements.append(elem)

        elements.append({{
            "id": "render_settings",
            "type": "settings_group",
            "label": "Render Settings",
            "value": {{
                "engine": scene.render.engine,
                "resolution_x": scene.render.resolution_x,
                "resolution_y": scene.render.resolution_y,
                "output_format": scene.render.image_settings.file_format,
            }},
            "editable": True,
            "actions": ["set_engine", "set_resolution", "set_samples", "render"],
        }})
        elements.append({{
            "id": "timeline_settings",
            "type": "settings_group",
            "label": "Timeline Settings",
            "value": {{
                "frame_start": scene.frame_start,
                "frame_end": scene.frame_end,
                "current_frame": scene.frame_current,
            }},
            "editable": True,
            "actions": ["set_frame_range", "set_current_frame"],
        }})

        obs = {{
            "app_state": {{
                "active_scene": scene.name,
                "current_frame": scene.frame_current,
            }},
            "elements": elements,
            "environment": {{
                "total_objects": len(scene.objects),
                "total_materials": len(bpy.data.materials),
            }},
        }}
        with open("{output_json}", "w") as f:
            json.dump(obs, f, indent=2)
    """)


def generate_action_script(action: Action) -> str:
    """Convert an ASIL Action into a bpy Python script string."""
    lines = action.params.get("script", [])
    return "\n".join(lines) + "\n"


class MockBlenderAdapter(ASILAdapter):
    """In-memory Blender mock for ground-truth evaluation without a Blender binary.

    Interprets bpy script lines to maintain scene state: objects, materials,
    render settings, and animation data.
    """
    app_name = "Blender"
    supported_action_types = ["invoke_function"]

    # Maps bpy.ops.mesh.primitive_*_add → default object name
    _PRIM_NAMES = {
        "primitive_cube_add": "Cube",
        "primitive_uv_sphere_add": "Sphere",
        "primitive_cylinder_add": "Cylinder",
        "primitive_plane_add": "Plane",
        "primitive_cone_add": "Cone",
        "primitive_torus_add": "Torus",
        "primitive_ico_sphere_add": "Icosphere",
        "primitive_circle_add": "Circle",
        "primitive_grid_add": "Grid",
        "primitive_monkey_add": "Suzanne",
    }

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        # Start with an empty scene — tasks are self-contained and add their own objects
        self._objects: list[dict] = []
        self._materials: list[dict] = []
        self._aliases: dict[str, dict] = {}
        self._render = {
            "engine": "CYCLES",
            "resolution_x": 1920,
            "resolution_y": 1080,
            "output_format": "PNG",
            "samples": 128,
        }
        self._active_object: dict | None = None
        self._active_material: str | None = None
        self._frame_start = 1
        self._frame_end = 250

    def _get_or_create_obj(self, name: str, obj_type: str = "MESH") -> dict:
        for o in self._objects:
            if o["name"] == name:
                return o
        obj = {"name": name, "type": obj_type, "location": [0, 0, 0],
               "scale": [1, 1, 1], "rotation_euler": [0, 0, 0], "dimensions": [2, 2, 2],
               "visible": True, "materials": [], "modifiers": [], "animation_data": False,
               "keyframe_frames": []}
        self._objects.append(obj)
        self._active_object = obj
        return obj

    def _exec_script(self, script_lines: list[str]) -> None:
        """Parse bpy script lines and update internal state."""
        import re
        for line in script_lines:
            line = line.strip()
            if not line or line.startswith("#") or line == "import bpy":
                continue

            # bpy.ops.object.select_all(action='SELECT') + delete
            if "bpy.ops.object.select_all" in line and "SELECT" in line:
                self._pending_select_all = True
                continue
            if "bpy.ops.object.delete" in line:
                if getattr(self, "_pending_select_all", False):
                    self._objects.clear()
                    self._aliases.clear()
                    self._active_object = None
                    self._pending_select_all = False
                continue
            self._pending_select_all = False

            # obj = bpy.context.object
            m = re.search(r"(\w+)\s*=\s*bpy\.context\.object\b", line)
            if m and self._active_object:
                self._aliases[m.group(1)] = self._active_object
                continue

            # bpy.ops.mesh.primitive_*_add(...)
            m = re.search(r"bpy\.ops\.mesh\.(primitive_\w+)\(", line)
            if m:
                prim = m.group(1)
                name = self._PRIM_NAMES.get(prim, prim.replace("primitive_", "").title())
                # Deduplicate: Cube.001, Cube.002 etc.
                base = name
                idx = 1
                existing = {o["name"] for o in self._objects}
                while name in existing:
                    name = f"{base}.{idx:03d}"
                    idx += 1
                loc = [0.0, 0.0, 0.0]
                lm = re.search(r"location=\(([^)]+)\)", line)
                if lm:
                    try:
                        loc = [float(x.strip()) for x in lm.group(1).split(",")]
                    except ValueError:
                        pass
                dims = [2.0, 2.0, 2.0]
                size_match = re.search(r"size=([0-9.]+)", line)
                radius_match = re.search(r"radius=([0-9.]+)", line)
                depth_match = re.search(r"depth=([0-9.]+)", line)
                major_match = re.search(r"major_radius=([0-9.]+)", line)
                minor_match = re.search(r"minor_radius=([0-9.]+)", line)
                if prim == "primitive_cube_add":
                    size = float(size_match.group(1)) if size_match else 2.0
                    dims = [size, size, size]
                elif prim in {"primitive_uv_sphere_add", "primitive_ico_sphere_add"}:
                    radius = float(radius_match.group(1)) if radius_match else 1.0
                    dims = [radius * 2, radius * 2, radius * 2]
                elif prim == "primitive_cylinder_add":
                    radius = float(radius_match.group(1)) if radius_match else 1.0
                    depth = float(depth_match.group(1)) if depth_match else 2.0
                    dims = [radius * 2, radius * 2, depth]
                elif prim == "primitive_plane_add":
                    size = float(size_match.group(1)) if size_match else 2.0
                    dims = [size, size, 0.0]
                elif prim == "primitive_cone_add":
                    radius = float(radius_match.group(1)) if radius_match else 1.0
                    depth = float(depth_match.group(1)) if depth_match else 2.0
                    dims = [radius * 2, radius * 2, depth]
                elif prim == "primitive_torus_add":
                    major = float(major_match.group(1)) if major_match else 1.0
                    minor = float(minor_match.group(1)) if minor_match else 0.25
                    dims = [(major + minor) * 2, (major + minor) * 2, minor * 2]
                obj = {"name": name, "type": "MESH", "location": loc,
                       "scale": [1, 1, 1], "rotation_euler": [0, 0, 0], "dimensions": dims,
                       "visible": True, "materials": [], "modifiers": [], "animation_data": False,
                       "keyframe_frames": []}
                self._objects.append(obj)
                self._active_object = obj
                continue

            # bpy.ops.object.light_add(type='POINT', ...)
            m = re.search(r"bpy\.ops\.object\.light_add\(type='(\w+)'", line)
            if m:
                ltype = m.group(1)
                name = ltype.title() + " Light"
                base = name
                idx = 1
                existing = {o["name"] for o in self._objects}
                while name in existing:
                    name = f"{base}.{idx:03d}"
                    idx += 1
                loc = [0.0, 0.0, 0.0]
                lm = re.search(r"location=\(([^)]+)\)", line)
                if lm:
                    try:
                        loc = [float(x.strip()) for x in lm.group(1).split(",")]
                    except ValueError:
                        pass
                obj = {"name": name, "type": "LIGHT", "location": loc,
                       "scale": [1, 1, 1], "rotation_euler": [0, 0, 0], "dimensions": [0, 0, 0],
                       "visible": True, "materials": [], "modifiers": [], "animation_data": False,
                       "light_type": ltype, "energy": 10.0, "keyframe_frames": []}
                self._objects.append(obj)
                self._active_object = obj
                continue

            # bpy.ops.object.camera_add(...)
            if "bpy.ops.object.camera_add" in line:
                name = "Camera"
                base = name
                idx = 1
                existing = {o["name"] for o in self._objects}
                while name in existing:
                    name = f"{base}.{idx:03d}"
                    idx += 1
                loc = [0.0, 0.0, 0.0]
                lm = re.search(r"location=\(([^)]+)\)", line)
                if lm:
                    try:
                        loc = [float(x.strip()) for x in lm.group(1).split(",")]
                    except ValueError:
                        pass
                obj = {"name": name, "type": "CAMERA", "location": loc,
                       "scale": [1, 1, 1], "rotation_euler": [0, 0, 0], "dimensions": [0, 0, 0],
                       "visible": True, "materials": [], "modifiers": [], "animation_data": False,
                       "keyframe_frames": []}
                self._objects.append(obj)
                self._active_object = obj
                continue

            # bpy.context.object.scale = (x, y, z)
            m = re.search(r"bpy\.context\.object\.scale\s*=\s*\(([^)]+)\)", line)
            if m and self._active_object:
                try:
                    self._active_object["scale"] = [float(x.strip()) for x in m.group(1).split(",")]
                    self._active_object["dimensions"] = [
                        abs(float(base) * float(scale))
                        for base, scale in zip(self._active_object["dimensions"], self._active_object["scale"])
                    ]
                except ValueError:
                    pass
                continue

            # bpy.context.object.location = (x, y, z)
            m = re.search(r"bpy\.context\.object\.location\s*=\s*\(([^)]+)\)", line)
            if m and self._active_object:
                try:
                    self._active_object["location"] = [float(x.strip()) for x in m.group(1).split(",")]
                except ValueError:
                    pass
                continue

            # alias.location = (x, y, z)
            m = re.search(r"(\w+)\.location\s*=\s*\(([^)]+)\)", line)
            if m:
                alias = self._aliases.get(m.group(1))
                if alias is not None:
                    try:
                        alias["location"] = [float(x.strip()) for x in m.group(2).split(",")]
                    except ValueError:
                        pass
                    self._active_object = alias
                    continue

            # bpy.context.object.rotation_euler = (x, y, z)
            m = re.search(r"bpy\.context\.object\.rotation_euler\s*=\s*\(([^)]+)\)", line)
            if m and self._active_object:
                try:
                    self._active_object["rotation_euler"] = [float(x.strip()) for x in m.group(1).split(",")]
                except ValueError:
                    pass
                continue

            # bpy.data.materials.new(name='...')
            m = re.search(r"bpy\.data\.materials\.new\(name=['\"]([^'\"]+)['\"]\)", line)
            if m:
                mat_name = m.group(1)
                if not any(mat["name"] == mat_name for mat in self._materials):
                    self._materials.append({"name": mat_name, "color": None})
                self._active_material = mat_name
                continue

            m = re.search(r"diffuse_color\s*=\s*\(([^)]+)\)", line)
            if m and self._active_material:
                try:
                    color = [float(x.strip()) for x in m.group(1).split(",")]
                except ValueError:
                    color = None
                if color is not None:
                    for mat in self._materials:
                        if mat["name"] == self._active_material:
                            mat["color"] = color
                            break
                continue

            m = re.search(r"Base Color'\]\.default_value\s*=\s*\(([^)]+)\)", line)
            if m and self._active_material:
                try:
                    color = [float(x.strip()) for x in m.group(1).split(",")]
                except ValueError:
                    color = None
                if color is not None:
                    for mat in self._materials:
                        if mat["name"] == self._active_material:
                            mat["color"] = color
                            break
                continue

            # bpy.context.object.data.materials.append(mat)
            if "materials.append" in line and self._active_object:
                mat_name = getattr(self, "_active_material", None)
                if mat_name and mat_name not in self._active_object["materials"]:
                    self._active_object["materials"].append(mat_name)
                continue

            # bpy.context.scene.render.resolution_x = N
            m = re.search(r"scene\.render\.resolution_x\s*=\s*(\d+)", line)
            if m:
                self._render["resolution_x"] = int(m.group(1))
                continue
            m = re.search(r"scene\.render\.resolution_y\s*=\s*(\d+)", line)
            if m:
                self._render["resolution_y"] = int(m.group(1))
                continue
            m = re.search(r"scene\.render\.engine\s*=\s*['\"](\w+)['\"]", line)
            if m:
                self._render["engine"] = m.group(1)
                continue

            # keyframe_insert
            if "keyframe_insert" in line and self._active_object:
                self._active_object["animation_data"] = True
                frame_match = re.search(r"frame\s*=\s*(\d+)", line)
                if frame_match:
                    frame = int(frame_match.group(1))
                    if frame not in self._active_object["keyframe_frames"]:
                        self._active_object["keyframe_frames"].append(frame)
                continue

            # alias.keyframe_insert(...)
            m = re.search(r"(\w+)\.keyframe_insert\(", line)
            if m:
                alias = self._aliases.get(m.group(1))
                if alias is not None:
                    alias["animation_data"] = True
                    frame_match = re.search(r"frame\s*=\s*(\d+)", line)
                    if frame_match:
                        frame = int(frame_match.group(1))
                        if frame not in alias["keyframe_frames"]:
                            alias["keyframe_frames"].append(frame)
                    self._active_object = alias
                    continue

            # bpy.context.scene.frame_start / frame_end
            m = re.search(r"scene\.frame_start\s*=\s*(\d+)", line)
            if m:
                self._frame_start = int(m.group(1))
                continue
            m = re.search(r"scene\.frame_end\s*=\s*(\d+)", line)
            if m:
                self._frame_end = int(m.group(1))
                continue

            # bpy.ops.object.modifier_add(type='...')
            m = re.search(r"bpy\.ops\.object\.modifier_add\(type=['\"](\w+)['\"]\)", line)
            if m and self._active_object:
                self._active_object["modifiers"].append({"name": m.group(1), "type": m.group(1), "levels": None, "render_levels": None})
                continue

            m = re.search(r"bpy\.context\.object\.data\.energy\s*=\s*([0-9.]+)", line)
            if m and self._active_object and self._active_object["type"] == "LIGHT":
                self._active_object["energy"] = float(m.group(1))
                continue

            # obj.hide_render = True/False
            m = re.search(r"\.hide_render\s*=\s*(True|False)", line)
            if m and self._active_object:
                self._active_object["visible"] = m.group(1) == "False"
                continue

    def reset_state(self) -> None:
        self._reset()

    def setup_state(self, initial_state: str) -> None:
        pass  # default state is the standard Blender scene

    def observe(self) -> Observation:
        elements: list[Element] = []
        for obj in self._objects:
            meta: dict = {}
            if obj["materials"]:
                meta["materials"] = [
                    next((mat for mat in self._materials if mat["name"] == name), {"name": name, "color": None})
                    for name in obj["materials"]
                ]
            if obj["modifiers"]:
                meta["modifiers"] = obj["modifiers"]
            if obj.get("animation_data"):
                meta["has_animation_data"] = True
                meta["keyframe_frames"] = sorted(obj.get("keyframe_frames", []))
            elements.append(Element(
                id=obj["name"],
                type=obj["type"].lower(),
                label=obj["name"],
                value={
                    "location": obj["location"],
                    "rotation_euler": obj["rotation_euler"],
                    "scale": obj["scale"],
                    "dimensions": obj["dimensions"],
                    "visible": obj["visible"],
                    "light_type": obj.get("light_type", ""),
                    "energy": obj.get("energy", 0.0),
                },
                editable=True,
                actions=["set_location", "set_rotation", "set_scale", "delete",
                         "set_material", "add_modifier", "set_keyframe"],
                metadata=meta,
            ))
        for mat in self._materials:
            elements.append(Element(
                id=f"material:{mat['name']}",
                type="material",
                label=mat["name"],
                value={"name": mat["name"], "color": mat.get("color")},
                editable=True,
                actions=["edit"],
            ))
        elements.append(Element(
            id="render_settings",
            type="settings_group",
            label="Render Settings",
            value=dict(self._render),
            editable=True,
            actions=["set_engine", "set_resolution", "set_samples", "render"],
        ))
        elements.append(Element(
            id="timeline_settings",
            type="settings_group",
            label="Timeline Settings",
            value={
                "frame_start": self._frame_start,
                "frame_end": self._frame_end,
                "current_frame": 1,
            },
            editable=True,
            actions=["set_frame_range", "set_current_frame"],
        ))
        return self._build_observation(
            source="mock",
            elements=elements,
            app_state={
                "active_scene": "Scene",
                "current_frame": 1,
                "frame_start": self._frame_start,
                "frame_end": self._frame_end,
            },
            environment={
                "total_objects": len(self._objects),
                "total_materials": len(self._materials),
            },
        )

    def execute(self, action: Action) -> Observation:
        script_lines = action.params.get("script", [])
        self._exec_script(script_lines)
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def render_to_png(self, output_path=None) -> Path:
        from PIL import Image, ImageDraw, ImageFont
        obs = self.observe()
        out = Path(output_path) if output_path else Path("blender_mock_state.png")
        img = Image.new("RGB", (600, 400), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), "Blender Mock State", fill=(255, 255, 255), font=font)
        y = 40
        for e in obs.interactive_elements[:20]:
            draw.text((10, y), f"{e.id} ({e.type})", fill=(200, 200, 200), font=font)
            y += 18
        img.save(out)
        return out


class BlenderAdapter(ASILAdapter):
    app_name = "Blender"
    supported_action_types = ["invoke_function"]

    def __init__(self, blend_path: str | Path = "", blender_bin: str = "blender") -> None:
        if blend_path:
            self.blend_path = Path(blend_path)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="asil_blender_"))
            self.blend_path = temp_dir / "session.blend"
        self.blender_bin = blender_bin

    @property
    def source_path(self) -> Path | None:
        return self.blend_path

    def clone(self, new_path: Path) -> "BlenderAdapter":
        self._ensure_workfile()
        if self.blend_path and self.blend_path.exists():
            import shutil
            shutil.copy2(self.blend_path, new_path)
        return BlenderAdapter(blend_path=new_path, blender_bin=self.blender_bin)

    def get_context(self) -> dict[str, str]:
        self._ensure_workfile()
        return {"blend_path": str(self.blend_path)} if self.blend_path else {}

    def _build_command(self, script_path: str, use_factory_startup: bool = False) -> list[str]:
        cmd = [self.blender_bin, "--background"]
        if use_factory_startup:
            cmd.append("--factory-startup")
        elif self.blend_path and self.blend_path.exists():
            cmd.append(str(self.blend_path))
        cmd.extend(["--python", script_path])
        return cmd

    def _initialize_workfile(self, initial_state: str = "default") -> None:
        if self.blend_path is None or self.blend_path.exists():
            return
        self.blend_path.parent.mkdir(parents=True, exist_ok=True)
        if initial_state == "blank":
            init_script = textwrap.dedent(f"""\
                import bpy
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete()
                bpy.ops.wm.save_as_mainfile(filepath=r"{self.blend_path}")
            """)
        else:
            init_script = textwrap.dedent(f"""\
                import bpy
                bpy.ops.wm.read_factory_settings(use_empty=False)
                bpy.ops.wm.save_as_mainfile(filepath=r"{self.blend_path}")
            """)
        self._run_script(init_script, use_factory_startup=True)

    def _ensure_workfile(self, initial_state: str = "default") -> None:
        if self.blend_path is not None and not self.blend_path.exists():
            self._initialize_workfile(initial_state=initial_state)

    def _run_script(
        self,
        script: str,
        *,
        save_mainfile: bool = False,
        use_factory_startup: bool = False,
    ) -> str:
        if not use_factory_startup:
            self._ensure_workfile()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            if save_mainfile:
                script = script.rstrip() + "\n\nimport bpy\nbpy.ops.wm.save_mainfile()\n"
            f.write(script)
            f.flush()
            cmd = self._build_command(f.name, use_factory_startup=use_factory_startup)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Blender script failed: {result.stderr}")
            return result.stdout

    def reset_state(self) -> None:
        if self.blend_path is not None and self.blend_path.exists():
            self.blend_path.unlink()
        self._initialize_workfile(initial_state="default")

    def setup_state(self, initial_state: str) -> None:
        if self.blend_path is not None and self.blend_path.exists():
            self.blend_path.unlink()
        self._initialize_workfile(initial_state=initial_state or "default")

    def observe(self) -> Observation:
        self._ensure_workfile()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_json = f.name

        script = generate_observe_script(output_json)
        self._run_script(script)

        with open(output_json) as f:
            data = json.load(f)

        elements = [Element(**e) for e in data.get("elements", [])]
        return self._build_observation(
            source="script_api",
            elements=elements,
            app_state=data.get("app_state", {}),
            environment=data.get("environment", {}),
        )

    def execute(self, action: Action) -> Observation:
        script = generate_action_script(action)
        self._run_script(script, save_mainfile=True)
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def describe_rendering(self) -> RenderArtifact:
        return RenderArtifact(
            filename="",
            kind="app_window",
            backend="x11-window-capture",
            actual_page=True,
            description="Screenshot of the real Blender window showing the current workspace",
        )

    def render_to_png(self, output_path: str | Path | None = None) -> Path:
        """Capture the real Blender GUI window for the current .blend file."""
        import shutil as _shutil

        out = Path(output_path) if output_path else Path("blender_render.png")
        blender = _shutil.which(self.blender_bin)
        if blender is None:
            raise RuntimeError("Blender is not installed.")
        self._ensure_workfile()
        proc = launch_gui_process(
            [
                blender,
                "--factory-startup",
                str(self.blend_path),
            ],
            extra_env={"LIBGL_ALWAYS_SOFTWARE": "1"},
        )
        try:
            try:
                send_keys_to_window("Blender", ["Escape"], timeout=45.0)
            except Exception:
                pass
            capture_window_to_png(
                out,
                title_pattern="Blender",
                timeout=45.0,
                margin=12,
                settle_delay=6.0,
            )
        finally:
            terminate_process(proc)
        return out
