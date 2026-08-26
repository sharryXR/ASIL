"""Tests for Blender adapter — tests script generation, not bpy execution."""

import json
from pathlib import Path
from unittest.mock import patch

from asil.adapters.blender import BlenderAdapter, MockBlenderAdapter, generate_observe_script, generate_action_script
from asil.protocol import Action


def test_generate_observe_script():
    script = generate_observe_script(output_json="/tmp/obs.json")
    assert "import bpy" in script
    assert "import json" in script
    assert "scene.objects" in script or "bpy.data.objects" in script
    assert "/tmp/obs.json" in script
    assert '"dimensions"' in script
    assert '"timeline_settings"' in script
    assert "keyframe_frames" in script


def test_generate_action_script_create_cube():
    action = Action(
        action_type="invoke_function",
        target="bpy",
        params={
            "script": [
                "import bpy",
                "bpy.ops.mesh.primitive_cube_add(location=(2, 0, 0), size=1.5)",
            ]
        },
    )
    script = generate_action_script(action)
    assert "primitive_cube_add" in script
    assert "location=(2, 0, 0)" in script


def test_generate_action_script_set_material():
    action = Action(
        action_type="invoke_function",
        target="bpy",
        params={
            "script": [
                "import bpy",
                "mat = bpy.data.materials.new('Red')",
                "mat.diffuse_color = (1, 0, 0, 1)",
                "bpy.data.objects['Cube'].data.materials.append(mat)",
            ]
        },
    )
    script = generate_action_script(action)
    assert "materials.new('Red')" in script
    assert "diffuse_color" in script


def test_generate_action_script_render():
    action = Action(
        action_type="invoke_function",
        target="bpy",
        params={
            "script": [
                "import bpy",
                "bpy.context.scene.render.filepath = '/tmp/out.png'",
                "bpy.ops.render.render(write_still=True)",
            ]
        },
    )
    script = generate_action_script(action)
    assert "render.render" in script
    assert "/tmp/out.png" in script


def test_adapter_validate_action():
    adapter = BlenderAdapter(blend_path="/tmp/test.blend")
    good = Action(action_type="invoke_function", target="bpy", params={})
    bad = Action(action_type="modify_file", target="x.svg", params={})
    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad)


def test_adapter_build_blender_command():
    adapter = BlenderAdapter(blend_path="/tmp/test.blend")
    cmd = adapter._build_command("/tmp/script.py")
    assert cmd[0] == "blender"
    assert "--background" in cmd
    assert "--python" in cmd
    assert "/tmp/script.py" in cmd


def test_execute_saves_mainfile():
    adapter = BlenderAdapter(blend_path="/tmp/test.blend")
    action = Action(action_type="invoke_function", target="bpy", params={"script": ["import bpy"]})

    with patch.object(adapter, "_run_script") as mock_run:
        with patch.object(adapter, "observe", return_value="obs"):
            result = adapter.execute(action)

    assert result == "obs"
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert "import bpy" in args[0]
    assert kwargs["save_mainfile"] is True


def test_reset_state_initializes_workfile(tmp_path):
    blend_path = tmp_path / "session.blend"
    adapter = BlenderAdapter(blend_path=blend_path)

    with patch.object(adapter, "_run_script") as mock_run:
        adapter.reset_state()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["use_factory_startup"] is True


def test_render_to_png_captures_real_blender_window(tmp_path):
    blend_path = tmp_path / "scene.blend"
    adapter = BlenderAdapter(blend_path=blend_path)

    with patch("shutil.which", return_value="/usr/bin/blender"):
        with patch.object(adapter, "_ensure_workfile") as mock_ensure:
            with patch("asil.adapters.blender.launch_gui_process") as mock_launch:
                with patch("asil.adapters.blender.send_keys_to_window") as mock_keys:
                    with patch("asil.adapters.blender.capture_window_to_png") as mock_capture:
                        with patch("asil.adapters.blender.terminate_process") as mock_terminate:
                            proc = mock_launch.return_value
                            out = adapter.render_to_png(tmp_path / "render.png")

    assert out == tmp_path / "render.png"
    mock_ensure.assert_called_once()
    launch_args = mock_launch.call_args[0][0]
    assert launch_args[:2] == ["/usr/bin/blender", "--factory-startup"]
    assert str(blend_path) in launch_args
    assert mock_launch.call_args.kwargs["extra_env"] == {"LIBGL_ALWAYS_SOFTWARE": "1"}
    mock_keys.assert_called_once_with("Blender", ["Escape"], timeout=45.0)
    mock_capture.assert_called_once_with(
        tmp_path / "render.png",
        title_pattern="Blender",
        timeout=45.0,
        margin=12,
        settle_delay=6.0,
    )
    mock_terminate.assert_called_once_with(proc)


def test_blender_19_ground_truth_script_advances_to_visible_end_state():
    task = json.loads(
        Path("evaluation_examples/examples/blender/blender_19.json").read_text()
    )
    script = "\n".join(task["_asil"]["actions"][0]["params"]["script"])

    assert "bpy.context.object.location = (3, 0, 0)" in script
    assert "bpy.context.scene.frame_set(20)" in script


def test_blender_10_ground_truth_script_advances_to_final_keyframe():
    task = json.loads(
        Path("evaluation_examples/examples/blender/blender_10.json").read_text()
    )
    script = "\n".join(task["_asil"]["actions"][0]["params"]["script"])

    assert "cube.location = (5, 0, 0)" in script
    assert "bpy.context.scene.frame_end = 24" in script
    assert "bpy.context.scene.frame_set(24)" in script


def test_mock_blender_alias_assignment_updates_object_location_and_keyframes():
    adapter = MockBlenderAdapter()
    action = Action(
        action_type="invoke_function",
        target="bpy",
        params={
            "script": [
                "import bpy",
                "bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))",
                "cube = bpy.context.object",
                "cube.location = (0, 0, 0)",
                "cube.keyframe_insert(data_path='location', frame=1)",
                "cube.location = (5, 0, 0)",
                "cube.keyframe_insert(data_path='location', frame=24)",
                "bpy.context.scene.frame_start = 1",
                "bpy.context.scene.frame_end = 24",
            ]
        },
    )

    obs = adapter.execute(action)
    cube = next(element for element in obs.interactive_elements if element.id == "Cube")

    assert cube.value["location"] == [5.0, 0.0, 0.0]
    assert cube.metadata["keyframe_frames"] == [1, 24]
