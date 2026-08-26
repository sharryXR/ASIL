import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from asil.adapters.gimp import GimpAdapter
from asil.eval.evaluator import evaluate_observation
from asil.protocol import Action


def _task(task_id: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "gimp"
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def test_from_evaluation_context_creates_default_document(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)

    obs = adapter.observe()
    ids = {element.id for element in obs.interactive_elements}
    doc = next(element for element in obs.interactive_elements if element.id == "image")

    assert adapter.source_path == tmp_path / "gimp_document.png"
    assert adapter.get_context()["image_path"] == str(tmp_path / "gimp_document.png")
    assert obs.meta.app_name == "GIMP"
    assert doc.type == "document"
    assert doc.value["width"] == 800
    assert doc.value["height"] == 600
    assert {"hero_bg", "logo_block", "accent_circle", "watermark_text"}.issubset(ids)


def test_setup_state_blank_clears_default_layers(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)

    adapter.setup_state("blank")
    obs = adapter.observe()

    ids = {element.id for element in obs.interactive_elements}
    assert ids == {"image"}


def test_execute_adds_updates_and_deletes_layers(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")

    action = Action(
        action_type="invoke_function",
        target="gimp",
        params={
            "operations": [
                {
                    "action": "add_layer",
                    "id": "banner",
                    "label": "Banner",
                    "kind": "rectangle",
                    "x": 40,
                    "y": 40,
                    "width": 220,
                    "height": 90,
                    "fill": "#0047ab",
                },
                {
                    "action": "add_text_layer",
                    "id": "headline",
                    "label": "Headline",
                    "text": "Launch Day",
                    "x": 72,
                    "y": 72,
                    "font_size": 28,
                    "color": "#ffffff",
                },
                {
                    "action": "update_layer",
                    "id": "banner",
                    "changes": {"opacity": 0.75, "blend_mode": "multiply"},
                },
                {
                    "action": "delete_layer",
                    "id": "headline",
                },
            ]
        },
    )

    obs = adapter.execute(action)
    banner = next(element for element in obs.interactive_elements if element.id == "banner")

    ids = {element.id for element in obs.interactive_elements}
    assert "banner" in ids
    assert "headline" not in ids
    assert banner.type == "layer"
    assert banner.value["kind"] == "rectangle"
    assert banner.value["fill"] == "#0047ab"
    assert banner.value["opacity"] == 0.75
    assert banner.value["blend_mode"] == "multiply"


def test_execute_adds_real_image_layer_and_filters(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")

    action = Action(
        action_type="invoke_function",
        target="gimp",
        params={
            "operations": [
                {
                    "action": "add_image_layer",
                    "id": "street_photo",
                    "label": "Street Photo",
                    "asset_path": "evaluation_examples/assets/realwork_images/coffee_street_960.jpg",
                    "x": 0,
                    "y": 0,
                    "width": 800,
                    "height": 600,
                    "filters": {"contrast": 1.1, "saturation": 0.8},
                },
                {
                    "action": "apply_filter",
                    "id": "street_photo",
                    "filters": {"brightness": 0.95},
                },
            ]
        },
    )

    obs = adapter.execute(action)
    photo = next(element for element in obs.interactive_elements if element.id == "street_photo")

    assert photo.value["kind"] == "image"
    assert photo.value["width"] == 800
    assert photo.value["height"] == 600
    assert photo.value["filters"]["contrast"] == 1.1
    assert photo.value["filters"]["brightness"] == 0.95
    with Image.open(adapter.source_path) as rendered:
        assert rendered.size == (800, 600)


def test_add_image_layer_requires_asset_path(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")

    action = Action(
        action_type="invoke_function",
        target="gimp",
        params={"operations": [{"action": "add_image_layer", "id": "missing_asset"}]},
    )

    try:
        adapter.execute(action)
        assert False, "Expected missing asset path to fail"
    except ValueError as exc:
        assert "asset_path" in str(exc)


def test_execute_crop_and_resize_updates_document_and_layers(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)

    action = Action(
        action_type="invoke_function",
        target="gimp",
        params={
            "operations": [
                {"action": "crop_image", "x": 20, "y": 20, "width": 400, "height": 300},
                {"action": "resize_image", "width": 200, "height": 150},
            ]
        },
    )

    obs = adapter.execute(action)
    doc = next(element for element in obs.interactive_elements if element.id == "image")
    logo = next(element for element in obs.interactive_elements if element.id == "logo_block")

    assert doc.value["width"] == 200
    assert doc.value["height"] == 150
    assert logo.value["x"] == 10
    assert logo.value["y"] == 10
    assert logo.value["width"] == 70
    assert logo.value["height"] == 45


def test_validate_action_checks_supported_contract(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)

    good = Action(action_type="invoke_function", target="gimp", params={"operations": []})
    bad_type = Action(action_type="modify_file", target="gimp", params={"operations": []})
    bad_target = Action(action_type="invoke_function", target="other", params={"operations": []})

    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad_type)
    assert not adapter.validate_action(bad_target)


@patch("asil.adapters.gimp.terminate_process")
@patch("asil.adapters.gimp.capture_window_to_png")
@patch("asil.adapters.gimp.send_keys_to_window")
@patch("asil.adapters.gimp.launch_gui_process")
@patch("asil.adapters.gimp.ensure_user_access")
@patch("asil.adapters.gimp.shutil.which", return_value="/usr/bin/gimp")
def test_render_to_png_captures_real_gimp_window(
    mock_which,
    mock_ensure_access,
    mock_launch,
    mock_send_keys,
    mock_capture,
    mock_terminate,
    tmp_path: Path,
):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")
    adapter.execute(
        Action(
            action_type="invoke_function",
            target="gimp",
            params={
                "operations": [
                    {
                        "action": "add_layer",
                        "id": "card",
                        "label": "Card",
                        "kind": "rectangle",
                        "x": 30,
                        "y": 30,
                        "width": 120,
                        "height": 80,
                        "fill": "#ff7f50",
                    }
                ]
            },
        )
    )
    mock_launch.return_value = object()

    output = tmp_path / "render.png"
    artifact = adapter.describe_rendering()
    result = adapter.render_to_png(output)

    assert artifact.actual_page is True
    assert artifact.backend == "x11-window-capture"
    assert "real gimp window" in artifact.description.lower()
    mock_which.assert_called_once_with("gimp")
    mock_ensure_access.assert_called_once_with(adapter.source_path.parent, run_as_user="asilgui")
    mock_launch.assert_called_once()
    mock_send_keys.assert_called_once_with(
        "GIMP|GNU Image Manipulation Program",
        ["Escape"],
        timeout=60.0,
        min_width=800,
        min_height=600,
    )
    launch_args, launch_kwargs = mock_launch.call_args
    assert launch_args[0][0] == "/usr/bin/gimp"
    assert launch_args[0][-1] == str(adapter.source_path)
    assert launch_kwargs["extra_env"] == {"LIBGL_ALWAYS_SOFTWARE": "1"}
    assert launch_kwargs["run_as_user"] == "asilgui"
    mock_capture.assert_called_once_with(
        output,
        title_pattern="GIMP|GNU Image Manipulation Program",
        timeout=60.0,
        margin=12,
        settle_delay=6.0,
        min_width=800,
        min_height=600,
    )
    mock_terminate.assert_called_once_with(mock_launch.return_value)
    assert result == output


def _xcf_layer_snapshot(
    name: str,
    *,
    canvas_size: tuple[int, int] = (800, 600),
    bbox: tuple[int, int, int, int] | None = None,
    color: str = "#000000",
    shape: str = "rectangle",
    visible: bool = True,
):
    from asil.adapters.gimp import _XcfLayerSnapshot

    pixels = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if bbox is not None:
        left, top, right, bottom = bbox
        draw = ImageDraw.Draw(pixels)
        fill = color + "ff"
        if shape == "ellipse":
            draw.ellipse((left, top, right - 1, bottom - 1), fill=fill)
        else:
            draw.rectangle((left, top, right - 1, bottom - 1), fill=fill)
    return _XcfLayerSnapshot(
        name=name,
        pixels=pixels,
        offset_x=0,
        offset_y=0,
        visible=visible,
    )


def test_gui_session_uses_native_xcf_project_instead_of_flat_png(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    written_states = []
    monkeypatch.setattr(
        adapter,
        "_write_xcf_project",
        lambda state: written_states.append(state),
        raising=False,
    )

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert adapter.gui_project_path == tmp_path / "gimp_document.xcf"
    assert spec.launch_command == (
        "gimp",
        "--new-instance",
        "--no-splash",
        str(adapter.gui_project_path),
    )
    assert spec.persist_shortcuts == ("ctrl+s",)
    assert written_states == [adapter._load_state()]


def test_xcf_writer_uses_imagemagick_compatible_legacy_layer_mode(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    captured_commands = []

    monkeypatch.setattr("asil.adapters.gimp.shutil.which", lambda executable: "/usr/bin/gimp-console")

    def fake_run(command, **kwargs):
        captured_commands.append(command)
        adapter.gui_project_path.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("asil.adapters.gimp.subprocess.run", fake_run)

    adapter._write_xcf_project(adapter._load_state())

    batch_script = captured_commands[0][captured_commands[0].index("-b") + 1]
    assert batch_script.count("(gimp-layer-set-mode layer NORMAL-MODE)") == 4


def test_xcf_layer_renderer_preserves_declared_shape_bounds(tmp_path: Path):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    state = adapter._load_state()
    logo = next(layer for layer in state["layers"] if layer["id"] == "logo_block")
    accent = next(layer for layer in state["layers"] if layer["id"] == "accent_circle")

    rendered_logo = adapter._render_xcf_layer(logo, (800, 600))
    rendered_accent = adapter._render_xcf_layer(accent, (800, 600))

    assert rendered_logo.getchannel("A").getbbox() == (40, 40, 180, 130)
    assert rendered_accent.getchannel("A").getbbox() == (620, 80, 740, 200)


def test_sync_from_gui_infers_new_rectangle_only_from_xcf_name_and_pixels(
    tmp_path: Path, monkeypatch
):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")
    rectangle = _xcf_layer_snapshot(
        "Blue Banner",
        bbox=(40, 40, 300, 140),
        color="#1f5eff",
    )
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), [rectangle]), raising=False)

    adapter.sync_from_gui()
    observation = adapter.observe()
    banner = next(element for element in observation.interactive_elements if element.id == "blue_banner")

    assert banner.label == "Blue Banner"
    assert banner.value["kind"] == "rectangle"
    assert banner.value["x"] == 40
    assert banner.value["y"] == 40
    assert banner.value["width"] == 260
    assert banner.value["height"] == 100
    assert banner.value["fill"] == "#1f5eff"
    assert observation.app_state.active_document == "gimp_document.xcf"
    assert observation.app_state.document_path == str(adapter.gui_composite_path)
    assert adapter.gui_composite_path.exists()
    evaluation = evaluate_observation(
        observation,
        evaluator={
            "selection": "best_score",
            "paths": [
                {
                    "path_id": "main",
                    "checkpoints": [
                        {
                            "id": "canvas_size",
                            "weight": 1.0,
                            "rule": {"image_size": {"width": 800, "height": 600}},
                        }
                    ],
                }
            ],
        },
    )
    assert evaluation.score == 1.0


def test_sync_from_gui_infers_ellipse_from_alpha_coverage(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")
    ellipse = _xcf_layer_snapshot(
        "Coral Accent",
        bbox=(300, 120, 480, 240),
        color="#ff7f50",
        shape="ellipse",
    )
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), [ellipse]), raising=False)

    adapter.sync_from_gui()
    observation = adapter.observe()
    accent = next(element for element in observation.interactive_elements if element.id == "coral_accent")

    assert accent.value["kind"] == "ellipse"
    assert accent.value["fill"] == "#ff7f50"
    assert accent.value["x"] == 300
    assert accent.value["y"] == 120
    assert accent.value["width"] == 180
    assert accent.value["height"] == 120


def test_sync_from_gui_uses_ocr_text_without_synthetic_text_fallback(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    adapter.setup_state("blank")
    text_pixels = _xcf_layer_snapshot(
        "Sale Text",
        bbox=(80, 80, 190, 125),
        color="#ffffff",
    )
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), [text_pixels]), raising=False)
    monkeypatch.setattr(adapter, "_ocr_layer_text", lambda pixels: "SALE", raising=False)

    adapter.sync_from_gui()
    observation = adapter.observe()
    sale = next(element for element in observation.interactive_elements if element.id == "sale_text")

    assert sale.value["kind"] == "text"
    assert sale.value["text"] == "SALE"
    assert sale.value["color"] == "#ffffff"

    monkeypatch.setattr(adapter, "_ocr_layer_text", lambda pixels: "")
    adapter.sync_from_gui()
    observation = adapter.observe()
    sale = next(element for element in observation.interactive_elements if element.id == "sale_text")
    assert "text" not in sale.value


def test_sync_from_gui_reads_existing_layer_recolor_from_pixels(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    navy_logo = _xcf_layer_snapshot(
        "Logo Block",
        bbox=(40, 40, 180, 130),
        color="#123d7a",
    )
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), [navy_logo]), raising=False)

    adapter.sync_from_gui()
    observation = adapter.observe()
    logo = next(element for element in observation.interactive_elements if element.id == "logo_block")

    assert logo.value["kind"] == "rectangle"
    assert logo.value["fill"] == "#123d7a"
    assert logo.value["fill"] != adapter._load_state()["layers"][1]["fill"]


def test_sync_from_gui_does_not_keep_deleted_synthetic_layer(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    remaining = [
        _xcf_layer_snapshot("Hero Background", bbox=(0, 0, 800, 600), color="#f5f1e8"),
        _xcf_layer_snapshot("Logo Block", bbox=(40, 40, 180, 130), color="#cc4444"),
        _xcf_layer_snapshot("Watermark", bbox=(560, 500, 730, 550), color="#b8b8b8"),
    ]
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), remaining), raising=False)
    monkeypatch.setattr(adapter, "_ocr_layer_text", lambda pixels: "DRAFT", raising=False)

    adapter.sync_from_gui()
    ids = {element.id for element in adapter.observe().interactive_elements}

    assert "accent_circle" not in ids
    assert {"hero_bg", "logo_block", "watermark_text"}.issubset(ids)


def test_hidden_accent_layer_is_not_mistaken_for_deletion(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    hidden_accent = _xcf_layer_snapshot(
        "Accent Circle",
        bbox=(620, 80, 740, 200),
        color="#4da3ff",
        shape="ellipse",
        visible=False,
    )
    monkeypatch.setattr(
        adapter,
        "_read_xcf_project",
        lambda: ((800, 600), [hidden_accent]),
        raising=False,
    )

    adapter.sync_from_gui()
    accent = next(
        element for element in adapter.observe().interactive_elements if element.id == "accent_circle"
    )

    assert accent.value["visible"] is False


def test_gimp_05_scores_only_after_accent_layer_is_really_deleted(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    snapshots = [
        _xcf_layer_snapshot("Hero Background", bbox=(0, 0, 800, 600), color="#f5f1e8"),
        _xcf_layer_snapshot("Logo Block", bbox=(40, 40, 180, 130), color="#cc4444"),
        _xcf_layer_snapshot(
            "Accent Circle",
            bbox=(620, 80, 740, 200),
            color="#4da3ff",
            shape="ellipse",
        ),
        _xcf_layer_snapshot("Watermark", bbox=(560, 500, 730, 550), color="#b8b8b8"),
    ]
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), snapshots))
    monkeypatch.setattr(adapter, "_ocr_layer_text", lambda pixels: "DRAFT")
    task = _task("gimp_05")

    adapter.sync_from_gui()
    before_delete = evaluate_observation(adapter.observe(), evaluator=task["evaluator"])

    snapshots[:] = [snapshot for snapshot in snapshots if snapshot.name != "Accent Circle"]
    adapter.sync_from_gui()
    after_delete = evaluate_observation(adapter.observe(), evaluator=task["evaluator"])

    assert before_delete.score == 0.0
    assert not before_delete.success
    assert after_delete.score == 1.0
    assert after_delete.success


def test_failed_ocr_does_not_restore_old_synthetic_watermark_text(tmp_path: Path, monkeypatch):
    adapter = GimpAdapter.from_evaluation_context(tmp_path)
    watermark = _xcf_layer_snapshot(
        "Watermark",
        bbox=(560, 500, 730, 550),
        color="#b8b8b8",
    )
    monkeypatch.setattr(adapter, "_read_xcf_project", lambda: ((800, 600), [watermark]))
    monkeypatch.setattr(adapter, "_ocr_layer_text", lambda pixels: "")

    adapter.sync_from_gui()
    observed = next(
        element for element in adapter.observe().interactive_elements if element.id == "watermark_text"
    )

    assert "text" not in observed.value
    assert "DRAFT" not in observed.value.values()


def test_gimp_example_set_contains_20_tasks():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples" / "gimp"
    tasks = sorted(path for path in root.glob("gimp_*.json") if path.stem[5:].isdigit())

    assert len(tasks) == 20
    ids = [json.loads(task.read_text(encoding="utf-8"))["id"] for task in tasks]
    assert ids == [f"gimp_{idx:02d}" for idx in range(1, 21)]


def test_representative_tasks_evaluate_successfully(tmp_path: Path):
    for task_id in ("gimp_01", "gimp_11", "gimp_20"):
        task = _task(task_id)
        adapter = GimpAdapter.from_evaluation_context(tmp_path / task_id)
        adapter.setup_state(task["_asil"].get("initial_state", "default"))

        observation = adapter.observe()
        for action_data in task["_asil"]["actions"]:
            action = Action(**action_data)
            observation = adapter.execute(action)

        report = evaluate_observation(
            observation,
            validation=task["_asil"].get("validation"),
            evaluator=task.get("evaluator"),
        )
        assert report.success, task_id


def test_realwork_gimp_task_set_exists():
    root = Path(__file__).resolve().parent.parent / "evaluation_examples"
    hard = json.loads((root / "test_full15_realwork_hard.json").read_text(encoding="utf-8"))
    hard_smoke = json.loads((root / "test_full15_realwork_hard_smoke.json").read_text(encoding="utf-8"))
    assert sum(len(task_ids) for task_ids in hard.values()) == 80
    assert len(hard["gimp"]) == 24
    assert len(hard["multi_apps"]) == 30
    assert sum(len(hard[software]) for software in ("drawio", "libreoffice", "libreoffice_writer", "libreoffice_impress")) == 12
    assert sum(len(hard[software]) for software in ("code_server", "gitea", "jupyterlab")) == 8
    assert sum(len(hard[software]) for software in ("nautilus", "thunderbird")) == 6
    assert sum(len(task_ids) for task_ids in hard_smoke.values()) == 9
    generated = [
        json.loads((root / "examples" / "gimp" / f"{task_id}.json").read_text(encoding="utf-8"))
        for task_id in hard["gimp"]
        if task_id >= "gimp_realwork_hard_007"
    ]
    assert len(generated) == 18
    assert all(task["_asil"]["category"] == "realwork_hard" for task in generated)
    assert all(task["_asil"]["workflow_complexity"]["meaningful_actions"] >= 9 for task in generated)
    assert all(task["_asil"]["workflow_complexity"]["uses_real_image_asset"] is True for task in generated)

    mixed_tasks = []
    for software, task_ids in hard.items():
        for task_id in task_ids:
            mixed_tasks.append(json.loads((root / "examples" / software / f"{task_id}.json").read_text(encoding="utf-8")))
    assert all(task["_asil"]["category"] == "realwork_hard" for task in mixed_tasks)
