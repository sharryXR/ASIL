from pathlib import Path

from asil.adapters.inkscape import InkscapeAdapter
from asil.protocol import Action


def test_observe_returns_all_elements(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    obs = adapter.observe()

    assert obs.meta.app_name == "Inkscape"
    assert obs.meta.observation_source == "file_parse"
    # sample_svg has: rect1, circle1, text1, rect2 + 2 layer groups = 6 elements
    assert len(obs.interactive_elements) == 6
    ids = {e.id for e in obs.interactive_elements}
    assert {"rect1", "circle1", "text1", "rect2"}.issubset(ids)


def test_observe_extracts_rect_attributes(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    obs = adapter.observe()

    rect = next(e for e in obs.interactive_elements if e.id == "rect1")
    assert rect.type == "rect"
    assert rect.value["x"] == "10"
    assert rect.value["y"] == "20"
    assert rect.value["width"] == "100"
    assert rect.value["height"] == "50"
    assert "fill:#ff0000" in rect.value["style"]


def test_observe_expands_single_rounded_rect_radius_to_svg_effective_pair(tmp_path: Path):
    svg = tmp_path / "rounded.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
             <rect id="rect1" x="10" y="20" width="100" height="50" rx="10"/>
           </svg>""",
        encoding="utf-8",
    )

    rect = next(
        element
        for element in InkscapeAdapter(svg).observe().interactive_elements
        if element.id == "rect1"
    )

    # SVG defines an omitted ry as equal to rx (and vice versa). Inkscape may
    # persist only one radius even though both toolbar fields visibly show 10.
    assert rect.value["rx"] == "10"
    assert rect.value["ry"] == "10"


def test_observe_extracts_circle_attributes(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    obs = adapter.observe()

    circle = next(e for e in obs.interactive_elements if e.id == "circle1")
    assert circle.type == "circle"
    assert circle.value["cx"] == "200"
    assert circle.value["cy"] == "150"
    assert circle.value["r"] == "40"


def test_observe_extracts_text(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    obs = adapter.observe()

    text = next(e for e in obs.interactive_elements if e.id == "text1")
    assert text.type == "text"
    assert text.value["text_content"] == "Hello"


def test_observe_extracts_layers(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    obs = adapter.observe()

    assert obs.app_state.current_view == "canvas"
    assert obs.environment.system.get("canvas_width") == 800.0


def test_observe_extracts_non_layer_groups_and_parent_metadata(tmp_path: Path):
    svg = tmp_path / "grouped.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg"
                 xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
                 width="200" height="100">
             <g id="layer1" inkscape:groupmode="layer" inkscape:label="Layer 1">
               <g id="shape_group">
                 <rect id="g_rect1" x="10" y="10" width="20" height="20" style="fill:red"/>
               </g>
             </g>
           </svg>""",
        encoding="utf-8",
    )

    adapter = InkscapeAdapter(svg_path=svg)
    obs = adapter.observe()

    group = next(e for e in obs.interactive_elements if e.id == "shape_group")
    rect = next(e for e in obs.interactive_elements if e.id == "g_rect1")

    assert group.type == "group"
    assert group.metadata["child_ids"] == ["g_rect1"]
    assert rect.metadata["parent_id"] == "shape_group"


def test_execute_set_attribute(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    action = Action(
        action_type="modify_file",
        target=str(sample_svg),
        params={
            "operations": [
                {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
            ]
        },
    )
    obs = adapter.execute(action)
    rect = next(e for e in obs.interactive_elements if e.id == "rect1")
    assert rect.value["width"] == "200"


def test_execute_add_element(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    action = Action(
        action_type="modify_file",
        target=str(sample_svg),
        params={
            "operations": [
                {
                    "action": "add_element",
                    "parent_xpath": "//*[@id='layer1']",
                    "tag": "rect",
                    "attributes": {
                        "id": "new_rect",
                        "x": "500",
                        "y": "400",
                        "width": "120",
                        "height": "80",
                        "style": "fill:#ffff00",
                    },
                }
            ]
        },
    )
    obs = adapter.execute(action)
    ids = {e.id for e in obs.interactive_elements}
    assert "new_rect" in ids


def test_execute_delete_element(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    action = Action(
        action_type="modify_file",
        target=str(sample_svg),
        params={
            "operations": [{"action": "delete", "xpath": "//*[@id='circle1']"}]
        },
    )
    obs = adapter.execute(action)
    ids = {e.id for e in obs.interactive_elements}
    assert "circle1" not in ids
    assert len(obs.interactive_elements) == 5


def test_validate_action(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    good = Action(action_type="modify_file", target="x.svg", params={})
    bad = Action(action_type="invoke_function", target="bpy", params={})
    assert adapter.validate_action(good)
    assert not adapter.validate_action(bad)


def test_render_to_png_raises_without_inkscape(sample_svg: Path, monkeypatch):
    """render_to_png should raise RuntimeError when Inkscape is not installed."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = InkscapeAdapter(svg_path=sample_svg)
    try:
        adapter.render_to_png()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Inkscape is not installed" in str(e)


def test_describe_rendering_reports_real_window_capture(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    artifact = adapter.describe_rendering()

    assert artifact.actual_page is True
    assert artifact.kind == "app_window"
    assert artifact.backend == "x11-window-capture"
    assert "real inkscape window" in artifact.description.lower()


def test_render_to_png_uses_real_window_capture(sample_svg: Path, monkeypatch, tmp_path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    calls = {}

    def fake_launch(command, **kwargs):
        calls["command"] = list(command)
        calls["launch_kwargs"] = dict(kwargs)
        return object()

    def fake_capture(output_path, **kwargs):
        calls["capture_output_path"] = Path(output_path)
        calls["capture_kwargs"] = dict(kwargs)
        Path(output_path).write_text("png", encoding="utf-8")
        return Path(output_path)

    def fake_terminate(proc, **kwargs):
        calls["terminated"] = proc
        calls["terminate_kwargs"] = dict(kwargs)

    monkeypatch.setattr("asil.adapters.inkscape.shutil.which", lambda _: "/usr/bin/inkscape")
    monkeypatch.setattr("asil.adapters.inkscape.launch_gui_process", fake_launch)
    monkeypatch.setattr("asil.adapters.inkscape.capture_window_to_png", fake_capture)
    monkeypatch.setattr(
        "asil.adapters.inkscape.ensure_user_access",
        lambda path, **kwargs: calls.setdefault("access", (Path(path), kwargs)),
    )
    monkeypatch.setattr("asil.adapters.inkscape.terminate_process", fake_terminate)

    out = adapter.render_to_png(tmp_path / "inkscape.png")

    assert out == tmp_path / "inkscape.png"
    assert calls["command"][0] == "/usr/bin/inkscape"
    assert calls["command"][-1] == str(sample_svg)
    assert calls["launch_kwargs"]["extra_env"] == {"LIBGL_ALWAYS_SOFTWARE": "1"}
    assert calls["launch_kwargs"]["run_as_user"] == "asilgui"
    assert calls["capture_output_path"] == tmp_path / "inkscape.png"
    assert calls["capture_kwargs"]["title_pattern"] == r".*Inkscape|.* - Inkscape"
    assert calls["capture_kwargs"]["timeout"] == 60.0
    assert calls["capture_kwargs"]["min_width"] == 900
    assert calls["capture_kwargs"]["min_height"] == 700
    assert calls["terminated"] is not None
