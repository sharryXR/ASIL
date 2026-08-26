from pathlib import Path

from PIL import Image

from asil.eval.evaluator import _select_group_winners, evaluate_observation, evaluate_task_result
from asil.protocol import AppState, Element, Environment, Meta, Observation


def _make_obs(elements, current_view=""):
    return Observation(
        meta=Meta(app_name="TestApp", observation_source="unit"),
        app_state=AppState(current_view=current_view),
        interactive_elements=elements,
        environment=Environment(),
    )


def _make_sheet_obs(cells: dict[str, tuple[str, str]]):
    elements = []
    for cell_id, (value, data_type) in cells.items():
        elements.append(
            Element(
                id=cell_id,
                type="cell",
                label=cell_id,
                value=value,
                data_type=data_type,
            )
        )
    return _make_obs(elements)


def _make_image_obs(path: Path):
    return Observation(
        meta=Meta(app_name="GIMP", observation_source="png_metadata"),
        app_state=AppState(current_view="canvas", active_document=path.name, document_path=str(path)),
        interactive_elements=[],
        environment=Environment(),
    )


def test_evaluator_first_complete_path_does_not_cross_score():
    obs = _make_obs(
        [
            Element(id="label:renamed", type="label", value={"color": "#000000"}),
        ]
    )
    spec = {
        "paths": [
            {
                "path_id": "rename_only",
                "conditions": [{"element_exists": "label:renamed"}],
            },
            {
                "path_id": "rename_and_color",
                "conditions": [
                    {"element_exists": "label:renamed"},
                    {"element_value": {"id": "label:renamed", "key": "color", "expected": "#ff0000"}},
                ],
            },
        ],
        "selection": "first_complete",
    }

    report = evaluate_observation(obs, evaluator=spec)

    assert report.matched_path_id == "rename_only"
    assert report.success is True
    assert report.score == 1.0
    assert report.path_reports[0].exclusive_group == "default"


def test_evaluator_checkpoint_scores_sum_within_one_path():
    obs = _make_obs(
        [
            Element(id="stream_status", type="status", value={"active": False}),
        ],
        current_view="BRB",
    )
    spec = {
        "paths": [
            {
                "path_id": "obs_live_setup",
                "checkpoints": [
                    {"id": "scene", "weight": 0.4, "rule": {"current_scene": "BRB"}},
                    {"id": "stream", "weight": 0.6, "rule": {"stream_active": True}},
                ],
            }
        ]
    }

    report = evaluate_observation(obs, evaluator=spec)

    assert report.matched_path_id == "obs_live_setup"
    assert report.success is False
    assert report.score == 0.4
    assert report.path_reports[0].checkpoints[0].passed is True
    assert report.path_reports[0].checkpoints[1].passed is False
    assert report.migration_mode == "native"


def test_evaluator_supports_rendered_image_size_and_region_stats(tmp_path: Path):
    image_path = tmp_path / "rendered.png"
    Image.new("RGB", (20, 10), "#202020").save(image_path)
    obs = _make_image_obs(image_path)
    spec = {
        "paths": [
            {
                "path_id": "image_pixels",
                "checkpoints": [
                    {"id": "size", "weight": 0.5, "rule": {"image_size": {"width": 20, "height": 10}}},
                    {
                        "id": "dark_region",
                        "weight": 0.5,
                        "rule": {
                            "image_region_stat": {
                                "box": [0, 0, 20, 10],
                                "metric": "luminance",
                                "min": 25,
                                "max": 40,
                            }
                        },
                    },
                ],
            }
        ]
    }

    report = evaluate_observation(obs, evaluator=spec)

    assert report.success is True
    assert report.score == 1.0


def test_evaluator_app_rule_filters_to_named_child_app():
    obs = _make_obs(
        [
            Element(id="code_server::file:report.md", type="file", value={"content": "ready"}, metadata={"app": "code_server", "local_id": "file:report.md"}),
            Element(id="jupyterlab::file:report.md", type="file", value={"content": "not ready"}, metadata={"app": "jupyterlab", "local_id": "file:report.md"}),
        ]
    )
    spec = {
        "paths": [
            {
                "path_id": "main",
                "checkpoints": [
                    {
                        "id": "code_file",
                        "weight": 1.0,
                        "rule": {
                            "app_rule": {
                                "app": "code_server",
                                "rule": {"element_contains": {"id": "file:report.md", "key": "content", "expected": "ready"}},
                            }
                        },
                    }
                ],
            }
        ]
    }

    report = evaluate_observation(obs, evaluator=spec)

    assert report.success is True
    assert report.score == 1.0


def test_evaluator_app_rule_fails_when_named_child_app_does_not_match():
    obs = _make_obs(
        [
            Element(id="code_server::state", type="state", value={"value": "ok"}, metadata={"app": "code_server", "local_id": "state"}),
        ]
    )
    spec = {"paths": [{"path_id": "main", "checkpoints": [{"id": "state", "weight": 1.0, "rule": {"app_rule": {"app": "jupyterlab", "rule": {"element_exists": "state"}}}}]}]}

    report = evaluate_observation(obs, evaluator=spec)

    assert report.success is False
    assert report.score == 0.0


def test_select_group_winners_returns_one_report_per_exclusive_group():
    reports = [
        type("PathReportLike", (), {"path_id": "path_a1", "score": 0.4, "complete": False, "exclusive_group": "success"})(),
        type("PathReportLike", (), {"path_id": "path_a2", "score": 0.8, "complete": False, "exclusive_group": "success"})(),
        type("PathReportLike", (), {"path_id": "path_b1", "score": 0.6, "complete": False, "exclusive_group": "fallback"})(),
        type("PathReportLike", (), {"path_id": "path_default", "score": 0.2, "complete": False, "exclusive_group": "default"})(),
    ]

    winners = _select_group_winners(reports, "best_score")

    assert [report.path_id for report in winners] == ["path_a2", "path_b1", "path_default"]


def test_evaluator_normalizes_path_scores_when_weights_do_not_sum_to_one():
    obs = _make_obs(
        [
            Element(id="Cube.001", type="mesh", metadata={"keyframe_frames": [1, 24]}),
            Element(id="timeline_settings", type="settings_group", value={"frame_start": 1, "frame_end": 24}),
            Element(id="Cube.002", type="mesh"),
            Element(id="Camera", type="camera"),
            Element(id="Light", type="light"),
        ]
    )
    spec = {
        "paths": [
            {
                "path_id": "underweighted_path",
                "checkpoints": [
                    {"id": "cube", "weight": 0.1, "rule": {"object_exists": "Cube.001"}},
                    {
                        "id": "keyframes",
                        "weight": 0.35,
                        "rule": {
                            "element_metadata_value": {
                                "id": "Cube.001",
                                "key": "keyframe_frames",
                                "expected": [1, 24],
                            }
                        },
                    },
                    {
                        "id": "timeline_start",
                        "weight": 0.2,
                        "rule": {"element_value": {"id": "timeline_settings", "key": "frame_start", "expected": 1}},
                    },
                    {
                        "id": "timeline_end",
                        "weight": 0.2,
                        "rule": {"element_value": {"id": "timeline_settings", "key": "frame_end", "expected": 24}},
                    },
                    {"id": "count", "weight": 0.1, "rule": {"scene_object_count": 4}},
                ],
            }
        ]
    }

    report = evaluate_observation(obs, evaluator=spec)

    assert report.success is True
    assert report.score == 1.0


def test_blender_06_uses_object_scoped_material_path():
    task = type(
        "Task",
        (),
        {
            "id": "blender_06",
            "software": "blender",
            "validation": {"material_exists": "RedMaterial"},
            "evaluator": {},
        },
    )()
    obs = _make_obs(
        [
            Element(id="Cube", type="mesh", metadata={"materials": [{"name": "RedMaterial"}]}),
            Element(id="Light", type="light", metadata={}),
        ]
    )

    report = evaluate_task_result(task, obs)

    assert report.mode == "paths"
    assert report.matched_path_id == "cube_red_material_applied"
    assert report.success is True
    assert report.migration_mode in {"native", "synthesized"}


def test_blender_06_fails_when_material_is_not_on_cube():
    task = type(
        "Task",
        (),
        {
            "id": "blender_06",
            "software": "blender",
            "validation": {"material_exists": "RedMaterial"},
            "evaluator": {},
        },
    )()
    obs = _make_obs(
        [
            Element(id="Cube", type="mesh", metadata={"materials": [{"name": "BlueMaterial"}]}),
            Element(id="Light", type="light", metadata={"materials": [{"name": "RedMaterial"}]}),
        ]
    )

    report = evaluate_task_result(task, obs)

    assert report.mode == "paths"
    assert report.success is False
    assert report.score < 1.0
    assert report.path_reports[0].checkpoints[0].gui_visible_required is False


def test_gui_visible_required_rejects_hidden_property_checkpoint_even_when_rule_matches():
    task = type(
        "Task",
        (),
        {
            "id": "drawio_hidden_canvas",
            "software": "drawio",
            "validation": {},
            "gui_expectations": {
                "success_surface": "diagram_canvas",
                "visible_change_summary": "A visible diagram change occurs.",
                "checkpoint_visibility": {"canvas_width": "visible_on_diagram"},
            },
            "evaluator": {
                "paths": [
                    {
                        "path_id": "main",
                        "checkpoints": [
                            {
                                "id": "canvas_width",
                                "weight": 1.0,
                                "gui_visible_required": True,
                                "rule": {
                                    "element_value": {
                                        "id": "canvas",
                                        "key": "width",
                                        "expected": 1200,
                                    }
                                },
                            }
                        ],
                    }
                ]
            },
        },
    )()
    obs = _make_obs([Element(id="canvas", type="canvas", value={"width": 1200})], current_view="diagram_canvas")

    report = evaluate_task_result(task, obs)

    assert report.success is False
    assert report.score == 0.0
    assert report.path_reports[0].checkpoints[0].passed is False


def test_gui_visible_required_accepts_visible_output_checkpoint():
    task = type(
        "Task",
        (),
        {
            "id": "jupyter_visible_output",
            "software": "jupyterlab",
            "validation": {},
            "gui_expectations": {
                "success_surface": "notebook",
                "visible_change_summary": "The notebook visibly shows the output.",
                "checkpoint_visibility": {"output": "visible_in_notebook_output"},
            },
            "evaluator": {
                "paths": [
                    {
                        "path_id": "main",
                        "checkpoints": [
                            {
                                "id": "output",
                                "weight": 1.0,
                                "gui_visible_required": True,
                                "rule": {
                                    "element_value": {
                                        "id": "cell:notebooks/analysis.ipynb:1",
                                        "key": "output",
                                        "expected": "42",
                                    }
                                },
                            }
                        ],
                    }
                ]
            },
        },
    )()
    obs = _make_obs(
        [
            Element(
                id="cell:notebooks/analysis.ipynb:1",
                type="cell",
                value={"output": "42", "source": "total = 21 * 2"},
            )
        ],
        current_view="notebook",
    )

    report = evaluate_task_result(task, obs)

    assert report.success is True
    assert report.score == 1.0


def test_gui_visible_required_accepts_visible_dimension_checkpoint():
    task = type(
        "Task",
        (),
        {
            "id": "gimp_visible_size",
            "software": "gimp",
            "validation": {},
            "gui_expectations": {
                "success_surface": "canvas",
                "visible_change_summary": "The visible layer becomes wider.",
                "checkpoint_visibility": {"logo_size": "visible_on_canvas"},
            },
            "evaluator": {
                "paths": [
                    {
                        "path_id": "main",
                        "checkpoints": [
                            {
                                "id": "logo_size",
                                "weight": 1.0,
                                "gui_visible_required": True,
                                "rule": {
                                    "element_value": {
                                        "id": "logo_block",
                                        "key": "width",
                                        "expected": 180,
                                    }
                                },
                            }
                        ],
                    }
                ]
            },
        },
    )()
    obs = _make_obs(
        [
            Element(
                id="logo_block",
                type="layer",
                value={"width": 180, "height": 110, "x": 500, "y": 360},
            )
        ],
        current_view="canvas",
    )

    report = evaluate_task_result(task, obs)

    assert report.success is True
    assert report.score == 1.0


def test_gui_visible_required_fails_when_checkpoint_visibility_mapping_is_missing_at_runtime():
    task = type(
        "Task",
        (),
        {
            "id": "missing_visibility_mapping",
            "software": "obs",
            "validation": {},
            "gui_expectations": {
                "success_surface": "program_view",
                "visible_change_summary": "The program view changes.",
                "checkpoint_visibility": {},
            },
            "evaluator": {
                "paths": [
                    {
                        "path_id": "main",
                        "checkpoints": [
                            {
                                "id": "scene",
                                "weight": 1.0,
                                "gui_visible_required": True,
                                "rule": {"app_view": "program_view"},
                            }
                        ],
                    }
                ]
            },
        },
    )()
    obs = _make_obs([], current_view="program_view")

    report = evaluate_task_result(task, obs)

    assert report.success is False
    assert report.score == 0.0


def test_element_value_matches_numeric_lists_with_tolerance():
    obs = _make_obs(
        [
            Element(
                id="Cylinder",
                type="mesh",
                value={"dimensions": [1.0, 1.0, 2.0]},
            )
        ]
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "cylinder_dimensions",
                    "checkpoints": [
                        {
                            "id": "dims",
                            "weight": 1.0,
                            "rule": {
                                "element_value": {
                                    "id": "Cylinder",
                                    "key": "dimensions",
                                    "expected": [1, 1, 2],
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert report.success is True
    assert report.score == 1.0


def test_blender_specific_rules_cover_modifier_and_material_color():
    obs = _make_obs(
        [
            Element(
                id="Suzanne",
                type="mesh",
                metadata={
                    "materials": [{"name": "GreenMaterial", "color": [0.0, 1.0, 0.0, 1.0]}],
                    "modifiers": [{"name": "Subdivision", "type": "SUBSURF"}],
                },
            )
        ]
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "suzanne_green_subsurf",
                    "checkpoints": [
                        {
                            "id": "modifier",
                            "weight": 0.5,
                            "rule": {"object_has_modifier": {"id": "Suzanne", "expected": "SUBSURF"}},
                        },
                        {
                            "id": "material_color",
                            "weight": 0.5,
                            "rule": {
                                "object_material_color": {
                                    "id": "Suzanne",
                                    "expected": [0, 1, 0, 1],
                                }
                            },
                        },
                    ],
                }
            ]
        },
    )

    assert report.success is True
    assert report.score == 1.0


def test_element_metadata_value_matches_keyframe_frames():
    obs = _make_obs(
        [
            Element(
                id="Cube.001",
                type="mesh",
                metadata={"keyframe_frames": [1, 24]},
            )
        ]
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "animated_cube",
                    "checkpoints": [
                        {
                            "id": "frames",
                            "weight": 1.0,
                            "rule": {
                                "element_metadata_value": {
                                    "id": "Cube.001",
                                    "key": "keyframe_frames",
                                    "expected": [1, 24],
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert report.success is True


def test_unknown_rule_does_not_silently_pass():
    obs = _make_obs([])

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "unknown_rule_path",
                    "checkpoints": [
                        {"id": "mystery", "weight": 1.0, "rule": {"totally_unknown_rule": {"x": 1}}},
                    ],
                }
            ]
        },
    )

    assert report.success is False
    assert report.score == 0.0


def test_any_element_matches_and_count_elements_matching_use_visible_properties():
    obs = _make_obs(
        [
            Element(
                id="rect42",
                type="rect",
                label="rect",
                value={"x": "10", "y": "20", "width": "100", "height": "50", "style": "fill:#0000ff"},
            ),
            Element(
                id="title",
                type="text",
                label="title",
                value={"text_content": "SVG"},
            ),
            Element(
                id="layer2",
                type="layer",
                label="Annotations",
                value={"label": "Annotations"},
            ),
        ]
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "visible_shapes",
                    "checkpoints": [
                        {
                            "id": "blue_rect",
                            "weight": 0.4,
                            "rule": {
                                "any_element_matches": {
                                    "type": "rect",
                                    "value": {
                                        "x": "10",
                                        "y": "20",
                                        "width": "100",
                                        "height": "50",
                                        "style": "fill:#0000ff",
                                    },
                                }
                            },
                        },
                        {
                            "id": "svg_text",
                            "weight": 0.3,
                            "rule": {
                                "any_element_matches": {
                                    "type": "text",
                                    "value": {"text_content": "SVG"},
                                }
                            },
                        },
                        {
                            "id": "one_annotations_layer",
                            "weight": 0.3,
                            "rule": {
                                "count_elements_matching": {
                                    "type": "layer",
                                    "value": {"label": "Annotations"},
                                    "expected": 1,
                                }
                            },
                        },
                    ],
                }
            ]
        },
    )

    assert report.success is True
    assert report.score == 1.0


def test_no_element_matches_and_app_view_rules_prevent_vacuous_passes():
    obs = _make_obs(
        [
            Element(
                id="repo:asil_admin/test-repo",
                type="repository",
                label="asil_admin/test-repo",
                value={"name": "test-repo"},
            )
        ],
        current_view="explore/repos",
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "query_result",
                    "checkpoints": [
                        {"id": "repos_page", "weight": 0.5, "rule": {"app_view": "explore/repos"}},
                        {
                            "id": "circle_removed",
                            "weight": 0.5,
                            "rule": {
                                "no_element_matches": {
                                    "type": "circle",
                                    "value": {"cx": "200", "cy": "150", "r": "40"},
                                }
                            },
                        },
                    ],
                }
            ]
        },
    )

    assert report.success is True


def test_spreadsheet_header_rule_matches_exact_header_sequence():
    obs = _make_sheet_obs(
        {
            "Sheet1!A1": ("Date", "string"),
            "Sheet1!B1": ("Description", "string"),
            "Sheet1!C1": ("Category", "string"),
            "Sheet1!D1": ("Amount", "string"),
        }
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "expense_headers",
                    "checkpoints": [
                        {
                            "id": "headers",
                            "weight": 1.0,
                            "rule": {
                                "spreadsheet_headers": {
                                    "sheet": "Sheet1",
                                    "row": 1,
                                    "expected": ["Date", "Description", "Category", "Amount"],
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert report.success is True


def test_spreadsheet_complete_rows_rule_rejects_partial_rows():
    obs = _make_sheet_obs(
        {
            "Sheet1!A1": ("ID", "string"),
            "Sheet1!B1": ("Task", "string"),
            "Sheet1!C1": ("Owner", "string"),
            "Sheet1!A2": ("T-001", "string"),
            "Sheet1!B2": ("Plan", "string"),
            "Sheet1!C2": ("Alice", "string"),
            "Sheet1!A3": ("T-002", "string"),
            "Sheet1!B3": ("Build", "string"),
        }
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "task_rows",
                    "checkpoints": [
                        {
                            "id": "rows",
                            "weight": 1.0,
                            "rule": {
                                "spreadsheet_complete_rows": {
                                    "sheet": "Sheet1",
                                    "columns": ["A", "B", "C"],
                                    "start_row": 2,
                                    "expected_rows": 2,
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert report.success is False


def test_spreadsheet_column_type_rule_supports_dates_numbers_and_ranges():
    obs = _make_sheet_obs(
        {
            "Sheet1!A1": ("Student", "string"),
            "Sheet1!B1": ("Math", "string"),
            "Sheet1!C1": ("English", "string"),
            "Sheet1!A2": ("Alice", "string"),
            "Sheet1!B2": ("95", "float"),
            "Sheet1!C2": ("88", "float"),
            "Sheet1!A3": ("Bob", "string"),
            "Sheet1!B3": ("76", "float"),
            "Sheet1!C3": ("91", "float"),
        }
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "grade_sheet",
                    "checkpoints": [
                        {
                            "id": "scores",
                            "weight": 1.0,
                            "rule": {
                                "spreadsheet_column_type": {
                                    "sheet": "Sheet1",
                                    "columns": ["B", "C"],
                                    "start_row": 2,
                                    "end_row": 3,
                                    "expected": "number",
                                    "min": 0,
                                    "max": 100,
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert report.success is True


def test_spreadsheet_column_sequence_and_uniqueness_rules():
    obs = _make_sheet_obs(
        {
            "Sheet1!A1": ("Month", "string"),
            "Sheet1!B1": ("Sales", "string"),
            "Sheet1!A2": ("Jan", "string"),
            "Sheet1!B2": ("1000", "float"),
            "Sheet1!A3": ("Feb", "string"),
            "Sheet1!B3": ("1100", "float"),
            "Sheet1!A4": ("Mar", "string"),
            "Sheet1!B4": ("1200", "float"),
        }
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "monthly_sales",
                    "checkpoints": [
                        {
                            "id": "months",
                            "weight": 0.5,
                            "rule": {
                                "spreadsheet_column_sequence": {
                                    "sheet": "Sheet1",
                                    "column": "A",
                                    "start_row": 2,
                                    "expected": ["Jan", "Feb", "Mar"],
                                }
                            },
                        },
                        {
                            "id": "unique_months",
                            "weight": 0.5,
                            "rule": {
                                "spreadsheet_column_unique": {
                                    "sheet": "Sheet1",
                                    "column": "A",
                                    "start_row": 2,
                                    "end_row": 4,
                                }
                            },
                        },
                    ],
                }
            ]
        },
    )

    assert report.success is True


def test_spreadsheet_row_numeric_relation_checks_formula_like_constraint():
    obs = _make_sheet_obs(
        {
            "Sheet1!D2": ("15000", "float"),
            "Sheet1!E2": ("3000", "float"),
            "Sheet1!F2": ("2000", "float"),
            "Sheet1!G2": ("1800", "float"),
            "Sheet1!H2": ("1200", "float"),
            "Sheet1!I2": ("17000", "float"),
            "Sheet1!D3": ("12000", "float"),
            "Sheet1!E3": ("2500", "float"),
            "Sheet1!F3": ("1500", "float"),
            "Sheet1!G3": ("1500", "float"),
            "Sheet1!H3": ("1000", "float"),
            "Sheet1!I3": ("13500", "float"),
        }
    )

    report = evaluate_observation(
        obs,
        evaluator={
            "paths": [
                {
                    "path_id": "payroll_net_pay",
                    "checkpoints": [
                        {
                            "id": "net_pay_relation",
                            "weight": 1.0,
                            "rule": {
                                "spreadsheet_row_numeric_relation": {
                                    "sheet": "Sheet1",
                                    "start_row": 2,
                                    "end_row": 3,
                                    "target_column": "I",
                                    "add_columns": ["D", "E", "F"],
                                    "subtract_columns": ["G", "H"],
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert report.success is True
