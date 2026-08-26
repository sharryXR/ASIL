import json
from pathlib import Path

from asil.eval.task_audit import audit_task_definition, audit_task_file, audit_task_tree


def test_audit_rejects_task_without_gui_expectations():
    task = {
        "id": "t1",
        "instruction": "Create rectangle",
        "evaluator": {
            "paths": [{"path_id": "p1", "checkpoints": [{"id": "cp1", "weight": 1.0, "rule": {"element_exists": "r1"}}]}]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("gui_expectations" in err for err in report.errors)


def test_audit_rejects_task_without_paths():
    task = {
        "id": "t1",
        "instruction": "Create rectangle",
        "gui_expectations": {
            "success_surface": "canvas",
            "visible_change_summary": "A rectangle appears on the canvas",
        },
        "evaluator": {"func": "element_exists", "expected": {"value": "r1"}},
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("paths" in err for err in report.errors)


def test_audit_rejects_missing_visibility_mapping_for_required_checkpoint():
    task = {
        "id": "t1",
        "instruction": "Create rectangle",
        "gui_expectations": {
            "success_surface": "canvas",
            "visible_change_summary": "A rectangle appears on the canvas",
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "p1",
                    "checkpoints": [
                        {
                            "id": "rect_visible",
                            "weight": 1.0,
                            "gui_visible_required": True,
                            "rule": {"element_exists": "r1"},
                        }
                    ],
                }
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("checkpoint_visibility" in err for err in report.errors)


def test_audit_rejects_hidden_gui_only_checkpoint():
    task = {
        "id": "t_hidden",
        "instruction": "Resize the hidden drawio canvas metadata.",
        "gui_expectations": {
            "success_surface": "diagram_canvas",
            "visible_change_summary": "The diagram changes.",
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
                            "rule": {"element_value": {"id": "canvas", "key": "width", "expected": 1200}},
                        }
                    ],
                }
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("hidden-only state" in err for err in report.errors)


def test_audit_rejects_duplicate_checkpoint_ids_within_path():
    task = {
        "id": "t1",
        "instruction": "Create rectangle",
        "gui_expectations": {
            "success_surface": "canvas",
            "visible_change_summary": "A rectangle appears on the canvas",
            "checkpoint_visibility": {"rect_visible": "visible_on_canvas"},
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "p1",
                    "checkpoints": [
                        {
                            "id": "rect_visible",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {"element_exists": "r1"},
                        },
                        {
                            "id": "rect_visible",
                            "weight": 0.5,
                            "rule": {"element_value": {"id": "r1", "key": "fill", "expected": "blue"}},
                        },
                    ],
                }
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("Duplicate checkpoint id" in err for err in report.errors)


def test_audit_rejects_overlapping_exclusive_paths():
    task = {
        "id": "t1",
        "instruction": "Create rectangle",
        "gui_expectations": {
            "success_surface": "canvas",
            "visible_change_summary": "A rectangle appears on the canvas",
            "checkpoint_visibility": {
                "rect_created": "visible_on_canvas",
                "rect_created_alt": "visible_on_canvas",
            },
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "p1",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "rect_created",
                            "weight": 1.0,
                            "gui_visible_required": True,
                            "rule": {"element_exists": "r1"},
                        }
                    ],
                },
                {
                    "path_id": "p2",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "rect_created_alt",
                            "weight": 1.0,
                            "gui_visible_required": True,
                            "rule": {"element_exists": "r1"},
                        }
                    ],
                },
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("exclusive_group" in err for err in report.errors)


def test_audit_accepts_well_formed_path_task():
    task = {
        "id": "t1",
        "instruction": "Create rectangle",
        "gui_expectations": {
            "success_surface": "canvas",
            "visible_change_summary": "A blue rectangle appears on the canvas",
            "checkpoint_visibility": {
                "rect_visible": "visible_on_canvas",
                "rect_blue": "visible_on_canvas",
            },
        },
        "evaluator": {
            "selection": "best_score",
            "paths": [
                {
                    "path_id": "main",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "rect_visible",
                            "weight": 0.6,
                            "gui_visible_required": True,
                            "rule": {"element_exists": "r1"},
                        },
                        {
                            "id": "rect_blue",
                            "weight": 0.4,
                            "gui_visible_required": True,
                            "rule": {"element_value": {"id": "r1", "key": "fill", "expected": "blue"}},
                        },
                    ],
                }
            ],
        },
    }

    report = audit_task_definition(task)

    assert report.ok is True
    assert report.errors == []


def test_audit_accepts_well_formed_multi_app_task():
    task = {
        "id": "multi_apps_001",
        "software": "multi_apps",
        "instruction": "Update code and notebook.",
        "related_apps": ["code_server", "jupyterlab"],
        "gui_expectations": {
            "success_surface": "multi_window",
            "visible_change_summary": "Both apps show their updates.",
            "checkpoint_visibility": {
                "code": "visible_in_code_server",
                "notebook": "visible_in_jupyterlab",
            },
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "main",
                    "checkpoints": [
                        {
                            "id": "code",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {
                                "app_rule": {
                                    "app": "code_server",
                                    "rule": {"element_exists": "file:reports/summary.md"},
                                }
                            },
                        },
                        {
                            "id": "notebook",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {
                                "app_rule": {
                                    "app": "jupyterlab",
                                    "rule": {"element_exists": "file:reports/summary.md"},
                                }
                            },
                        },
                    ],
                }
            ]
        },
        "_asil": {
            "software": "multi_apps",
            "app_initial_states": {"code_server": "default", "jupyterlab": "default"},
            "primary_app": "code_server",
        },
    }

    report = audit_task_definition(task)

    assert report.ok is True
    assert report.errors == []


def test_audit_rejects_multi_app_task_outside_full15():
    task = {
        "id": "multi_apps_bad",
        "software": "multi_apps",
        "instruction": "Use an unsupported app.",
        "related_apps": ["code_server", "vlc"],
        "gui_expectations": {
            "success_surface": "multi_window",
            "visible_change_summary": "A visible update.",
            "checkpoint_visibility": {"code": "visible_in_code_server"},
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "main",
                    "checkpoints": [
                        {
                            "id": "code",
                            "weight": 1.0,
                            "gui_visible_required": True,
                            "rule": {"app_rule": {"app": "code_server", "rule": {"element_exists": "file:a.md"}}},
                        }
                    ],
                }
            ]
        },
        "_asil": {
            "software": "multi_apps",
            "app_initial_states": {"code_server": "default", "vlc": "default"},
            "primary_app": "vlc",
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("non-full15" in err for err in report.errors)


def test_audit_rejects_global_scene_counts_inside_multi_path_exclusive_group():
    task = {
        "id": "blender_room_paths",
        "instruction": "Animate either cube variant over the same timeline.",
        "gui_expectations": {
            "success_surface": "viewport_and_timeline",
            "visible_change_summary": "A cube animation appears.",
            "checkpoint_visibility": {
                "cube_a": "visible_in_viewport",
                "cube_b": "visible_in_viewport",
                "count_a": "visible_in_viewport",
                "count_b": "visible_in_viewport",
            },
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "cube_a_path",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "cube_a",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {"object_exists": "Cube.001"},
                        },
                        {
                            "id": "count_a",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {"scene_object_count": 4},
                        },
                    ],
                },
                {
                    "path_id": "cube_b_path",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "cube_b",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {"object_exists": "Cube"},
                        },
                        {
                            "id": "count_b",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {"scene_object_count": 3},
                        },
                    ],
                },
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("global aggregate checkpoint" in err for err in report.errors)


def test_audit_rejects_copy_tasks_that_require_source_to_disappear():
    task = {
        "id": "nautilus_copy_conflict",
        "instruction": "Copy release-plan.md to Inbox and open Inbox.",
        "gui_expectations": {
            "success_surface": "file_list",
            "visible_change_summary": "Inbox shows a copied file.",
            "checkpoint_visibility": {
                "copy_visible": "visible_in_file_list",
                "location": "visible_in_location_bar",
            },
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "main",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "location",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {"element_value": {"id": "location", "key": "path", "expected": "Inbox"}},
                        },
                        {
                            "id": "copy_visible",
                            "weight": 0.25,
                            "gui_visible_required": True,
                            "rule": {
                                "any_element_matches": {
                                    "type": "directory_entry",
                                    "value": {"path": "Inbox/release-plan.md"},
                                }
                            },
                        },
                        {
                            "id": "source_removed",
                            "weight": 0.25,
                            "gui_visible_required": False,
                            "rule": {
                                "count_elements_matching": {
                                    "type": "directory_entry",
                                    "value": {"path": "Projects/alpha/release-plan.md"},
                                    "expected": 0,
                                }
                            },
                        },
                    ],
                }
            ]
        },
        "_asil": {
            "actions": [
                {
                    "action_type": "invoke_function",
                    "target": "nautilus",
                    "params": {
                        "operations": [
                            {"action": "copy_entry", "path": "release-plan.md", "destination_dir": "../../Inbox"},
                            {"action": "open_directory", "path": "../../Inbox"},
                        ]
                    },
                }
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("copy semantics" in err for err in report.errors)


def test_audit_rejects_updated_paragraph_tasks_that_also_require_old_text():
    task = {
        "id": "writer_update_conflict",
        "instruction": 'Add "New copy." and then update the second paragraph to "Updated copy.".',
        "gui_expectations": {
            "success_surface": "document_page",
            "visible_change_summary": "A paragraph is updated.",
            "checkpoint_visibility": {
                "old_text": "visible_on_page",
                "new_text": "visible_on_page",
            },
        },
        "evaluator": {
            "paths": [
                {
                    "path_id": "main",
                    "exclusive_group": "success",
                    "checkpoints": [
                        {
                            "id": "old_text",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {
                                "any_element_matches": {
                                    "type": "paragraph",
                                    "value": {"text_content": "New copy."},
                                }
                            },
                        },
                        {
                            "id": "new_text",
                            "weight": 0.5,
                            "gui_visible_required": True,
                            "rule": {
                                "element_value": {
                                    "id": "paragraph:2",
                                    "key": "text_content",
                                    "expected": "Updated copy.",
                                }
                            },
                        },
                    ],
                }
            ]
        },
        "_asil": {
            "actions": [
                {
                    "action_type": "modify_file",
                    "target": "writer",
                    "params": {
                        "operations": [
                            {"action": "add_paragraph", "text": "New copy."},
                            {"action": "set_paragraph_text", "index": 2, "text": "Updated copy."},
                        ]
                    },
                }
            ]
        },
    }

    report = audit_task_definition(task)

    assert report.ok is False
    assert any("pre-update paragraph text" in err for err in report.errors)


def test_audit_task_file_reports_path(tmp_path: Path):
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "id": "t1",
                "instruction": "Create rectangle",
                "gui_expectations": {
                    "success_surface": "canvas",
                    "visible_change_summary": "A rectangle appears on the canvas",
                    "checkpoint_visibility": {"rect_visible": "visible_on_canvas"},
                },
                "evaluator": {
                    "paths": [
                        {
                            "path_id": "main",
                            "checkpoints": [
                                {
                                    "id": "rect_visible",
                                    "weight": 1.0,
                                    "gui_visible_required": True,
                                    "rule": {"element_exists": "r1"},
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_task_file(task_file)

    assert report.path == task_file
    assert report.ok is True


def test_audit_task_tree_aggregates_reports(tmp_path: Path):
    ok_dir = tmp_path / "inkscape"
    ok_dir.mkdir()
    bad_dir = tmp_path / "obs"
    bad_dir.mkdir()

    (ok_dir / "ink_01.json").write_text(
        json.dumps(
            {
                "id": "ink_01",
                "instruction": "Create rectangle",
                "gui_expectations": {
                    "success_surface": "canvas",
                    "visible_change_summary": "A rectangle appears on the canvas",
                    "checkpoint_visibility": {"rect_visible": "visible_on_canvas"},
                },
                "evaluator": {
                    "paths": [
                        {
                            "path_id": "main",
                            "checkpoints": [
                                {
                                    "id": "rect_visible",
                                    "weight": 1.0,
                                    "gui_visible_required": True,
                                    "rule": {"element_exists": "r1"},
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (bad_dir / "obs_01.json").write_text(
        json.dumps({"id": "obs_01", "instruction": "Switch scene", "evaluator": {"func": "current_scene"}}),
        encoding="utf-8",
    )

    reports = audit_task_tree(tmp_path)

    assert len(reports) == 2
    assert sum(report.ok for report in reports) == 1
    assert sum(not report.ok for report in reports) == 1


def test_audit_task_tree_supports_single_software_directory(tmp_path: Path):
    (tmp_path / "libreoffice_01.json").write_text(
        json.dumps(
            {
                "id": "libreoffice_01",
                "instruction": "Set A1",
                "gui_expectations": {
                    "success_surface": "spreadsheet",
                    "visible_change_summary": "Cell A1 changes visibly",
                    "checkpoint_visibility": {"a1": "visible_in_sheet"},
                },
                "evaluator": {
                    "paths": [
                        {
                            "path_id": "main",
                            "checkpoints": [
                                {
                                    "id": "a1",
                                    "weight": 1.0,
                                    "gui_visible_required": True,
                                    "rule": {"element_value": {"id": "Sheet1!A1", "key": None, "expected": "X"}},
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    reports = audit_task_tree(tmp_path)

    assert len(reports) == 1
    assert reports[0].ok is True
