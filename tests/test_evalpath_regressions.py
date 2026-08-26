import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples"


def _task(software: str, task_id: str) -> dict:
    return json.loads((ROOT / software / f"{task_id}.json").read_text(encoding="utf-8"))


def _checkpoints(task: dict) -> list[dict]:
    return task["evaluator"]["paths"][0]["checkpoints"]


def _rules(task: dict) -> list[dict]:
    return [checkpoint["rule"] for checkpoint in _checkpoints(task)]


def _has_element_value(task: dict, element_id: str, key: str, expected=None) -> bool:
    for rule in _rules(task):
        spec = rule.get("element_value")
        if not isinstance(spec, dict):
            continue
        if spec.get("id") == element_id and spec.get("key") == key:
            if expected is None or spec.get("expected") == expected:
                return True
    return False


def _has_any_element_matches(task: dict, element_type: str, value: dict) -> bool:
    for rule in _rules(task):
        spec = rule.get("any_element_matches")
        if not isinstance(spec, dict):
            continue
        if spec.get("type") != element_type:
            continue
        rule_value = spec.get("value")
        if isinstance(rule_value, dict) and all(rule_value.get(key) == expected for key, expected in value.items()):
            return True
    return False


def _has_count_match(task: dict, element_type: str, value: dict, expected: int) -> bool:
    for rule in _rules(task):
        spec = rule.get("count_elements_matching")
        if not isinstance(spec, dict):
            continue
        if spec.get("type") != element_type or spec.get("expected") != expected:
            continue
        rule_value = spec.get("value")
        if isinstance(rule_value, dict) and all(rule_value.get(key) == wanted for key, wanted in value.items()):
            return True
    return False


def test_blender_10_is_single_path_and_no_longer_uses_cross_scoring_scene_counts():
    task = _task("blender", "blender_10")

    assert len(task["evaluator"]["paths"]) == 1
    assert not any("scene_object_count" in rule for rule in _rules(task))
    assert any("any_element_matches" in rule for rule in _rules(task))


def test_nautilus_12_preserves_the_source_file_for_copy_semantics():
    task = _task("nautilus", "nautilus_12")

    assert _has_any_element_matches(task, "directory_entry", {"path": "Inbox/release-plan.md"})
    assert _has_any_element_matches(task, "workspace_entry", {"path": "Projects/alpha/release-plan.md"})
    assert not any(
        rule.get("count_elements_matching", {}).get("value", {}).get("path") == "Projects/alpha/release-plan.md"
        and rule["count_elements_matching"].get("expected") == 0
        for rule in _rules(task)
        if "count_elements_matching" in rule
    )


def test_thunderbird_tasks_require_exact_body_and_open_compose_window():
    for index in range(1, 21):
        task = _task("thunderbird", f"thunderbird_{index:02d}")
        checkpoints = _checkpoints(task)
        ids = [checkpoint["id"] for checkpoint in checkpoints]

        assert ids == ["draft_to", "draft_subject", "draft_body", "draft_open"]
        assert _has_element_value(task, "compose:draft", "to")
        assert _has_element_value(task, "compose:draft", "subject")
        assert _has_element_value(task, "compose:draft", "body")
        assert _has_element_value(task, "compose:draft", "is_open", True)
        assert not any("element_contains" in rule for rule in _rules(task))


def test_drawio_complex_tasks_cover_every_explicit_shape_and_connector():
    expected_shapes = {
        "drawio_06": ["start", "review", "deploy"],
        "drawio_11": ["intake", "triage", "support"],
        "drawio_13": ["draft", "review", "approve", "release"],
        "drawio_16": ["ticket", "review", "escalate", "resolve"],
        "drawio_18": ["collect", "review", "update", "publish"],
        "drawio_19": ["start", "review", "ship", "archive"],
        "drawio_20": ["idea", "draft", "review", "approve", "release", "measure"],
    }
    expected_connectors = {
        "drawio_06": [
            {"source_label": "Start", "target_label": "Review", "label": "submit"},
            {"source_label": "Review", "target_label": "Deploy", "label": "approved"},
        ],
        "drawio_11": [
            {"source_label": "Intake", "target_label": "Triage", "label": "route"},
            {"source_label": "Triage", "target_label": "Support", "label": "assign"},
        ],
        "drawio_13": [
            {"source_label": "Draft", "target_label": "Review", "label": "submit"},
            {"source_label": "Review", "target_label": "Approve", "label": "ready"},
            {"source_label": "Approve", "target_label": "Release", "label": "ship"},
        ],
        "drawio_16": [
            {"source_label": "Ticket", "target_label": "Review", "label": "open"},
            {"source_label": "Review", "target_label": "Escalate", "label": "needs help"},
            {"source_label": "Review", "target_label": "Resolve", "label": "ready"},
        ],
        "drawio_18": [
            {"source_label": "Collect", "target_label": "Review", "label": "send"},
            {"source_label": "Review", "target_label": "Update", "label": "change"},
            {"source_label": "Update", "target_label": "Publish", "label": "ready"},
            {"source_label": "Publish", "target_label": "Collect", "label": "loop"},
        ],
        "drawio_19": [
            {"source_label": "Check", "target_label": "Ship", "label": "ship"},
            {"source_label": "Ship", "target_label": "Archive", "label": "archive"},
        ],
        "drawio_20": [
            {"source_label": "Idea", "target_label": "Draft", "label": "capture"},
            {"source_label": "Draft", "target_label": "Review", "label": "share"},
            {"source_label": "Review", "target_label": "Approve", "label": "ready"},
            {"source_label": "Approve", "target_label": "Release", "label": "go"},
            {"source_label": "Release", "target_label": "Measure", "label": "observe"},
        ],
    }

    for task_id, shape_ids in expected_shapes.items():
        task = _task("drawio", task_id)
        for shape_id in shape_ids:
            assert any(
                rule.get("element_exists") == f"shape:{shape_id}"
                or rule.get("element_value", {}).get("id") == f"shape:{shape_id}"
                for rule in _rules(task)
            ), task_id
        for connector in expected_connectors[task_id]:
            assert _has_any_element_matches(task, "connector", connector), (task_id, connector)


def test_obs_simple_tasks_do_not_require_unrelated_scene_or_program_state():
    no_scene_context = ("obs_04", "obs_05", "obs_07", "obs_12", "obs_13", "obs_15", "obs_17")
    for task_id in no_scene_context:
        task = _task("obs", task_id)
        assert not any("current_scene" in rule for rule in _rules(task)), task_id

    for task_id in ("obs_05", "obs_13"):
        task = _task("obs", task_id)
        assert not any("input_muted" in rule for rule in _rules(task)), task_id


def test_libreoffice_impress_20_checks_every_explicit_slide_change():
    task = _task("libreoffice_impress", "libreoffice_impress_20")

    assert _has_element_value(task, "slide:2:body:1", "text_content", "Confirm escalation routing.")
    assert _has_element_value(task, "slide:2:body:2", "text_content", "Review coverage with operations.")
    assert _has_element_value(task, "slide:3:title", "text_content", "Summary")
    assert _has_element_value(task, "slide:3:body:1", "text_content", "Hiring plans will be revisited next month.")


def test_libreoffice_writer_19_only_checks_the_final_updated_paragraph_state():
    task = _task("libreoffice_writer", "libreoffice_writer_19")

    assert _has_any_element_matches(task, "heading", {"text_content": "Announcements"})
    assert _has_any_element_matches(task, "paragraph", {"text_content": "Office hours move to Wednesday."})
    assert _has_element_value(task, "paragraph:2", "text_content", "Stakeholder feedback was collected.")
    assert not _has_any_element_matches(task, "paragraph", {"text_content": "The roadmap review is public now."})


def test_jupyterlab_09_checks_first_cell_type_instead_of_snapshot_specific_source():
    task = _task("jupyterlab", "jupyterlab_09")

    assert _has_element_value(task, "cell:notebooks/analysis.ipynb:0", "cell_type", "code")
    assert not any(
        rule.get("element_contains", {}).get("id") == "cell:notebooks/analysis.ipynb:0"
        for rule in _rules(task)
    )


def test_jupyterlab_16_and_17_cover_exact_editor_and_cell_state():
    checklist = _task("jupyterlab", "jupyterlab_16")
    assert _rules(checklist) == [
        {
            "element_value": {
                "id": "editor:notes/checklist.md",
                "key": "content",
                "expected": "# Checklist\n- Verify cells\n- Export summary\n",
            }
        }
    ]

    status = _task("jupyterlab", "jupyterlab_17")
    assert _rules(status)[0] == {
        "element_value": {
            "id": "cell:notebooks/analysis.ipynb:3",
            "key": "source",
            "expected": 'status = "ready"',
        }
    }


def test_gimp_reviewed_tasks_cover_geometry_and_final_layout_state():
    expected_pairs = {
        "gimp_13": {
            ("hero_banner", "x"), ("hero_banner", "y"), ("hero_banner", "width"), ("hero_banner", "height"),
            ("headline", "x"), ("headline", "y"), ("headline", "font_size"),
            ("accent", "x"), ("accent", "y"), ("accent", "width"), ("accent", "height"),
        },
        "gimp_14": {
            ("image", "width"), ("image", "height"),
            ("logo_block", "x"), ("logo_block", "y"), ("logo_block", "width"), ("logo_block", "height"),
        },
        "gimp_15": {
            ("spotlight", "x"), ("spotlight", "y"), ("spotlight", "width"), ("spotlight", "height"),
        },
        "gimp_16": {
            ("sale_sticker", "x"), ("sale_sticker", "y"), ("sale_sticker", "width"), ("sale_sticker", "height"),
            ("sticker_copy", "x"), ("sticker_copy", "y"), ("sticker_copy", "font_size"),
        },
        "gimp_17": {
            ("watermark_text", "x"), ("watermark_text", "y"), ("logo_block", "width"), ("logo_block", "height"),
        },
        "gimp_18": {
            ("thumb_bg", "width"), ("thumb_bg", "height"),
            ("thumb_glow", "x"), ("thumb_glow", "y"), ("thumb_glow", "width"), ("thumb_glow", "height"),
            ("thumb_copy", "x"), ("thumb_copy", "y"), ("thumb_copy", "font_size"),
        },
        "gimp_19": {
            ("image", "width"), ("image", "height"),
            ("logo_block", "x"), ("logo_block", "y"), ("logo_block", "width"), ("logo_block", "height"),
        },
        "gimp_20": {
            ("poster_bg", "x"), ("poster_bg", "y"), ("poster_bg", "width"), ("poster_bg", "height"),
            ("burst", "x"), ("burst", "y"), ("burst", "width"), ("burst", "height"),
            ("poster_title", "x"), ("poster_title", "y"), ("poster_title", "font_size"),
            ("poster_cta", "x"), ("poster_cta", "y"), ("poster_cta", "font_size"), ("poster_cta", "color"),
        },
    }

    for task_id, pairs in expected_pairs.items():
        task = _task("gimp", task_id)
        covered = {
            (rule["element_value"]["id"], rule["element_value"]["key"])
            for rule in _rules(task)
            if "element_value" in rule
        }
        assert pairs.issubset(covered), (task_id, sorted(pairs - covered))


def test_inkscape_reviewed_tasks_use_exact_id_based_checks_from_validation():
    for index in range(12, 21):
        task = _task("inkscape", f"inkscape_{index:02d}")
        checkpoints = _checkpoints(task)
        validation_conditions = task["_asil"]["validation"]["conditions"]

        assert len(checkpoints) >= len(validation_conditions)
        assert all(
            any(key in checkpoint["rule"] for key in ("element_value", "element_exists"))
            for checkpoint in checkpoints
        ), task["id"]
        assert not any(
            any(key in checkpoint["rule"] for key in ("any_element_matches", "count_elements_matching"))
            for checkpoint in checkpoints
        ), task["id"]
