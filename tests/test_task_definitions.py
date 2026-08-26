import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "evaluation_examples" / "examples"


def _task(software: str, task_id: str) -> dict:
    return json.loads((ROOT / software / f"{task_id}.json").read_text(encoding="utf-8"))


def test_blender_room_and_product_tasks_do_not_depend_on_default_scene_residue():
    room = _task("blender", "blender_18")
    product = _task("blender", "blender_20")

    assert room["_asil"]["initial_state"] == "blank"
    assert product["_asil"]["initial_state"] == "blank"
    room_rules = [cp["rule"] for cp in room["evaluator"]["paths"][0]["checkpoints"]]
    product_rules = [cp["rule"] for cp in product["evaluator"]["paths"][0]["checkpoints"]]
    assert {"count_elements_matching": {"type": "mesh", "expected": 5}} in room_rules
    assert {"count_elements_matching": {"type": "camera", "expected": 1}} in room_rules
    assert {"count_elements_matching": {"type": "light", "value": {"light_type": "AREA"}, "expected": 3}} in product_rules
    assert {"count_elements_matching": {"type": "camera", "expected": 1}} in product_rules


def test_blender_09_uses_visible_object_counts_instead_of_hidden_layout_names():
    task = _task("blender", "blender_09")
    rules = [cp["rule"] for cp in task["evaluator"]["paths"][0]["checkpoints"]]

    assert {"count_elements_matching": {"type": "mesh", "expected": 3}} in rules
    assert {"count_elements_matching": {"type": "camera", "expected": 1}} in rules
    assert {"count_elements_matching": {"type": "light", "value": {"light_type": "SUN"}, "expected": 1}} in rules


def test_gitea_01_does_not_require_extra_description_checkpoint():
    task = _task("gitea", "gitea_01")
    checkpoints = task["evaluator"]["paths"][0]["checkpoints"]

    assert [cp["id"] for cp in checkpoints] == ["gitea_01_1", "gitea_01_2"]
    assert "description" not in str(checkpoints)


def test_obs_tasks_remove_context_only_checkpoint_scoring_and_require_live_controls():
    simple_ids = ["obs_02", "obs_06", "obs_08", "obs_14", "obs_16", "obs_18"]
    for task_id in simple_ids:
        task = _task("obs", task_id)
        checkpoints = task["evaluator"]["paths"][0]["checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["weight"] == 1.0

    obs_20 = _task("obs", "obs_20")
    rules = [cp["rule"] for cp in obs_20["evaluator"]["paths"][0]["checkpoints"]]
    assert {"element_value": {"id": "studio_mode", "key": "enabled", "expected": True}} in rules
    assert {"element_value": {"id": "preview_scene", "key": "name", "expected": "BRB"}} in rules


def test_libreoffice_closed_set_tasks_embed_exact_dataset_in_instruction():
    task = _task("libreoffice", "libreoffice_06")

    assert "Jan=12500" in task["instruction"]
    assert "Jun=18200" in task["instruction"]
    rules = [cp["rule"] for cp in task["evaluator"]["paths"][0]["checkpoints"]]
    assert {"element_value": {"id": "Sheet1!B2", "key": None, "expected": "12500"}} in rules


def test_libreoffice_open_task_uses_constraint_rules_instead_of_hidden_values():
    task = _task("libreoffice", "libreoffice_14")
    rules = [cp["rule"] for cp in task["evaluator"]["paths"][0]["checkpoints"]]

    assert any("spreadsheet_headers" in rule for rule in rules)
    assert any("spreadsheet_complete_rows" in rule for rule in rules)
    assert any("spreadsheet_column_type" in rule for rule in rules)
    assert not any(
        rule.get("element_value", {}).get("id") == "Sheet1!B2"
        and rule.get("element_value", {}).get("expected") == "Math"
        for rule in rules
    )


def test_libreoffice_payroll_task_aligns_headers_and_formula_with_instruction():
    task = _task("libreoffice", "libreoffice_19")
    rules = [cp["rule"] for cp in task["evaluator"]["paths"][0]["checkpoints"]]

    assert "Position Allowance" in task["instruction"]
    assert "Performance Bonus" in task["instruction"]
    assert "Net Pay should equal Base Salary + Position Allowance + Performance Bonus - Insurance - Tax" in task["instruction"]
    assert {"element_value": {"id": "Sheet1!E1", "key": None, "expected": "Allowance"}} not in rules
    assert any("spreadsheet_row_numeric_relation" in rule for rule in rules)


def test_inkscape_tasks_use_visible_property_rules_instead_of_hidden_ids():
    rect_task = _task("inkscape", "inkscape_01")
    layer_task = _task("inkscape", "inkscape_05")
    icon_task = _task("inkscape", "inkscape_10")

    rect_rules = [cp["rule"] for cp in rect_task["evaluator"]["paths"][0]["checkpoints"]]
    layer_rules = [cp["rule"] for cp in layer_task["evaluator"]["paths"][0]["checkpoints"]]
    icon_rules = [cp["rule"] for cp in icon_task["evaluator"]["paths"][0]["checkpoints"]]

    assert any("any_element_matches" in rule for rule in rect_rules)
    assert any(
        rule.get("any_element_matches", {}).get("type") == "layer"
        and rule.get("any_element_matches", {}).get("value", {}).get("label") == "Annotations"
        for rule in layer_rules
    )
    assert all("element_exists" not in rule for rule in icon_rules)


def test_blender_tasks_remove_default_scene_count_bias_and_hidden_names():
    simple_cube = _task("blender", "blender_01")
    room = _task("blender", "blender_18")
    product = _task("blender", "blender_20")

    assert simple_cube["_asil"]["initial_state"] == "blank"

    cube_rules = [cp["rule"] for cp in simple_cube["evaluator"]["paths"][0]["checkpoints"]]
    assert all("scene_object_count" not in rule for rule in cube_rules)
    assert any("count_elements_matching" in rule for rule in cube_rules)

    room_rules = [cp["rule"] for cp in room["evaluator"]["paths"][0]["checkpoints"]]
    product_rules = [cp["rule"] for cp in product["evaluator"]["paths"][0]["checkpoints"]]
    assert any(
        rule.get("count_elements_matching", {}).get("type") == "mesh"
        for rule in room_rules
    )
    assert any(
        rule.get("count_elements_matching", {}).get("value", {}).get("light_type") == "AREA"
        for rule in product_rules
    )


def test_obs_vacuous_tasks_start_from_non_success_states():
    expected_states = {
        "obs_01": "scene_intermission",
        "obs_07": "recording_active",
        "obs_12": "mic_muted",
        "obs_15": "streaming_active",
    }

    for task_id, initial_state in expected_states.items():
        task = _task("obs", task_id)
        assert task["_asil"]["initial_state"] == initial_state


def test_gitea_tasks_remove_hidden_answers_and_brittle_merge_proxy():
    create_repo = _task("gitea", "gitea_01")
    list_repos = _task("gitea", "gitea_03")
    pr_task = _task("gitea", "gitea_06")
    merge_task = _task("gitea", "gitea_18")
    milestone_task = _task("gitea", "gitea_19")

    assert "test-repo" not in create_repo["instruction"]

    list_rules = [cp["rule"] for cp in list_repos["evaluator"]["paths"][0]["checkpoints"]]
    assert {"app_view": "explore/repos"} in list_rules
    assert not any("repo:asil_admin/old-project" in str(rule) for rule in list_rules)

    pr_rules = [cp["rule"] for cp in pr_task["evaluator"]["paths"][0]["checkpoints"]]
    assert not any(
        rule.get("element_contains", {}).get("key") in {"title", "body"}
        for rule in pr_rules
    )

    merge_rules = [cp["rule"] for cp in merge_task["evaluator"]["paths"][0]["checkpoints"]]
    assert {"element_not_exists": "pr:1"} not in merge_rules
    assert any("any_element_value" in rule for rule in merge_rules)

    milestone_rules = [cp["rule"] for cp in milestone_task["evaluator"]["paths"][0]["checkpoints"]]
    assert not any(
        rule.get("element_contains", {}).get("key") == "description"
        for rule in milestone_rules
    )


def test_drawio_tasks_remove_hidden_connector_ids_and_canvas_property_checkpoints():
    connector_tasks = [
        "drawio_01",
        "drawio_06",
        "drawio_07",
        "drawio_11",
        "drawio_13",
        "drawio_15",
        "drawio_16",
        "drawio_17",
        "drawio_18",
        "drawio_19",
        "drawio_20",
    ]

    resized_task = _task("drawio", "drawio_03")
    resized_rules = [cp["rule"] for cp in resized_task["evaluator"]["paths"][0]["checkpoints"]]
    assert not any(
        rule.get("element_value", {}).get("id") == "canvas"
        and rule.get("element_value", {}).get("key") in {"width", "height"}
        for rule in resized_rules
    )

    for task_id in connector_tasks:
        task = _task("drawio", task_id)
        rules = [cp["rule"] for cp in task["evaluator"]["paths"][0]["checkpoints"]]
        assert not any(
            rule.get("element_value", {}).get("id", "").startswith("connector:")
            and rule.get("element_value", {}).get("key") in {"source", "target"}
            for rule in rules
        ), task_id
        assert not any(
            rule.get("element_value", {}).get("id") == "canvas"
            and rule.get("element_value", {}).get("key") in {"width", "height"}
            for rule in rules
        ), task_id


def test_jupyterlab_tasks_make_insert_position_zero_based_explicit():
    task_08 = _task("jupyterlab", "jupyterlab_08")
    task_17 = _task("jupyterlab", "jupyterlab_17")

    assert "zero-based position 2" in task_08["instruction"]
    assert "zero-based position 3" in task_17["instruction"]


def test_kdenlive_profile_tasks_are_replaced_with_timeline_visible_tasks():
    for task_id in ("kdenlive_07", "kdenlive_08", "kdenlive_19"):
        task = _task("kdenlive", task_id)
        assert task["gui_expectations"]["success_surface"] == "timeline"
        rules = [cp["rule"] for cp in task["evaluator"]["paths"][0]["checkpoints"]]
        assert not any(
            rule.get("element_value", {}).get("id") == "project_settings"
            for rule in rules
        ), task_id
