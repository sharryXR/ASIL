"""Tests for the evaluation runner and metrics."""

import json
import tempfile
import shutil
from pathlib import Path

from asil.eval.metrics import TaskResult, StepResult, compute_metrics
from asil.eval.runner import TaskDefinition, run_evaluation, run_task


def test_task_definition_from_dict():
    td = TaskDefinition(
        id="inkscape_01",
        software="inkscape",
        difficulty="simple",
        description="Create a blue rectangle",
        actions=[
            {"action_type": "modify_file", "target": "test.svg", "params": {
                "operations": [{"action": "add_element", "parent_xpath": "//svg:svg",
                                "tag": "rect", "attributes": {"id": "r1", "width": "100", "height": "50", "style": "fill:blue"}}]
            }}
        ],
        validation={"element_exists": "r1"},
    )
    assert td.id == "inkscape_01"
    assert td.difficulty == "simple"
    assert len(td.actions) == 1
    # instruction defaults to description
    assert td.instruction == "Create a blue rectangle"


def test_task_definition_instruction():
    td = TaskDefinition(
        id="t1", software="inkscape", difficulty="simple",
        description="desc", instruction="Do something specific",
    )
    assert td.instruction == "Do something specific"


def test_task_result_creation():
    result = TaskResult(
        task_id="inkscape_01",
        success=True,
        score=1.0,
        steps=1,
        step_results=[
            StepResult(step_num=1, action_type="modify_file", target="x.svg", latency_ms=120.5),
        ],
        e2e_time_s=0.5,
        deadlocked=False,
        observation_element_count=5,
        total_element_count=5,
    )
    assert result.success is True
    assert result.coverage == 1.0
    assert result.avg_latency_ms == 120.5
    assert result.score == 1.0


def test_task_result_trajectory():
    result = TaskResult(
        task_id="t1",
        success=True,
        score=1.0,
        steps=2,
        step_results=[
            StepResult(step_num=1, action_type="set_value", target="r1", latency_ms=50),
            StepResult(step_num=2, action_type="modify_file", target="x.svg", latency_ms=80),
        ],
    )
    d = result.to_dict()
    assert len(d["trajectory"]) == 2
    assert d["trajectory"][0]["step_num"] == 1
    assert d["trajectory"][1]["latency_ms"] == 80


def test_task_result_save_trajectory(tmp_path):
    result = TaskResult(
        task_id="t1", success=True, score=1.0, steps=1,
        step_results=[
            StepResult(step_num=1, action_type="modify_file", target="x.svg", latency_ms=50),
        ],
    )
    out = tmp_path / "traj"
    result.save_trajectory(out)
    result.save_result(out)

    traj_lines = (out / "traj.jsonl").read_text().strip().split("\n")
    assert len(traj_lines) == 1
    entry = json.loads(traj_lines[0])
    assert entry["step_num"] == 1
    assert entry["action"]["action_type"] == "modify_file"

    score_text = (out / "result.txt").read_text()
    assert score_text == "1.0"


def test_compute_metrics_single_task():
    results = [
        TaskResult(
            task_id="t1", software="inkscape", success=True, score=1.0, steps=2,
            step_results=[
                StepResult(step_num=1, action_type="set_value", target="r1", latency_ms=100),
                StepResult(step_num=2, action_type="modify_file", target="x.svg", latency_ms=150),
            ],
            e2e_time_s=1.0, deadlocked=False,
            observation_element_count=10, total_element_count=10,
        ),
    ]
    m = compute_metrics(results)
    assert m["aggregate"]["success_rate"] == 1.0
    assert m["aggregate"]["avg_latency_ms"] == 125.0
    assert m["aggregate"]["avg_e2e_s"] == 1.0
    assert m["aggregate"]["avg_steps"] == 2.0
    assert "inkscape" in m["per_software"]


def test_compute_metrics_partial_success():
    results = [
        TaskResult(
            task_id="t1", software="inkscape", success=True, score=1.0, steps=1,
            step_results=[StepResult(step_num=1, action_type="set_value", target="r1", latency_ms=100)],
            e2e_time_s=0.5, deadlocked=False,
            observation_element_count=8, total_element_count=10,
        ),
        TaskResult(
            task_id="t2", software="inkscape", success=False, score=0.0, steps=5,
            step_results=[StepResult(step_num=i, action_type="set_value", target="r1", latency_ms=200) for i in range(1, 6)],
            e2e_time_s=5.0, deadlocked=True,
            observation_element_count=3, total_element_count=10,
        ),
    ]
    m = compute_metrics(results)
    assert m["aggregate"]["success_rate"] == 0.5
    assert m["aggregate"]["deadlock_rate"] == 0.5
    assert m["aggregate"]["avg_coverage"] == 0.55  # (0.8 + 0.3) / 2


def test_run_evaluation_batch():
    """Test batch evaluation across multiple tasks."""
    from asil.adapters.inkscape import InkscapeAdapter

    tmp = Path(tempfile.mkdtemp())
    svg = tmp / "test.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<rect id="r1" x="0" y="0" width="100" height="50"/>'
        '</svg>'
    )

    try:
        adapter = InkscapeAdapter(svg_path=svg)
        tasks = [
            TaskDefinition(id="t1", software="inkscape", difficulty="simple",
                description="Set width to 200",
                actions=[{"action_type": "modify_file", "target": str(svg),
                    "params": {"operations": [{"xpath": "//*[@id='r1']", "attribute": "width", "value": "200"}]}}],
                validation={"element_value": {"id": "r1", "key": "width", "expected": "200"}}),
            TaskDefinition(id="t2", software="inkscape", difficulty="simple",
                description="Set height to 75",
                actions=[{"action_type": "modify_file", "target": str(svg),
                    "params": {"operations": [{"xpath": "//*[@id='r1']", "attribute": "height", "value": "75"}]}}],
                validation={"element_value": {"id": "r1", "key": "height", "expected": "75"}}),
        ]
        results = run_evaluation(adapter, tasks)
        assert len(results) == 2
        assert results[0].success is True
        assert results[0].score == 1.0
        assert results[0].steps == 1
        assert len(results[0].step_results) == 1
        assert results[1].success is True
    finally:
        shutil.rmtree(tmp)


def test_run_evaluation_isolates_directory_backed_adapter_via_reset_state():
    class DirectoryLikeAdapter:
        def __init__(self):
            self.source_path = Path(tempfile.mkdtemp())
            self.value = "initial"
            self.reset_calls = 0
            self.setup_calls: list[str] = []

        def reset_state(self):
            self.reset_calls += 1
            self.value = "initial"

        def setup_state(self, initial_state):
            self.setup_calls.append(initial_state)
            if initial_state == "alt":
                self.value = "alt"

        def observe(self):
            from asil.protocol import Observation, Meta, Element

            return Observation(
                meta=Meta(app_name="dir-app", observation_source="test"),
                interactive_elements=[Element(id="state", type="text", value=self.value)],
            )

        def execute(self, action):
            self.value = action.params["value"]
            return self.observe()

        def validate_action(self, action):
            return True

    adapter = DirectoryLikeAdapter()
    tasks = [
        TaskDefinition(
            id="t1",
            software="code_server",
            difficulty="simple",
            description="set first",
            actions=[{"action_type": "modify_file", "target": "workspace", "params": {"value": "first"}}],
            validation={"element_value": {"id": "state", "key": None, "expected": "first"}},
            initial_state="default",
        ),
        TaskDefinition(
            id="t2",
            software="code_server",
            difficulty="simple",
            description="set second",
            actions=[{"action_type": "modify_file", "target": "workspace", "params": {"value": "second"}}],
            validation={"element_value": {"id": "state", "key": None, "expected": "second"}},
            initial_state="alt",
        ),
    ]

    try:
        results = run_evaluation(adapter, tasks, isolate_tasks=True)
    finally:
        shutil.rmtree(adapter.source_path)

    assert [result.success for result in results] == [True, True]
    assert adapter.reset_calls == 2
    assert adapter.setup_calls == ["default", "alt"]


def test_run_evaluation_isolates_file_backed_adapter_using_task_initial_state():
    class FileLikeAdapter:
        def __init__(self, source_path: Path):
            self.source_path = source_path
            self.setup_calls: list[str] = []
            self.setup_state("default")

        def setup_state(self, initial_state):
            self.setup_calls.append(initial_state)
            value = "alt" if initial_state == "alt" else "initial"
            self.source_path.write_text(value, encoding="utf-8")

        def observe(self):
            from asil.protocol import Observation, Meta, Element

            value = self.source_path.read_text(encoding="utf-8")
            return Observation(
                meta=Meta(app_name="file-app", observation_source="test"),
                interactive_elements=[Element(id="state", type="text", value=value)],
            )

        def execute(self, action):
            self.source_path.write_text(action.params["value"], encoding="utf-8")
            return self.observe()

        def validate_action(self, action):
            return True

    tmp = Path(tempfile.mkdtemp())
    source = tmp / "state.txt"
    adapter = FileLikeAdapter(source)
    tasks = [
        TaskDefinition(
            id="t1",
            software="celluloid",
            difficulty="simple",
            description="set from default",
            actions=[{"action_type": "modify_file", "target": "state.txt", "params": {"value": "first"}}],
            validation={"element_value": {"id": "state", "key": None, "expected": "first"}},
            initial_state="default",
        ),
        TaskDefinition(
            id="t2",
            software="celluloid",
            difficulty="simple",
            description="confirm alt state is restored before mutation",
            actions=[{"action_type": "modify_file", "target": "state.txt", "params": {"value": "second"}}],
            validation={"element_value": {"id": "state", "key": None, "expected": "second"}},
            initial_state="alt",
        ),
    ]

    try:
        results = run_evaluation(adapter, tasks, isolate_tasks=True)
    finally:
        shutil.rmtree(tmp)

    assert [result.success for result in results] == [True, True]
    assert adapter.setup_calls == ["default", "alt", "default"]


def test_step_result_to_dict():
    step = StepResult(
        step_num=1,
        action_type="modify_file",
        target="test.svg",
        params={"operations": [{"xpath": "//*[@id='r1']", "attribute": "width", "value": "200"}]},
        success=True,
        latency_ms=42.5,
        observation_element_count=5,
        observation_source="file_parse",
        grounding_error=False,
    )
    d = step.to_dict()
    assert d["step_num"] == 1
    assert d["action"]["action_type"] == "modify_file"
    assert d["latency_ms"] == 42.5
    assert d["grounding_error"] is False


def test_task_definition_from_osworld_dict_preserves_evaluator():
    task = TaskDefinition._from_osworld_dict(
        {
            "id": "obs_20",
            "instruction": "Do the thing",
            "related_apps": ["obs"],
            "evaluator": {
                "paths": [
                    {
                        "path_id": "main",
                        "checkpoints": [
                            {"id": "scene", "weight": 0.5, "rule": {"current_scene": "BRB"}},
                            {"id": "stream", "weight": 0.5, "rule": {"stream_active": True}},
                        ],
                    }
                ],
                "selection": "best_score",
            },
            "_asil": {
                "software": "obs",
                "difficulty": "complex",
                "description": "Do the thing",
                "actions": [],
                "validation": {},
            },
            "gui_expectations": {
                "success_surface": "program_view",
                "visible_change_summary": "OBS program scene changes to BRB",
                "checkpoint_visibility": {"scene": "visible_in_obs_ui"},
            },
        }
    )

    assert task.evaluator["selection"] == "best_score"
    assert task.evaluator["paths"][0]["path_id"] == "main"
    assert task.gui_expectations["success_surface"] == "program_view"


def test_task_definition_from_osworld_dict_preserves_multi_app_fields():
    task = TaskDefinition._from_osworld_dict(
        {
            "id": "multi_apps_001",
            "software": "multi_apps",
            "instruction": "Coordinate apps.",
            "related_apps": ["code_server", "jupyterlab"],
            "_asil": {
                "software": "multi_apps",
                "difficulty": "medium",
                "description": "Coordinate apps.",
                "app_initial_states": {"code_server": "src_focus", "jupyterlab": "notebook_focus"},
                "primary_app": "jupyterlab",
            },
        }
    )

    assert task.software == "multi_apps"
    assert task.related_apps == ["code_server", "jupyterlab"]
    assert task.app_initial_states["code_server"] == "src_focus"
    assert task.primary_app == "jupyterlab"


def test_run_evaluation_calls_prepare_task_for_composite_adapter():
    class PreparedAdapter:
        def __init__(self):
            self.prepared: list[str] = []
            self.value = ""

        def prepare_task(self, task):
            self.prepared.append(task.id)
            self._prepared_task_id = task.id
            self.value = task.id

        def get_context(self):
            return {}

        def observe(self):
            from asil.protocol import Element, Meta, Observation

            return Observation(
                meta=Meta(app_name="prepared", observation_source="unit"),
                interactive_elements=[Element(id="state", type="state", value=self.value)],
            )

        def execute(self, action):
            self.value = action.params["value"]
            return self.observe()

        def validate_action(self, action):
            return True

    adapter = PreparedAdapter()
    tasks = [
        TaskDefinition(
            id="one",
            software="multi_apps",
            difficulty="medium",
            description="one",
            actions=[{"action_type": "set", "target": "state", "params": {"value": "done-one"}}],
            validation={"element_value": {"id": "state", "key": None, "expected": "done-one"}},
        ),
        TaskDefinition(
            id="two",
            software="multi_apps",
            difficulty="medium",
            description="two",
            actions=[{"action_type": "set", "target": "state", "params": {"value": "done-two"}}],
            validation={"element_value": {"id": "state", "key": None, "expected": "done-two"}},
        ),
    ]

    results = run_evaluation(adapter, tasks, isolate_tasks=True)

    assert adapter.prepared == ["one", "two"]
    assert [result.success for result in results] == [True, True]


def test_task_definition_from_top_level_single_task_fields():
    task = TaskDefinition._from_osworld_dict(
        {
            "id": "libreoffice_impress_01",
            "software": "libreoffice_impress",
            "difficulty": "simple",
            "description": "Rename the first slide title.",
            "instruction": "Rename the first slide title to Launch Readiness.",
            "actions": [
                {
                    "action_type": "modify_file",
                    "target": "{{odp_path}}",
                    "params": {"operations": [{"action": "set_slide_title", "slide_index": 1, "text": "Launch Readiness"}]},
                }
            ],
            "validation": {},
            "related_apps": ["libreoffice_impress"],
            "gui_expectations": {
                "success_surface": "slide_canvas",
                "visible_change_summary": "Slide 1 title changes.",
                "checkpoint_visibility": {"title": "visible_on_slide"},
            },
            "render_target": {"mode": "single_slide", "slide_indices": [1]},
            "evaluator": {
                "selection": "best_score",
                "paths": [
                    {
                        "path_id": "main",
                        "checkpoints": [
                            {
                                "id": "title",
                                "weight": 1.0,
                                "rule": {
                                    "element_value": {
                                        "id": "slide:1:title",
                                        "key": "text_content",
                                        "expected": "Launch Readiness",
                                    }
                                },
                            }
                        ],
                    }
                ],
            },
        }
    )

    assert task.software == "libreoffice_impress"
    assert task.difficulty == "simple"
    assert len(task.actions) == 1
    assert task.actions[0]["params"]["operations"][0]["action"] == "set_slide_title"
    assert task.render_target["slide_indices"] == [1]
