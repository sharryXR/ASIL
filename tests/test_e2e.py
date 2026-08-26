"""End-to-end integration tests — agent loop with real adapters."""

import json
import tempfile
import shutil
from pathlib import Path

from asil.agent import ASILAgent, create_llm_fn
from asil.adapters.inkscape import InkscapeAdapter
from asil.protocol import Action, Observation
from asil.eval.runner import TaskDefinition, run_task, run_task_with_agent
from asil.eval.metrics import compute_metrics


def _create_svg(tmp: Path) -> Path:
    svg = tmp / "e2e_test.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">'
        '<g id="layer1">'
        '<rect id="rect1" x="10" y="20" width="100" height="50" style="fill:red"/>'
        '<circle id="circle1" cx="200" cy="150" r="40" style="fill:green"/>'
        '</g></svg>'
    )
    return svg


def test_e2e_agent_modify_inkscape():
    """Full observe-think-act cycle: agent modifies an SVG element."""
    tmp = Path(tempfile.mkdtemp())
    svg = _create_svg(tmp)

    try:
        adapter = InkscapeAdapter(svg_path=svg)

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "action_type": "modify_file",
                "target": str(svg),
                "params": {"operations": [
                    {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
                ]}
            })

        agent = ASILAgent(adapter=adapter, llm_fn=mock_llm)
        obs, action, new_obs, trace = agent.step("Set rect1 width to 200")

        assert action.action_type == "modify_file"
        assert trace.action == action
        rect = next(e for e in new_obs.interactive_elements if e.id == "rect1")
        assert rect.value["width"] == "200"
    finally:
        shutil.rmtree(tmp)


def test_e2e_agent_multi_step():
    """Agent performs multiple steps in sequence."""
    tmp = Path(tempfile.mkdtemp())
    svg = _create_svg(tmp)

    try:
        adapter = InkscapeAdapter(svg_path=svg)
        step_count = [0]

        def mock_llm(prompt: str) -> str:
            step_count[0] += 1
            if step_count[0] == 1:
                return json.dumps({
                    "action_type": "modify_file",
                    "target": str(svg),
                    "params": {"operations": [
                        {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
                    ]}
                })
            else:
                return json.dumps({
                    "action_type": "modify_file",
                    "target": str(svg),
                    "params": {"operations": [
                        {"xpath": "//*[@id='rect1']", "attribute": "style", "value": "fill:blue"}
                    ]}
                })

        agent = ASILAgent(adapter=adapter, llm_fn=mock_llm, max_steps=2)
        history = agent.run("Modify rect1")

        assert len(history) == 2
        assert history[0][1].params["operations"][0]["value"] == "200"
    finally:
        shutil.rmtree(tmp)


def test_e2e_evaluation_with_agent():
    """Run full evaluation pipeline: task definition -> agent execution -> metrics."""
    tmp = Path(tempfile.mkdtemp())
    svg = _create_svg(tmp)

    try:
        adapter = InkscapeAdapter(svg_path=svg)

        task = TaskDefinition(
            id="e2e_test",
            software="inkscape",
            difficulty="simple",
            description="Set rect1 width to 200",
            actions=[{
                "action_type": "modify_file",
                "target": str(svg),
                "params": {"operations": [
                    {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
                ]}
            }],
            validation={"element_value": {"id": "rect1", "key": "width", "expected": "200"}},
        )

        result = run_task(adapter, task)
        assert result.success is True
        assert result.score == 1.0
        assert result.steps == 1
        assert len(result.step_results) == 1
        assert result.step_results[0].success is True

        metrics = compute_metrics([result])
        assert metrics["aggregate"]["success_rate"] == 1.0
        assert metrics["aggregate"]["avg_steps"] == 1.0
    finally:
        shutil.rmtree(tmp)


def test_e2e_task_trajectory_output():
    """Verify trajectory output (traj.jsonl) works correctly."""
    tmp = Path(tempfile.mkdtemp())
    svg = _create_svg(tmp)

    try:
        adapter = InkscapeAdapter(svg_path=svg)
        task = TaskDefinition(
            id="traj_test",
            software="inkscape",
            difficulty="simple",
            description="Set rect1 width to 200",
            actions=[{
                "action_type": "modify_file",
                "target": str(svg),
                "params": {"operations": [
                    {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
                ]}
            }],
            validation={"element_value": {"id": "rect1", "key": "width", "expected": "200"}},
        )

        result = run_task(adapter, task)

        # Save trajectory
        traj_dir = tmp / "output" / "inkscape" / "traj_test"
        result.save_trajectory(traj_dir)
        result.save_result(traj_dir)

        # Verify traj.jsonl
        traj_path = traj_dir / "traj.jsonl"
        assert traj_path.exists()
        lines = traj_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["step_num"] == 1
        assert entry["action"]["action_type"] == "modify_file"

        # Verify result.txt
        result_txt = traj_dir / "result.txt"
        assert result_txt.exists()
        assert result_txt.read_text() == "1.0"
    finally:
        shutil.rmtree(tmp)


def test_e2e_agent_based_task():
    """Run task with agent loop (observe-think-act) instead of predefined actions."""
    tmp = Path(tempfile.mkdtemp())
    svg = _create_svg(tmp)

    try:
        adapter = InkscapeAdapter(svg_path=svg)
        task = TaskDefinition(
            id="agent_test",
            software="inkscape",
            difficulty="simple",
            description="Set rect1 width to 200",
            instruction="Set the width of rect1 to 200",
            validation={"element_value": {"id": "rect1", "key": "width", "expected": "200"}},
        )

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "action_type": "modify_file",
                "target": str(svg),
                "params": {"operations": [
                    {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
                ]}
            })

        result = run_task_with_agent(adapter, task, llm_fn=mock_llm, max_steps=5)
        assert result.success is True
        assert result.score == 1.0
        assert len(result.step_results) >= 1
    finally:
        shutil.rmtree(tmp)


def test_create_llm_fn_mock():
    """Mock LLM factory should return a normalized provider output."""
    llm = create_llm_fn(provider="mock")
    result = llm("test prompt")
    assert result.provider == "mock"
    assert result.model == "mock"
    _, action_text = result.text.split("Action:", 1)
    data = json.loads(action_text)
    assert "action_type" in data
    assert "target" in data
    assert "params" in data


def test_create_llm_fn_unknown_provider():
    """Unknown provider should raise ValueError."""
    try:
        create_llm_fn(provider="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
