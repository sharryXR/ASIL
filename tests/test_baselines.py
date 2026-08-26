"""Tests for CLI and GUI baselines."""
import tempfile
import shutil
from pathlib import Path

from asil.adapters.inkscape import InkscapeAdapter
from asil.baselines.cli_baseline import CLIBaseline
from asil.baselines.gui_agent import GUIAgentBaseline
from asil.protocol import Action


def test_cli_baseline_observe_degrades_elements(sample_svg: Path):
    """CLI baseline should flatten structured values to text."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    cli = CLIBaseline(delegate)
    obs = cli.observe()

    assert obs.meta.observation_source == "cli_output"
    assert obs.meta.app_name == "CLI-Inkscape"
    assert len(obs.interactive_elements) == 6

    # Values should be text (flattened from dict)
    rect = next(e for e in obs.interactive_elements if e.id == "rect1")
    assert isinstance(rect.value, str)
    assert "x=10" in rect.value

    # No constraints or actions in CLI output
    assert rect.constraints == {}
    assert rect.actions == []


def test_cli_baseline_execute_delegates(sample_svg: Path):
    """CLI baseline should still execute actions via delegate."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    cli = CLIBaseline(delegate)
    action = Action(
        action_type="modify_file",
        target=str(sample_svg),
        params={
            "operations": [
                {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}
            ]
        },
    )
    obs = cli.execute(action)
    rect = next(e for e in obs.interactive_elements if e.id == "rect1")
    # After execution, CLI output is flattened text — should contain updated width
    assert "width=200" in rect.value


def test_gui_agent_observe_partial_visibility(sample_svg: Path):
    """GUI agent should only see ~50% of elements (simulating screenshot)."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=0.5, seed=42)
    obs = gui.observe()

    assert obs.meta.observation_source == "screenshot"
    assert obs.meta.app_name == "GUI-Inkscape"
    assert len(obs.interactive_elements) < 4
    assert len(obs.interactive_elements) >= 1
    # Environment should be empty (GUI can't see background)
    assert obs.environment.undo_stack_depth == 0
    assert obs.environment.system == {}


def test_gui_agent_execute_has_failure_rate(sample_svg: Path):
    """GUI agent should sometimes fail actions (grounding errors)."""
    tmp = Path(tempfile.mkdtemp())
    svg_content = sample_svg.read_text()
    action = Action(
        action_type="modify_file",
        target="dummy",
        params={
            "operations": [
                {"xpath": "//*[@id='rect1']", "attribute": "width", "value": "999"}
            ]
        },
    )

    successes = 0
    failures = 0
    try:
        for seed in range(100):
            fresh_svg = tmp / f"test_{seed}.svg"
            fresh_svg.write_text(svg_content)
            delegate = InkscapeAdapter(svg_path=fresh_svg)
            gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, action_failure_rate=0.5, seed=seed)
            action.target = str(fresh_svg)
            gui.execute(action)

            # Check the actual file to see if it was modified
            content = fresh_svg.read_text()
            if 'width="999"' in content:
                successes += 1
            else:
                failures += 1
    finally:
        shutil.rmtree(tmp)

    assert successes > 20, f"Expected >20 successes, got {successes}"
    assert failures > 20, f"Expected >20 failures, got {failures}"


def test_gui_agent_step_log_records_grounding_errors(sample_svg: Path):
    """GUI agent should record grounding errors in step log."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, action_failure_rate=0.5, seed=42)

    action = Action(
        action_type="modify_file",
        target=str(sample_svg),
        params={"operations": [{"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}]},
    )

    for i in range(20):
        gui.execute(action)

    log = gui.step_log
    assert len(log) == 20

    errors = [e for e in log if e["grounding_error"]]
    successes = [e for e in log if not e["grounding_error"]]
    assert len(errors) > 0, "Should have at least some grounding errors"
    assert len(successes) > 0, "Should have at least some successes"

    error_types = {e["error_type"] for e in errors}
    assert error_types.issubset({"click_miss", "wrong_element", "timeout"})


def test_gui_agent_validate_accepts_all():
    """GUI agent accepts all action types."""
    delegate = InkscapeAdapter(svg_path="/tmp/test.svg")
    gui = GUIAgentBaseline(delegate)

    for action_type in ["modify_file", "set_value", "api_call", "navigate", "invoke_function"]:
        action = Action(action_type=action_type, target="x", params={})
        assert gui.validate_action(action)


# ---------------------------------------------------------------------------
# New tests: GUI baseline OSWorld alignment
# ---------------------------------------------------------------------------

def test_gui_agent_trajectory_recorded(sample_svg: Path):
    """GUI baseline should record an OSWorld-compatible trajectory entry per execute()."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, action_failure_rate=0.0, seed=0)

    action = Action(
        action_type="modify_file", target=str(sample_svg),
        params={"operations": [{"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}]},
    )
    gui.execute(action)

    traj = gui.trajectory
    assert len(traj) == 1
    entry = traj[0]
    assert entry["step_num"] == 1
    assert "action_timestamp" in entry
    assert entry["action"]["action_type"] == "modify_file"
    assert entry["grounding_error"] is False
    assert "observation_element_count" in entry


def test_gui_agent_trajectory_grounding_error(sample_svg: Path):
    """Grounding errors should be recorded in trajectory."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    # 100% failure rate
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, action_failure_rate=1.0, seed=0)

    action = Action(action_type="modify_file", target=str(sample_svg), params={})
    gui.execute(action)

    traj = gui.trajectory
    assert len(traj) == 1
    assert traj[0]["grounding_error"] is True
    assert traj[0]["grounding_error_type"] in {"click_miss", "wrong_element", "timeout"}


def test_gui_agent_save_trajectory(sample_svg: Path, tmp_path):
    """save_trajectory() should write OSWorld-compatible traj.jsonl."""
    import json as _json

    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, action_failure_rate=0.0, seed=0)

    action = Action(
        action_type="modify_file", target=str(sample_svg),
        params={"operations": [{"xpath": "//*[@id='rect1']", "attribute": "width", "value": "200"}]},
    )
    gui.execute(action)
    gui.save_trajectory(tmp_path / "traj_out")

    traj_file = tmp_path / "traj_out" / "traj.jsonl"
    assert traj_file.exists()
    lines = traj_file.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = _json.loads(lines[0])
    assert entry["step_num"] == 1
    assert entry["observation_source"] == "screenshot"


def test_gui_agent_reset_clears_trajectory(sample_svg: Path):
    """reset() should clear trajectory and step_log."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, action_failure_rate=0.0, seed=0)

    action = Action(action_type="modify_file", target=str(sample_svg), params={})
    gui.execute(action)
    assert len(gui.trajectory) == 1
    assert len(gui.step_log) == 1

    gui.reset()
    assert len(gui.trajectory) == 0
    assert len(gui.step_log) == 0


def test_gui_agent_format_a11y_tree(sample_svg: Path):
    """format_observation_as_a11y_tree() should return structured text."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, seed=0)

    obs = gui.observe()
    a11y = gui.format_observation_as_a11y_tree(obs)

    assert "Application:" in a11y
    assert "tag\tname\tvalue" in a11y
    # At least one element row
    lines = a11y.split("\n")
    assert len(lines) > 4


def test_gui_agent_predict_returns_response(sample_svg: Path):
    """predict() should return an OSWorld-compatible (response, actions) tuple."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=1.0, seed=0)

    obs = gui.observe()
    response, actions = gui.predict("Set rect1 width to 200", obs)

    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(actions, list)


def test_cli_baseline_clone_is_independent(sample_svg: Path):
    """CLIBaseline clone should create an independent wrapper."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    cli = CLIBaseline(delegate)

    tmp = Path(tempfile.mkdtemp())
    try:
        clone_path = tmp / "clone.svg"
        cli_clone = cli.clone(clone_path)

        assert isinstance(cli_clone, CLIBaseline)
        assert cli_clone.source_path == clone_path
        assert cli_clone.source_path != cli.source_path
    finally:
        shutil.rmtree(tmp)


def test_gui_baseline_clone_preserves_params(sample_svg: Path):
    """GUIAgentBaseline clone should preserve visibility_ratio and failure_rate."""
    delegate = InkscapeAdapter(svg_path=sample_svg)
    gui = GUIAgentBaseline(delegate, visibility_ratio=0.3, action_failure_rate=0.2, seed=7)

    tmp = Path(tempfile.mkdtemp())
    try:
        clone_path = tmp / "clone.svg"
        gui_clone = gui.clone(clone_path)

        assert isinstance(gui_clone, GUIAgentBaseline)
        assert gui_clone._visibility_ratio == 0.3
        assert gui_clone._failure_rate == 0.2
        assert gui_clone.source_path == clone_path
    finally:
        shutil.rmtree(tmp)
