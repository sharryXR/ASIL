from unittest.mock import MagicMock
from pathlib import Path

from asil.agent import ASILAgent, AgentModelOutput
from asil.adapters.inkscape import InkscapeAdapter
from asil.protocol import Action, Observation


def test_agent_format_observation_prompt(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    agent = ASILAgent(adapter=adapter, llm_fn=MagicMock())
    obs = adapter.observe()
    prompt = agent.format_observation(obs, task_description="Create a blue rectangle")

    assert "Create a blue rectangle" in prompt
    assert "rect1" in prompt
    assert "Inkscape" in prompt


def test_agent_parse_action_from_json():
    agent = ASILAgent(adapter=MagicMock(), llm_fn=MagicMock())
    llm_output = '{"action_type": "modify_file", "target": "test.svg", "params": {"operations": []}}'
    action = agent.parse_action(llm_output)
    assert action.action_type == "modify_file"


def test_agent_parse_action_from_markdown_code_block():
    agent = ASILAgent(adapter=MagicMock(), llm_fn=MagicMock())
    llm_output = 'I will modify the file:\n```json\n{"action_type": "set_value", "target": "cell_A1", "params": {"value": "test"}}\n```'
    action = agent.parse_action(llm_output)
    assert action.action_type == "set_value"


def test_agent_parse_trace_from_thought_action_sections():
    agent = ASILAgent(adapter=MagicMock(), llm_fn=MagicMock())
    llm_output = (
        "Thought: Check whether rect1 already satisfies the width requirement. "
        "If not, update width to 200 and verify.\n"
        'Action: {"action_type": "modify_file", "target": "test.svg", "params": {"operations": []}}'
    )

    trace = agent.parse_trace(llm_output, instruction="Set rect1 width to 200")

    assert trace.instruction == "Set rect1 width to 200"
    assert trace.thought.startswith("Check whether rect1")
    assert trace.action.action_type == "modify_file"


def test_agent_parse_trace_from_model_output():
    agent = ASILAgent(adapter=MagicMock(), llm_fn=MagicMock())
    output = AgentModelOutput(
        text='Thought: verify cube material\nAction: {"action_type": "done", "target": "", "params": {}}',
        reasoning_summary="Inspect current scene, then stop if success criteria are met.",
        provider="openai",
        model="gpt-5.4",
    )

    trace = agent.parse_trace(output, instruction="Add a red material to the default cube")

    assert trace.reasoning_summary.startswith("Inspect current scene")
    assert trace.provider == "openai"
    assert trace.model == "gpt-5.4"
    assert trace.action.action_type == "done"


def test_agent_step(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    mock_llm = MagicMock(
        return_value=(
            "Thought: rect1 width is still 100, so update it to 200.\n"
            'Action: {"action_type": "modify_file", "target": "' + str(sample_svg) + '", '
            '"params": {"operations": [{"xpath": "//*[@id=\'rect1\']", "attribute": "width", "value": "200"}]}}'
        )
    )
    agent = ASILAgent(adapter=adapter, llm_fn=mock_llm)

    obs, action, new_obs, trace = agent.step("Set rect1 width to 200")
    assert action.action_type == "modify_file"
    assert "update it to 200" in trace.thought
    rect = next(e for e in new_obs.interactive_elements if e.id == "rect1")
    assert rect.value["width"] == "200"


def test_agent_step_done_does_not_execute_adapter(sample_svg: Path):
    adapter = InkscapeAdapter(svg_path=sample_svg)
    original_observe = adapter.observe()
    original_execute = adapter.execute

    execute_calls = {"count": 0}

    def fail_if_called(action):
        execute_calls["count"] += 1
        return original_execute(action)

    adapter.execute = fail_if_called  # type: ignore[method-assign]
    mock_llm = MagicMock(
        return_value=(
            "Thought: The success criteria are already satisfied, so stop.\n"
            'Action: {"action_type": "done", "target": "", "params": {}}'
        )
    )
    agent = ASILAgent(adapter=adapter, llm_fn=mock_llm)

    obs, action, new_obs, trace = agent.step("Leave rect1 unchanged if already correct")

    assert action.action_type == "done"
    assert trace.thought.startswith("The success criteria")
    assert execute_calls["count"] == 0
    assert [e.id for e in obs.interactive_elements] == [e.id for e in original_observe.interactive_elements]
    assert [e.id for e in new_obs.interactive_elements] == [e.id for e in original_observe.interactive_elements]
