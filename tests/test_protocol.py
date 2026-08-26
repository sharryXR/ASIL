from asil.protocol import Observation, Action, Element, AppState, Environment, Navigation, Meta


def test_element_creation():
    elem = Element(
        id="rect1",
        type="rect",
        label="My Rectangle",
        value={"x": 10, "y": 20, "width": 100, "height": 50},
        editable=True,
        actions=["set_attribute", "move", "resize", "delete"],
    )
    assert elem.id == "rect1"
    assert elem.type == "rect"
    assert elem.editable is True
    assert "move" in elem.actions


def test_observation_creation():
    obs = Observation(
        meta=Meta(app_name="Inkscape", observation_source="file_parse"),
        app_state=AppState(current_view="canvas", active_document="test.svg"),
        interactive_elements=[
            Element(id="rect1", type="rect", label="Rect", value={}, editable=True, actions=["move"]),
        ],
        environment=Environment(),
    )
    assert obs.meta.app_name == "Inkscape"
    assert len(obs.interactive_elements) == 1
    assert obs.interactive_elements[0].id == "rect1"


def test_observation_to_json_roundtrip():
    obs = Observation(
        meta=Meta(app_name="Test", observation_source="file_parse"),
        app_state=AppState(current_view="main"),
        interactive_elements=[],
        environment=Environment(),
    )
    json_str = obs.model_dump_json()
    restored = Observation.model_validate_json(json_str)
    assert restored.meta.app_name == "Test"


def test_action_set_value():
    action = Action(
        action_type="set_value",
        target="cell_A1",
        params={"value": "Hello"},
    )
    assert action.action_type == "set_value"
    assert action.target == "cell_A1"
    assert action.params["value"] == "Hello"


def test_action_modify_file():
    action = Action(
        action_type="modify_file",
        target="drawing.svg",
        params={
            "operations": [
                {"xpath": "//rect[@id='rect1']", "attribute": "width", "value": "200"}
            ]
        },
    )
    assert action.action_type == "modify_file"
    assert len(action.params["operations"]) == 1


def test_action_invoke_function():
    action = Action(
        action_type="invoke_function",
        target="bpy",
        params={"script": ["import bpy", "bpy.ops.mesh.primitive_cube_add()"]},
    )
    assert action.action_type == "invoke_function"
    assert len(action.params["script"]) == 2


def test_action_api_call():
    action = Action(
        action_type="api_call",
        target="gitea_rest",
        params={"method": "POST", "endpoint": "/api/v1/user/repos", "body": {"name": "test"}},
    )
    assert action.action_type == "api_call"
    assert action.params["method"] == "POST"


def test_action_batch():
    action = Action(
        action_type="batch",
        target="inkscape",
        params={
            "actions": [
                {"action_type": "set_value", "target": "rect1", "params": {"attribute": "width", "value": "200"}},
                {"action_type": "set_value", "target": "rect1", "params": {"attribute": "height", "value": "100"}},
            ]
        },
    )
    assert action.action_type == "batch"
    assert len(action.params["actions"]) == 2


from asil.adapter import ASILAdapter
from asil.protocol import Observation, Action


class MockAdapter(ASILAdapter):
    """A minimal adapter for testing the interface."""
    app_name = "MockApp"
    app_version = "1.0"
    supported_action_types = ["set_value"]

    def observe(self) -> Observation:
        return self._build_observation(
            source="file_parse",
            elements=[],
            app_state={"current_view": "main"},
        )

    def execute(self, action: Action) -> Observation:
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types


def test_adapter_interface():
    adapter = MockAdapter()
    obs = adapter.observe()
    assert obs.meta.app_name == "MockApp"
    assert obs.meta.observation_source == "file_parse"


def test_adapter_observe_execute_loop():
    adapter = MockAdapter()
    obs = adapter.observe()
    action = Action(action_type="set_value", target="elem1", params={"value": "test"})
    assert adapter.validate_action(action)
    new_obs = adapter.execute(action)
    assert new_obs.meta.app_name == "MockApp"


def test_adapter_rejects_unsupported_action():
    adapter = MockAdapter()
    action = Action(action_type="invoke_function", target="bpy", params={})
    assert not adapter.validate_action(action)
