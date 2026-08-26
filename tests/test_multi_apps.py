from __future__ import annotations

from pathlib import Path

from asil.adapter import ASILAdapter, GUISessionSpec
from asil.adapters.code_server import CodeServerAdapter
from asil.adapters.jupyterlab import JupyterLabAdapter
from asil.adapters.multi_apps import MultiAppAdapter
from asil.adapters.nautilus import NautilusAdapter
from asil.eval.runner import TaskDefinition
from asil.protocol import Action, Element


class FakeChildAdapter(ASILAdapter):
    app_name = "fake"
    supported_action_types = ["set_value", "append"]

    def __init__(self, software: str) -> None:
        self.software = software
        self.value = "initial"
        self.reset_calls = 0
        self.setup_calls: list[str] = []
        self.synced = False

    def reset_state(self) -> None:
        self.reset_calls += 1
        self.value = "reset"

    def setup_state(self, initial_state: str) -> None:
        self.setup_calls.append(initial_state)
        self.value = initial_state

    def observe(self):
        return self._build_observation(
            source="unit",
            elements=[
                Element(
                    id="state",
                    type="state",
                    label="State",
                    value={"value": self.value},
                )
            ],
            app_state={"current_view": self.software, "active_document": "state"},
        )

    def execute(self, action: Action):
        if action.action_type == "set_value":
            self.value = str(action.params["value"])
        elif action.action_type == "append":
            self.value += str(action.params["suffix"])
        else:
            raise ValueError(action.action_type)
        return self.observe()

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types

    def get_gui_session_spec(self):
        return GUISessionSpec(
            surface_type="desktop",
            window_title_pattern=self.software,
            launch_command=(self.software,),
            persist_shortcuts=("ctrl+s",),
        )

    def sync_from_gui(self, session=None) -> None:
        self.synced = True


class FailingSyncChildAdapter(FakeChildAdapter):
    def sync_from_gui(self, session=None) -> None:
        raise RuntimeError(f"{self.software} sync failed")


class OrderedSpecChildAdapter(FakeChildAdapter):
    calls: list[str] = []

    def get_gui_session_spec(self):
        type(self).calls.append(self.software)
        return GUISessionSpec(
            surface_type="browser" if self.software == "gitea" else "desktop",
            window_title_pattern=self.software,
            launch_command=() if self.software == "gitea" else (self.software,),
        )


def _task() -> TaskDefinition:
    return TaskDefinition(
        id="multi",
        software="multi_apps",
        difficulty="medium",
        description="test multi",
        related_apps=["code_server", "jupyterlab"],
        app_initial_states={"code_server": "src_focus", "jupyterlab": "notebook_focus"},
        primary_app="jupyterlab",
    )


def test_prepare_task_creates_and_sets_up_child_adapters(tmp_path: Path):
    created: dict[str, FakeChildAdapter] = {}

    def factory(software, child_tmp, sandbox, mock):
        del child_tmp, sandbox, mock
        adapter = FakeChildAdapter(software)
        created[software] = adapter
        return adapter

    adapter = MultiAppAdapter(tmp_path, adapter_factory=factory)
    adapter.prepare_task(_task())

    assert adapter.related_apps == ["code_server", "jupyterlab"]
    assert adapter.primary_app == "jupyterlab"
    assert created["code_server"].reset_calls == 1
    assert created["code_server"].setup_calls == ["src_focus"]
    assert created["jupyterlab"].setup_calls == ["notebook_focus"]


def test_observe_namespaces_child_elements_and_metadata(tmp_path: Path):
    adapter = MultiAppAdapter(
        tmp_path,
        adapter_factory=lambda software, child_tmp, sandbox, mock: FakeChildAdapter(software),
    )
    adapter.prepare_task(_task())

    obs = adapter.observe()
    ids = {element.id for element in obs.interactive_elements}
    state = next(element for element in obs.interactive_elements if element.id == "code_server::state")

    assert "code_server::state" in ids
    assert "jupyterlab::state" in ids
    assert state.metadata["app"] == "code_server"
    assert state.metadata["local_id"] == "state"
    assert obs.app_state.current_view == "jupyterlab"


def test_execute_dispatches_batch_and_single_app_actions(tmp_path: Path):
    adapter = MultiAppAdapter(
        tmp_path,
        adapter_factory=lambda software, child_tmp, sandbox, mock: FakeChildAdapter(software),
    )
    adapter.prepare_task(_task())

    adapter.execute(
        Action(
            action_type="batch",
            target="multi_apps",
            params={
                "actions": [
                    {"app": "code_server", "action": {"action_type": "set_value", "target": "state", "params": {"value": "code"}}},
                    {"app": "jupyterlab", "action": {"action_type": "set_value", "target": "state", "params": {"value": "notebook"}}},
                ]
            },
        )
    )
    adapter.execute(
        Action(
            action_type="append",
            target="jupyterlab::state",
            params={"suffix": "-done"},
        )
    )

    obs = adapter.observe()
    values = {element.id: element.value["value"] for element in obs.interactive_elements if element.id.endswith("::state")}
    assert values["code_server::state"] == "code"
    assert values["jupyterlab::state"] == "notebook-done"


def test_execute_canonicalizes_namespaced_workspace_file_actions(tmp_path: Path):
    def factory(software, child_tmp, sandbox, mock):
        del sandbox, mock
        if software == "code_server":
            return CodeServerAdapter(child_tmp / "code-server-workspace")
        if software == "jupyterlab":
            return JupyterLabAdapter(child_tmp / "jupyterlab-workspace", active_file="README.md")
        raise AssertionError(software)

    adapter = MultiAppAdapter(tmp_path, adapter_factory=factory)
    adapter.prepare_task(_task())

    adapter.execute(
        Action(
            action_type="set_file_text",
            target="code_server::editor:README.md",
            params={"path": "reports/handoff.md", "text": "handoff complete"},
        )
    )
    adapter.execute(
        Action(
            action_type="set_file_text",
            target="jupyterlab::editor:README.md",
            params={"path": "reports/analysis.md", "content": "analysis saved"},
        )
    )

    obs = adapter.observe()
    by_id = {element.id: element for element in obs.interactive_elements}
    assert by_id["code_server::file:reports/handoff.md"].value["content"] == "handoff complete"
    assert by_id["jupyterlab::file:reports/analysis.md"].value["content"] == "analysis saved"


def test_prepare_task_primes_workspace_checkpoint_files_for_gui(tmp_path: Path):
    def factory(software, child_tmp, sandbox, mock):
        del sandbox, mock
        if software == "code_server":
            return CodeServerAdapter(child_tmp / "code-server-workspace")
        if software == "jupyterlab":
            return JupyterLabAdapter(child_tmp / "jupyterlab-workspace", active_file="README.md")
        raise AssertionError(software)

    task = TaskDefinition(
        id="multi",
        software="multi_apps",
        difficulty="medium",
        description="test multi",
        related_apps=["code_server", "jupyterlab"],
        app_initial_states={"code_server": "default", "jupyterlab": "readme_focus"},
        primary_app="code_server",
        evaluator={
            "paths": [
                {
                    "path_id": "all",
                    "checkpoints": [
                        {
                            "rule": {
                                "app_rule": {
                                    "app": "code_server",
                                    "rule": {
                                        "element_contains": {
                                            "id": "file:reports/handoff.md",
                                            "key": "content",
                                            "expected": "done",
                                        }
                                    },
                                }
                            }
                        },
                        {
                            "rule": {
                                "app_rule": {
                                    "app": "jupyterlab",
                                    "rule": {
                                        "element_contains": {
                                            "id": "file:reports/analysis.md",
                                            "key": "content",
                                            "expected": "saved",
                                        }
                                    },
                                }
                            }
                        },
                    ],
                }
            ]
        },
    )

    adapter = MultiAppAdapter(tmp_path, adapter_factory=factory)
    adapter.prepare_task(task)

    code = adapter.active_adapters["code_server"]
    notebook = adapter.active_adapters["jupyterlab"]
    assert (code.workspace_path / "reports/handoff.md").read_text(encoding="utf-8") == ""
    assert (notebook.workspace_path / "reports/analysis.md").read_text(encoding="utf-8") == ""
    assert code.get_context()["active_file"] == "reports/handoff.md"
    assert notebook.get_context()["active_file"] == "reports/analysis.md"


def test_execute_canonicalizes_nautilus_modify_file_rename(tmp_path: Path):
    def factory(software, child_tmp, sandbox, mock):
        del sandbox, mock
        if software == "nautilus":
            return NautilusAdapter.from_evaluation_context(child_tmp)
        if software == "code_server":
            return CodeServerAdapter(child_tmp / "code-server-workspace")
        raise AssertionError(software)

    adapter = MultiAppAdapter(tmp_path, adapter_factory=factory)
    adapter.prepare_task(
        TaskDefinition(
            id="multi",
            software="multi_apps",
            difficulty="medium",
            description="test multi",
            related_apps=["nautilus", "code_server"],
            app_initial_states={"nautilus": "default", "code_server": "default"},
            primary_app="nautilus",
        )
    )

    adapter.execute(
        Action(
            action_type="modify_file",
            target="nautilus::workspace",
            params={
                "operations": [
                    {
                        "action": "rename_path",
                        "path": "meeting-notes.md",
                        "new_path": "meeting-notes-renamed.md",
                    }
                ]
            },
        )
    )

    ids = {element.id for element in adapter.observe().interactive_elements}
    assert "nautilus::workspace:meeting-notes-renamed.md" in ids
    assert "nautilus::workspace:meeting-notes.md" not in ids


def test_coerce_child_action_normalizes_common_multi_app_aliases(tmp_path: Path):
    adapter = MultiAppAdapter(tmp_path)

    gimp = adapter._coerce_child_action(
        "gimp",
        Action(
            action_type="invoke_function",
            target="gimp::image",
            params={
                "operations": [
                    {"action": "add_text_layer", "id": "ma_label", "text": "MA-032 asset"}
                ]
            },
        ),
    )
    assert gimp.action_type == "invoke_function"
    assert gimp.target == "gimp"
    assert gimp.params["operations"][0]["id"] == "gimp_multi_032"

    audacity = adapter._coerce_child_action(
        "audacity",
        Action(
            action_type="modify_file",
            target="audacity::document",
            params={"operations": [{"action": "add_label", "text": "MA-045 audio cue"}]},
        ),
    )
    assert audacity.action_type == "modify_file"
    assert audacity.target == "audacity_project"
    assert audacity.params["operations"][0]["label_id"] == "aud_multi_045"

    kdenlive = adapter._coerce_child_action(
        "kdenlive",
        Action(
            action_type="modify_file",
            target="kdenlive::project",
            params={"operations": [{"action": "add_marker", "comment": "MA-047 edit marker"}]},
        ),
    )
    marker = kdenlive.params["operations"][0]
    assert marker["action"] == "add_element"
    assert marker["attributes"]["id"] == "kd_multi_047"
    assert marker["attributes"]["comment"] == "MA-047 edit marker"

    thunderbird = adapter._coerce_child_action(
        "thunderbird",
        Action(
            action_type="add_tag",
            target="thunderbird::message:msg_client_followup",
            params={"tag": "multi-057"},
        ),
    )
    assert thunderbird.action_type == "invoke_function"
    assert thunderbird.target == "thunderbird"
    assert thunderbird.params["operations"][0] == {
        "action": "add_tag",
        "tag": "multi-057",
        "id": "msg_client_followup",
    }


def test_get_gui_session_spec_builds_multi_window_child_specs(tmp_path: Path):
    adapter = MultiAppAdapter(
        tmp_path,
        adapter_factory=lambda software, child_tmp, sandbox, mock: FakeChildAdapter(software),
    )
    adapter.prepare_task(_task())

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert spec.surface_type == "multi_window"
    assert spec.primary_child == "jupyterlab"
    assert set(spec.child_specs) == {"code_server", "jupyterlab"}
    assert spec.capture_active_window is True
    assert spec.min_width == 1200
    assert spec.min_height == 700


def test_get_gui_session_spec_resolves_browser_children_before_desktop_specs(tmp_path: Path):
    OrderedSpecChildAdapter.calls = []
    adapter = MultiAppAdapter(
        tmp_path,
        adapter_factory=lambda software, child_tmp, sandbox, mock: OrderedSpecChildAdapter(software),
    )
    adapter.prepare_task(
        TaskDefinition(
            id="multi",
            software="multi_apps",
            difficulty="medium",
            description="test multi",
            related_apps=["nautilus", "gitea"],
            app_initial_states={"nautilus": "default", "gitea": "default"},
            primary_app="nautilus",
        )
    )

    spec = adapter.get_gui_session_spec()

    assert spec is not None
    assert OrderedSpecChildAdapter.calls == ["gitea", "nautilus"]
    assert list(spec.child_specs) == ["gitea", "nautilus"]


def test_sync_from_gui_keeps_observation_available_when_one_child_sync_fails(tmp_path: Path):
    def factory(software, child_tmp, sandbox, mock):
        del child_tmp, sandbox, mock
        if software == "code_server":
            return FailingSyncChildAdapter(software)
        return FakeChildAdapter(software)

    adapter = MultiAppAdapter(tmp_path, adapter_factory=factory)
    adapter.prepare_task(_task())

    adapter.sync_from_gui(type("Session", (), {"child_sessions": {}})())
    obs = adapter.observe()

    assert adapter.__dict__["_last_gui_sync_errors"] == {
        "code_server": "RuntimeError: code_server sync failed"
    }
    assert {element.id for element in obs.interactive_elements} >= {
        "code_server::state",
        "jupyterlab::state",
    }


def test_persist_gui_state_restores_preexisting_active_window(monkeypatch, tmp_path: Path):
    adapter = MultiAppAdapter(
        tmp_path,
        adapter_factory=lambda software, child_tmp, sandbox, mock: FakeChildAdapter(software),
    )
    adapter.prepare_task(_task())
    spec = adapter.get_gui_session_spec()
    assert spec is not None
    calls: list[str] = []

    class Controller:
        display = None

        def persist(self, child_spec):
            calls.append(f"persist:{child_spec.window_title_pattern}")

        def activate_window_id(self, window_id):
            calls.append(f"activate:{window_id}")

    monkeypatch.setattr("asil.rendering.active_window_id", lambda display=None: "555")

    adapter.persist_gui_state(Controller(), spec)

    assert calls[-1] == "activate:555"
