"""Regression tests for script-level adapter wiring in run_evaluation.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from asil.rendering import RenderArtifact
from asil.protocol import Action, AppState, Element, Environment, Meta, Observation
from asil.eval.metrics import StepResult, TaskResult


def _load_run_evaluation_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("asil_run_evaluation_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_adapter_obs_uses_env_config(tmp_path):
    module = _load_run_evaluation_module()

    with patch.dict(
        "os.environ",
        {
            "OBS_WS_HOST": "obs-gui",
            "OBS_WS_PORT": "5566",
            "OBS_WS_PASSWORD": "secret",
            "OBS_REAL_GUI": "true",
            "OBS_WS_PROTOCOL": "v4",
        },
        clear=False,
    ):
        with patch("asil.adapters.obs.OBSAdapter") as mock_adapter:
            module._create_adapter("obs", tmp_path)

    mock_adapter.assert_called_once_with(
        host="obs-gui",
        port=5566,
        password="secret",
        use_real_gui=True,
        ws_protocol="v4",
    )


def test_create_adapter_obs_uses_local_socket_for_real_gui_inside_sandbox(tmp_path):
    module = _load_run_evaluation_module()
    sandbox = SimpleNamespace(obs_ws_host="obs-mock", obs_ws_port=4444)

    with patch.dict(
        "os.environ",
        {
            "OBS_REAL_GUI": "true",
            "OBS_WS_PASSWORD": "",
            "OBS_WS_PROTOCOL": "auto",
        },
        clear=False,
    ):
        with patch("asil.adapters.obs.OBSAdapter") as mock_adapter:
            module._create_adapter("obs", tmp_path, sandbox=sandbox)

    mock_adapter.assert_called_once_with(
        host="localhost",
        port=4444,
        password="",
        use_real_gui=True,
        ws_protocol="auto",
    )


def test_create_adapter_obs_uses_local_socket_for_real_gui_inside_managed_eval(tmp_path):
    module = _load_run_evaluation_module()

    with patch.dict(
        "os.environ",
        {
            "ASIL_SANDBOX": "true",
            "OBS_REAL_GUI": "true",
            "OBS_WS_HOST": "obs-mock",
            "OBS_WS_PORT": "4444",
            "OBS_WS_PASSWORD": "",
            "OBS_WS_PROTOCOL": "auto",
        },
        clear=False,
    ):
        with patch("asil.adapters.obs.OBSAdapter") as mock_adapter:
            module._create_adapter("obs", tmp_path)

    mock_adapter.assert_called_once_with(
        host="localhost",
        port=4444,
        password="",
        use_real_gui=True,
        ws_protocol="auto",
    )


def test_create_adapter_obs_mock_overrides_sandbox_and_real_gui(tmp_path):
    module = _load_run_evaluation_module()
    sandbox = SimpleNamespace(obs_ws_host="obs-mock", obs_ws_port=4444)

    with patch.dict(
        "os.environ",
        {
            "ASIL_SANDBOX": "true",
            "OBS_REAL_GUI": "true",
            "OBS_WS_HOST": "obs-mock",
            "OBS_WS_PORT": "4444",
        },
        clear=False,
    ):
        with patch("asil.adapters.obs.OBSAdapter") as mock_adapter, \
             patch("asil.adapters.obs.MockOBSWSClient") as mock_ws:
            module._create_adapter("obs", tmp_path, sandbox=sandbox, mock=True)

    mock_ws.assert_called_once_with()
    mock_adapter.assert_called_once_with(ws=mock_ws.return_value)


def test_create_adapter_gitea_reads_token_file(tmp_path):
    module = _load_run_evaluation_module()
    token_file = tmp_path / "gitea_token.txt"
    token_file.write_text("test-token\n")

    env = {
        "GITEA_URL": "http://gitea:3000",
        "GITEA_TOKEN": "",
        "GITEA_TOKEN_FILE": str(token_file),
        "GITEA_OWNER": "asil_admin",
        "GITEA_REPO": "test-repo",
    }
    with patch.dict("os.environ", env, clear=False):
        with patch("asil.adapters.gitea.GiteaAdapter") as mock_adapter:
            module._create_adapter("gitea", tmp_path)

    mock_adapter.assert_called_once_with(
        base_url="http://gitea:3000",
        token="test-token",
        owner="asil_admin",
        repo="test-repo",
    )


def test_create_adapter_gitea_requires_token(tmp_path):
    module = _load_run_evaluation_module()

    with patch.dict(
        "os.environ",
        {
            "GITEA_URL": "http://gitea:3000",
            "GITEA_TOKEN": "",
            "GITEA_TOKEN_FILE": "",
            "ASIL_SANDBOX": "false",
        },
        clear=False,
    ):
        try:
            module._create_adapter("gitea", tmp_path)
            assert False, "Expected RuntimeError when no Gitea token is available"
        except RuntimeError as e:
            assert "Gitea token not found" in str(e)


def test_software_choices_include_expansion_wave():
    module = _load_run_evaluation_module()

    assert {
        "gimp",
        "libreoffice_writer",
        "libreoffice_impress",
        "code_server",
        "thunderbird",
        "nautilus",
        "celluloid",
        "kdenlive",
        "audacity",
        "drawio",
        "jupyterlab",
    }.issubset(set(module.SOFTWARE_CHOICES))


def test_create_expansion_adapter_reports_missing_module(tmp_path):
    module = _load_run_evaluation_module()

    with patch("importlib.import_module", side_effect=ModuleNotFoundError("missing module")):
        try:
            module._create_adapter("gimp", tmp_path)
            assert False, "Expected RuntimeError when expansion adapter module is missing"
        except RuntimeError as e:
            assert "registered but its adapter module" in str(e)


def test_create_expansion_adapter_uses_from_evaluation_context(tmp_path):
    module = _load_run_evaluation_module()

    fake_module = types.ModuleType("asil.adapters.gimp")

    class GimpAdapter:
        @classmethod
        def from_evaluation_context(cls, tmp, sandbox=None, mock=False):
            return {"tmp": str(tmp), "sandbox": sandbox, "mock": mock}

    fake_module.GimpAdapter = GimpAdapter
    with patch.dict(sys.modules, {"asil.adapters.gimp": fake_module}):
        adapter = module._create_adapter("gimp", tmp_path, sandbox="sandbox", mock=True)

    assert adapter == {"tmp": str(tmp_path), "sandbox": "sandbox", "mock": True}


def test_create_adapter_supports_nautilus(tmp_path):
    module = _load_run_evaluation_module()

    fake_module = types.ModuleType("asil.adapters.nautilus")

    class NautilusAdapter:
        @classmethod
        def from_evaluation_context(cls, tmp, sandbox=None, mock=False):
            return {"tmp": str(tmp), "sandbox": sandbox, "mock": mock}

    fake_module.NautilusAdapter = NautilusAdapter
    with patch.dict(sys.modules, {"asil.adapters.nautilus": fake_module}):
        adapter = module._create_adapter("nautilus", tmp_path, sandbox="sandbox", mock=True)

    assert adapter == {"tmp": str(tmp_path), "sandbox": "sandbox", "mock": True}


def test_adapter_class_name_handles_libreoffice_prefix():
    module = _load_run_evaluation_module()

    assert module._adapter_class_name("libreoffice_writer") == "LibreOfficeWriterAdapter"
    assert module._adapter_class_name("libreoffice_impress") == "LibreOfficeImpressAdapter"
    assert module._adapter_class_name("nautilus") == "NautilusAdapter"
    assert module._adapter_class_name("celluloid") == "CelluloidAdapter"
    assert module._adapter_class_name("jupyterlab") == "JupyterLabAdapter"


def test_create_expansion_adapter_reports_missing_class(tmp_path):
    module = _load_run_evaluation_module()

    fake_module = types.ModuleType("asil.adapters.gimp")
    with patch.dict(sys.modules, {"asil.adapters.gimp": fake_module}):
        try:
            module._create_adapter("gimp", tmp_path)
            assert False, "Expected RuntimeError when expansion adapter class is missing"
        except RuntimeError as e:
            assert "class `GimpAdapter` was not found" in str(e)


def test_render_step_writes_png_metadata(tmp_path):
    module = _load_run_evaluation_module()

    class FakeAdapter:
        def describe_rendering(self):
            return RenderArtifact(
                filename="",
                kind="test_render",
                backend="fake",
                actual_page=True,
                capture_complete=True,
                description="fake render",
            )

        def render_to_png(self, output_path):
            self._last_capture_complete = False
            Path(output_path).write_text("png")

    artifact = module._render_step(FakeAdapter(), tmp_path, 3)

    assert artifact is not None
    assert (tmp_path / "step_3.png").exists()
    meta = json.loads((tmp_path / "step_3.render.json").read_text())
    assert meta["kind"] == "test_render"
    assert meta["actual_page"] is True
    assert meta["capture_complete"] is False


def test_run_agent_artifacts_write_task_info_and_step_action(tmp_path):
    module = _load_run_evaluation_module()

    class FakeAdapter:
        def __init__(self):
            self.source_path = None

        def observe(self):
            return Observation(
                meta=Meta(app_name="Inkscape", observation_source="test"),
                app_state=AppState(active_document="test.svg", document_path="test.svg"),
                interactive_elements=[
                    Element(id="rect1", type="rect", label="Rect 1", value={"width": "100"}, actions=["modify"]),
                ],
                environment=Environment(),
            )

        def execute(self, action):
            return self.observe()

        def describe_rendering(self):
            return RenderArtifact(
                filename="",
                kind="test_render",
                backend="fake",
                actual_page=True,
                capture_complete=True,
                description="fake render",
            )

        def render_to_png(self, output_path):
            Path(output_path).write_text("png")

    class FakeTask:
        id = "inkscape_01"
        software = "inkscape"
        difficulty = "simple"
        description = "Create a blue rectangle"
        instruction = "Create a blue rectangle"
        snapshot = "inkscape_svg_default"
        initial_state = "default"
        validation = {"element_exists": "rect1"}
        evaluator = {}
        render_target = {"mode": "single_slide", "slide_indices": [1]}
        gui_expectations = {
            "success_surface": "canvas",
            "visible_change_summary": "A blue rectangle is visible on the canvas",
            "checkpoint_visibility": {},
        }

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, *args, **kwargs):
            obs = FakeAdapter().observe()
            action = Action(action_type="modify_file", target="test.svg", params={"operations": []})
            trace = type(
                "Trace",
                (),
                {
                    "thought": "Inspect rect1, then keep the rectangle as required.",
                    "action": action,
                    "model_latency_ms": 12.5,
                    "action_execution_latency_ms": 34.5,
                    "render_latency_ms": 0.0,
                    "evaluation_latency_ms": 0.0,
                    "step_total_latency_ms": 47.0,
                },
            )()
            return obs, action, obs, trace

    with patch("asil.agent.ASILAgent", FakeAgent):
        result = module._run_agent_with_artifacts(
            FakeAdapter(),
            FakeTask(),
            llm_fn=lambda prompt: "",
            max_steps=1,
            task_dir=tmp_path,
        )

    assert result.task_id == "inkscape_01"
    task_info = json.loads((tmp_path / "task_info.json").read_text())
    assert task_info["task_name"] == "Create a blue rectangle"
    assert task_info["instruction"] == "Create a blue rectangle"
    assert task_info["gui_expectations"]["success_surface"] == "canvas"
    assert task_info["render_target"]["slide_indices"] == [1]

    step_action = json.loads((tmp_path / "step_1_action.json").read_text())
    assert step_action["instruction"] == "Create a blue rectangle"
    assert step_action["thought"].startswith("Inspect rect1")
    assert step_action["action"]["action_type"] == "modify_file"
    assert step_action["model_latency_ms"] >= 0
    assert step_action["step_total_latency_ms"] >= step_action["model_latency_ms"]

    traj_entries = (tmp_path / "traj.jsonl").read_text().strip().splitlines()
    assert len(traj_entries) == 1
    traj = json.loads(traj_entries[0])
    assert traj["step_action_file"] == "step_1_action.json"
    assert traj["thought"].startswith("Inspect rect1")
    assert traj["render_actual_page"] is True
    assert traj["render_capture_complete"] is True
    assert "model_latency_ms" in traj
    assert "render_latency_ms" in traj

    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert evaluation["success"] is True
    assert evaluation["mode"] in {"legacy_validation", "paths"}
    assert evaluation["migration_mode"] in {"legacy", "native", "synthesized"}


def test_run_agent_artifacts_can_disable_evaluator_hint_and_record_policy(tmp_path):
    module = _load_run_evaluation_module()
    captured_hints = []

    class FakeAdapter:
        def observe(self):
            return Observation(
                meta=Meta(app_name="Inkscape", observation_source="test"),
                app_state=AppState(active_document="test.svg", document_path="test.svg"),
                interactive_elements=[],
                environment=Environment(),
            )

        def execute(self, action):
            return self.observe()

        def describe_rendering(self):
            return RenderArtifact(
                filename="",
                kind="test_render",
                backend="fake",
                actual_page=True,
                capture_complete=True,
                description="fake render",
            )

        def render_to_png(self, output_path):
            Path(output_path).write_bytes(b"png")

    class FakeTask:
        id = "inkscape_hint_01"
        software = "inkscape"
        difficulty = "simple"
        description = "Keep the canvas unchanged"
        instruction = description
        snapshot = "default"
        initial_state = "default"
        validation = {}
        evaluator = {}
        render_target = {}
        gui_expectations = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, task_instruction, obs, *, success_hint):
            captured_hints.append(success_hint)
            action = Action(action_type="done", target="task", params={})
            trace = type(
                "Trace",
                (),
                {
                    "thought": "done",
                    "action": action,
                    "model_latency_ms": 1.0,
                    "action_execution_latency_ms": 0.0,
                    "render_latency_ms": 0.0,
                    "evaluation_latency_ms": 0.0,
                    "step_total_latency_ms": 1.0,
                },
            )()
            return obs, action, obs, trace

    with patch("asil.agent.ASILAgent", FakeAgent):
        module._run_agent_with_artifacts(
            FakeAdapter(),
            FakeTask(),
            llm_fn=lambda prompt: "",
            max_steps=1,
            task_dir=tmp_path,
            success_hint_policy="none",
        )

    assert captured_hints == [""]
    task_info = json.loads((tmp_path / "task_info.json").read_text())
    action_info = json.loads((tmp_path / "step_1_action.json").read_text())
    trajectory = json.loads((tmp_path / "traj.jsonl").read_text())
    assert task_info["asil_success_hint"] == "none"
    assert action_info["asil_success_hint"] == "none"
    assert trajectory["asil_success_hint"] == "none"


def test_run_agent_artifacts_can_write_independent_raw_evaluation(tmp_path):
    module = _load_run_evaluation_module()

    class FakeAdapter:
        def observe(self):
            return Observation(
                meta=Meta(app_name="Inkscape", observation_source="test"),
                app_state=AppState(active_document="test.svg", document_path="test.svg"),
                interactive_elements=[],
                environment=Environment(),
            )

        def describe_rendering(self):
            return RenderArtifact(
                filename="",
                kind="test_render",
                backend="fake",
                actual_page=False,
                capture_complete=True,
                description="fake render",
            )

        def render_to_png(self, output_path):
            Path(output_path).write_bytes(b"png")

    class FakeTask:
        id = "inkscape_raw_01"
        software = "inkscape"
        difficulty = "simple"
        description = "Done"
        instruction = "Done"
        snapshot = "default"
        initial_state = "default"
        validation = {}
        evaluator = {}
        render_target = {}
        gui_expectations = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, task_instruction, obs, *, success_hint):
            action = Action(action_type="done", target="task", params={})
            trace = SimpleNamespace(
                thought="done",
                action=action,
                model_latency_ms=1.0,
                action_execution_latency_ms=0.0,
                render_latency_ms=0.0,
                evaluation_latency_ms=0.0,
                step_total_latency_ms=1.0,
            )
            return obs, action, obs, trace

    raw_report = {
        "schema_version": "1.0",
        "complete": True,
        "score": 1.0,
        "evaluator_score": 1.0,
        "agreement": True,
    }
    with (
        patch("asil.agent.ASILAgent", FakeAgent),
        patch("asil.eval.raw_validation.validate_raw_final_state", return_value=raw_report) as validate,
    ):
        module._run_agent_with_artifacts(
            FakeAdapter(),
            FakeTask(),
            llm_fn=lambda prompt: "",
            max_steps=1,
            task_dir=tmp_path,
            independent_raw_validation=True,
        )

    validate.assert_called_once()
    assert json.loads((tmp_path / "independent_evaluation.json").read_text()) == raw_report
    assert json.loads((tmp_path / "task_info.json").read_text())["independent_raw_validation"] is True


def test_run_evaluation_agent_mode_handles_directory_backed_adapter(tmp_path):
    module = _load_run_evaluation_module()

    class DirectoryAdapter:
        def __init__(self, root: Path):
            self.source_path = root
            self.reset_calls = 0
            self.setup_calls: list[str] = []
            self.root = root
            root.mkdir(parents=True, exist_ok=True)

        def reset_state(self):
            self.reset_calls += 1

        def setup_state(self, initial_state):
            self.setup_calls.append(initial_state)

    fake_task = SimpleNamespace(
        id="code_server_01",
        software="code_server",
        difficulty="simple",
        description="Update the README",
        instruction="Update the README",
        actions=[{"action_type": "modify_file", "target": "workspace", "params": {"operations": []}}],
        initial_state="default",
    )

    adapter = DirectoryAdapter(tmp_path / "workspace")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_all.json").write_text(
        json.dumps({"code_server": ["code_server_01"]}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        software=["code_server"],
        output=tmp_path / "results.json",
        output_dir=tmp_path / "agent_results",
        agent=True,
        comparison=False,
        provider="mock",
        model=None,
        max_steps=1,
        test_config_base_dir=config_dir,
        test_all="test_all.json",
        osworld_format=False,
        docker=False,
        mock=False,
        dry_run=False,
    )

    with patch("asil.agent.create_llm_fn", return_value=lambda prompt: ""), \
         patch.object(module, "_create_adapter", return_value=adapter), \
         patch("asil.eval.runner.TaskDefinition.from_index", return_value=[fake_task]), \
         patch.object(
             module,
             "_run_agent_with_artifacts",
             return_value=TaskResult(task_id="code_server_01", software="code_server", success=True, score=1.0, steps=1, e2e_time_s=0.01),
         ):
        module._run_evaluation(args, sandbox=None)

    assert adapter.reset_calls == 1
    assert adapter.setup_calls == ["default"]


def test_gui_evaluation_creates_task_local_llm_instances(tmp_path):
    module = _load_run_evaluation_module()

    tasks = [
        SimpleNamespace(
            id=task_id,
            software="code_server",
            difficulty="simple",
            description=f"Task {task_id}",
            instruction=f"Instruction {task_id}",
            actions=[{"action_type": "noop", "target": "workspace", "params": {}}],
            initial_state="default",
        )
        for task_id in ("code_server_01", "code_server_02")
    ]

    class FakeAdapter:
        app_name = "CodeServer"

        def prepare_task(self, task):
            del task

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_all.json").write_text(
        json.dumps({"code_server": [task.id for task in tasks]}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        software=["code_server"],
        output=tmp_path / "results.json",
        output_dir=tmp_path / "gui_results",
        participant="gui",
        run_mode="single",
        agent=False,
        comparison=False,
        provider="openai",
        model="gpt-5.4",
        max_steps=50,
        test_config_base_dir=config_dir,
        test_all="test_all.json",
        osworld_format=True,
        docker=False,
        mock=False,
        dry_run=False,
        force_rerun=True,
    )
    created_llms = []
    llms_seen_by_tasks = []

    def fake_create_gui_llm_fn(**kwargs):
        llm = object()
        created_llms.append((llm, kwargs))
        return llm

    def fake_run_gui_agent_task(adapter, task, llm_fn, *, max_steps, task_dir):
        del adapter, max_steps, task_dir
        llms_seen_by_tasks.append((task.id, llm_fn))
        return TaskResult(
            task_id=task.id,
            software=task.software,
            difficulty=task.difficulty,
            instruction=task.instruction,
            success=True,
            score=1.0,
            steps=1,
            e2e_time_s=0.01,
        )

    with patch.object(module, "_create_adapter", return_value=FakeAdapter()), \
         patch("asil.eval.runner.TaskDefinition.from_index", return_value=tasks), \
         patch("asil.gui_agent.create_gui_llm_fn", side_effect=fake_create_gui_llm_fn), \
         patch("asil.gui_agent.run_gui_agent_task", side_effect=fake_run_gui_agent_task):
        module._run_evaluation(args, sandbox=None)

    assert len(created_llms) == 2
    assert [kwargs for _, kwargs in created_llms] == [
        {"provider": "openai", "model": "gpt-5.4"},
        {"provider": "openai", "model": "gpt-5.4"},
    ]
    assert [task_id for task_id, _ in llms_seen_by_tasks] == [task.id for task in tasks]
    assert llms_seen_by_tasks[0][1] is created_llms[0][0]
    assert llms_seen_by_tasks[1][1] is created_llms[1][0]
    assert llms_seen_by_tasks[0][1] is not llms_seen_by_tasks[1][1]


def test_run_evaluation_deterministic_osworld_format_writes_task_dirs_and_summary(tmp_path):
    module = _load_run_evaluation_module()

    fake_task = SimpleNamespace(
        id="code_server_01",
        software="code_server",
        difficulty="simple",
        description="Update the README",
        instruction="Update the README",
        actions=[{"action_type": "modify_file", "target": "workspace", "params": {"operations": []}}],
        initial_state="default",
    )

    result = TaskResult(
        task_id="code_server_01",
        software="code_server",
        difficulty="simple",
        instruction="Update the README",
        success=True,
        score=1.0,
        steps=1,
        e2e_time_s=0.01,
        step_results=[
            StepResult(
                step_num=1,
                action_type="modify_file",
                target="workspace",
                params={"operations": []},
                success=True,
                latency_ms=12.0,
                observation_element_count=3,
                observation_source="file_parse",
            )
        ],
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_all.json").write_text(
        json.dumps({"code_server": ["code_server_01"]}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        software=["code_server"],
        output=tmp_path / "det_results" / "flat.json",
        output_dir=tmp_path / "det_results",
        agent=False,
        comparison=False,
        provider="mock",
        model=None,
        max_steps=20,
        test_config_base_dir=config_dir,
        test_all="test_all.json",
        osworld_format=True,
        docker=False,
        mock=False,
        dry_run=False,
    )

    with patch.object(module, "_create_adapter", return_value=SimpleNamespace()), \
         patch("asil.eval.runner.TaskDefinition.from_index", return_value=[fake_task]), \
         patch("asil.eval.runner.run_evaluation", return_value=[result]):
        module._run_evaluation(args, sandbox=None)

    result_root = tmp_path / "det_results" / "semantic" / "structured" / "asil-deterministic"
    task_dir = result_root / "code_server" / "code_server_01"
    assert (task_dir / "traj.jsonl").exists()
    assert (task_dir / "result.txt").read_text(encoding="utf-8") == "1.0"
    evaluation = json.loads((task_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["task_id"] == "code_server_01"

    summary_dir = result_root / "summary"
    assert (summary_dir / "results.json").exists()
    assert (summary_dir / "aggregate.json").exists()

    flat = json.loads((tmp_path / "det_results" / "flat.json").read_text(encoding="utf-8"))
    assert flat["code_server"]["aggregate"]["passed"] == 1


def test_run_evaluation_forces_obs_mock_in_deterministic_mode(tmp_path):
    module = _load_run_evaluation_module()

    fake_task = SimpleNamespace(
        id="obs_01",
        software="obs",
        difficulty="simple",
        description="Switch scene named 'Main Scene'",
        instruction="Switch scene named 'Main Scene'",
        actions=[{"action_type": "api_call", "target": "obs-websocket", "params": {"method": "SetCurrentProgramScene", "args": {"sceneName": "Main Scene"}}}],
        initial_state="default",
    )

    result = TaskResult(
        task_id="obs_01",
        software="obs",
        difficulty="simple",
        instruction="Switch scene named 'Main Scene'",
        success=True,
        score=1.0,
        steps=1,
        e2e_time_s=0.01,
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "test_all.json").write_text(
        json.dumps({"obs": ["obs_01"]}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        software=["obs"],
        output=tmp_path / "det_results" / "flat.json",
        output_dir=tmp_path / "det_results",
        agent=False,
        comparison=False,
        provider="mock",
        model=None,
        max_steps=20,
        test_config_base_dir=config_dir,
        test_all="test_all.json",
        osworld_format=True,
        docker=False,
        mock=False,
        dry_run=False,
    )

    with patch.object(module, "_create_adapter", return_value=SimpleNamespace()) as mock_create_adapter, \
         patch("asil.eval.runner.TaskDefinition.from_index", return_value=[fake_task]), \
         patch("asil.eval.runner.run_evaluation", return_value=[result]):
        module._run_evaluation(args, sandbox=None)

    mock_create_adapter.assert_called_once()
    assert mock_create_adapter.call_args.kwargs["mock"] is True
