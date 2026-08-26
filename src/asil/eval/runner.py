"""ASIL task runner — executes task definitions and collects metrics.

Aligned with OSWorld's evaluation loop:
- Per-step trajectory recording
- Agent-based execution (observe → think → act loop)
- Three-way comparison support with state isolation
- OSWorld-style result directory output
"""

from __future__ import annotations
import copy
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asil.adapter import ASILAdapter
from asil.eval.evaluator import evaluate_task_result, legacy_validate
from asil.eval.metrics import TaskResult, StepResult, compute_metrics
from asil.protocol import Action


@dataclass
class TaskDefinition:
    """A single evaluation task — aligned with OSWorld's task format.

    Fields mirror OSWorld: id, instruction, config (setup), evaluator (validation).
    Supports two JSON formats:
    - Legacy bundled: {"software": "...", "tasks": [...]}
    - OSWorld-style single-task: {"id": "...", "_asil": {...}, ...}
    """
    id: str
    software: str
    difficulty: str  # simple | medium | complex
    description: str
    instruction: str = ""  # Natural language instruction for the agent
    actions: list[dict[str, Any]] = field(default_factory=list)  # Ground truth actions (for deterministic eval)
    config: list[dict[str, Any]] = field(default_factory=list)  # Environment setup steps
    validation: dict[str, Any] = field(default_factory=dict)  # Evaluator specification
    related_apps: list[str] = field(default_factory=list)
    app_initial_states: dict[str, str] = field(default_factory=dict)
    primary_app: str = ""
    snapshot: str = ""  # Environment snapshot name
    initial_state: str = "default"  # "default" | "blank" — which initial file to use
    gui_expectations: dict[str, Any] = field(default_factory=dict)
    render_target: dict[str, Any] = field(default_factory=dict)
    # OSWorld-compatible fields
    source: str = ""
    evaluator: dict[str, Any] = field(default_factory=dict)
    proxy: bool = False
    fixed_ip: bool = False
    possibility_of_env_change: str = "low"

    def __post_init__(self):
        # instruction defaults to description if not specified
        if not self.instruction:
            self.instruction = self.description

    @classmethod
    def from_json(cls, path: Path) -> list["TaskDefinition"]:
        """Load tasks from a JSON file. Auto-detects format:
        - Legacy bundled: {"tasks": [...]} or bare list
        - OSWorld single-task: {"id": "...", "_asil": {...}}
        """
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return [cls._from_task_dict(t) for t in data]
        if "tasks" in data:
            return [cls._from_task_dict(t) for t in data["tasks"]]
        if "_asil" in data or "id" in data:
            return [cls._from_osworld_dict(data)]
        raise ValueError(f"Unrecognized task file format: {path}")

    @classmethod
    def _from_task_dict(cls, t: dict) -> "TaskDefinition":
        """Construct from legacy bundled task dict."""
        return cls(**{k: v for k, v in t.items() if k in cls.__dataclass_fields__})

    @classmethod
    def _from_osworld_dict(cls, data: dict) -> "TaskDefinition":
        """Construct from OSWorld-style single-task dict with _asil namespace."""
        asil = data.get("_asil", {})
        return cls(
            id=data["id"],
            software=asil.get("software", data.get("software", data.get("related_apps", ["unknown"])[0])),
            difficulty=asil.get("difficulty", data.get("difficulty", "unknown")),
            description=asil.get("description", data.get("description", data.get("instruction", ""))),
            instruction=data.get("instruction", asil.get("description", "")),
            actions=asil.get("actions", data.get("actions", [])),
            config=data.get("config", []),
            validation=asil.get("validation", data.get("validation", {})),
            related_apps=data.get("related_apps", [data.get("software")] if data.get("software") else []),
            app_initial_states=asil.get("app_initial_states", data.get("app_initial_states", {})),
            primary_app=asil.get("primary_app", data.get("primary_app", "")),
            snapshot=data.get("snapshot", ""),
            initial_state=asil.get("initial_state", data.get("initial_state", "default")),
            gui_expectations=data.get("gui_expectations", {}),
            render_target=data.get("render_target", {}),
            source=data.get("source", ""),
            evaluator=data.get("evaluator", {}),
            proxy=data.get("proxy", False),
            fixed_ip=data.get("fixed_ip", False),
            possibility_of_env_change=data.get("possibility_of_env_change", "low"),
        )

    @classmethod
    def from_index(cls, index_path: Path, domain: str | None = None) -> list["TaskDefinition"]:
        """Load tasks from an OSWorld-style index file (test_all.json / test_small.json).

        Args:
            index_path: Path to the index JSON file.
            domain: If specified, load only tasks for this software domain.
        """
        base_dir = index_path.parent / "examples"
        with open(index_path) as f:
            index = json.load(f)

        tasks: list["TaskDefinition"] = []
        domains = [domain] if domain else list(index.keys())
        for d in domains:
            for task_id in index.get(d, []):
                task_file = base_dir / d / f"{task_id}.json"
                tasks.extend(cls.from_json(task_file))
        return tasks


# ---------------------------------------------------------------------------
# Placeholder resolution
# ---------------------------------------------------------------------------

def resolve_placeholders(obj: Any, context: dict[str, str]) -> Any:
    """Recursively substitute {{key}} placeholders in action spec dicts/strings.

    Args:
        obj: An action spec dict, list, or string (or any nested combination).
        context: Mapping from placeholder key to replacement value.

    Returns:
        A deep copy of obj with all {{key}} patterns substituted.
    """
    if isinstance(obj, str):
        def _replace(m: re.Match) -> str:
            return context.get(m.group(1), m.group(0))
        return re.sub(r"\{\{(\w+)\}\}", _replace, obj)
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(item, context) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _parse_kv_string(text: str, key: str) -> str | None:
    """Parse 'k1=v1 k2=v2' format and return the value for the given key."""
    for pair in text.split():
        if "=" in pair:
            k, _, v = pair.partition("=")
            if k == key:
                return v
    return None


def _find_element(obs, target_id: str):
    """Find an element by exact ID, or by type-prefix match for service-based adapters.

    For resource types whose IDs are globally auto-incremented (label, milestone),
    the numeric suffix may differ across runs. If the exact ID is not found, fall
    back to the first element whose ID starts with the same type prefix.

    This allows validation rules like {"id": "label:1"} to match "label:5" when
    the repo was recreated and the label counter did not reset.
    """
    exact = next((e for e in obs.interactive_elements if e.id == target_id), None)
    if exact is not None:
        return exact
    # Fallback: match by type prefix for non-sequential ID types
    prefix = target_id.rsplit(":", 1)[0] + ":"
    if prefix in ("label:", "milestone:"):
        return next((e for e in obs.interactive_elements if e.id.startswith(prefix)), None)
    return None


def _check_single(obs, rule: dict[str, Any]) -> bool:
    """Check a single validation rule against an observation.

    Supported rule types (each rule dict contains exactly one key):
    - element_exists: "id"
    - element_not_exists: "id"
    - element_value: {id, key, expected}   — exact match
    - element_contains: {id, key, expected} — substring match
    - count_equals: int
    - count_at_least: int
    """
    ids = {e.id for e in obs.interactive_elements}

    if "element_exists" in rule:
        return rule["element_exists"] in ids

    if "element_not_exists" in rule:
        return rule["element_not_exists"] not in ids

    if "element_value" in rule:
        spec = rule["element_value"]
        target_id = spec["id"]
        key = spec.get("key")
        expected = spec["expected"]
        elem = _find_element(obs, target_id)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            actual = elem.value.get(key) if key else elem.value
        elif key and isinstance(elem.value, str):
            actual = _parse_kv_string(elem.value, key)
        else:
            actual = elem.value
        return str(actual) == str(expected)

    if "element_contains" in rule:
        spec = rule["element_contains"]
        target_id = spec["id"]
        key = spec.get("key")
        expected = spec["expected"]
        elem = _find_element(obs, target_id)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            actual = elem.value.get(key, "") if key else str(elem.value)
        elif key and isinstance(elem.value, str):
            actual = _parse_kv_string(elem.value, key) or ""
        else:
            actual = str(elem.value)
        return expected in str(actual)

    if "any_element_contains" in rule:
        # Check if ANY element of the given type has a field containing the expected value.
        # Spec: {"type": "comment", "key": "body", "expected": "some text"}
        spec = rule["any_element_contains"]
        elem_type = spec.get("type")
        key = spec.get("key")
        expected = spec["expected"]
        for e in obs.interactive_elements:
            if elem_type and e.type != elem_type:
                continue
            if isinstance(e.value, dict):
                actual = e.value.get(key, "") if key else str(e.value)
            else:
                actual = str(e.value)
            if expected in str(actual):
                return True
        return False

    if "any_element_value" in rule:
        # Check if ANY element of the given type has a field equal to the expected value.
        # Spec: {"type": "milestone", "key": "open_issues", "expected": 1}
        spec = rule["any_element_value"]
        elem_type = spec.get("type")
        key = spec.get("key")
        expected = spec["expected"]
        for e in obs.interactive_elements:
            if elem_type and e.type != elem_type:
                continue
            if isinstance(e.value, dict):
                actual = e.value.get(key) if key else e.value
            else:
                actual = e.value
            if str(actual) == str(expected):
                return True
        return False

    if "count_equals" in rule:
        return len(obs.interactive_elements) == rule["count_equals"]

    if "count_at_least" in rule:
        return len(obs.interactive_elements) >= rule["count_at_least"]

    # --- Blender-specific validation types ---

    if "object_exists" in rule:
        # Check if an object with the given name exists in the scene
        name = rule["object_exists"]
        return any(e.id == name for e in obs.interactive_elements)

    if "scene_object_count" in rule:
        # Count scene objects (exclude settings_group elements like render_settings)
        expected = rule["scene_object_count"]
        count = sum(1 for e in obs.interactive_elements if e.type != "settings_group")
        return count == expected

    if "material_exists" in rule:
        # Check if any object has a material with the given name
        name = rule["material_exists"]
        for e in obs.interactive_elements:
            if isinstance(e.metadata, dict):
                for mat in e.metadata.get("materials", []):
                    if mat.get("name") == name:
                        return True
        return False

    if "render_setting" in rule:
        # Check a render setting value: {"key": "resolution_x", "expected": 1920}
        spec = rule["render_setting"]
        key = spec["key"]
        expected = spec["expected"]
        rs = next((e for e in obs.interactive_elements if e.id == "render_settings"), None)
        if rs is None or not isinstance(rs.value, dict):
            return False
        return rs.value.get(key) == expected

    if "object_type_exists" in rule:
        # Check if any object of the given type exists (e.g. "LIGHT", "CAMERA")
        obj_type = rule["object_type_exists"].lower()
        return any(e.type == obj_type for e in obs.interactive_elements)

    if "animation_data_exists" in rule:
        # Check if the named object has animation data
        name = rule["animation_data_exists"]
        elem = next((e for e in obs.interactive_elements if e.id == name), None)
        if elem is None:
            return False
        if isinstance(elem.metadata, dict):
            return elem.metadata.get("has_animation_data", False)
        return False

    # --- OBS-specific validation types ---

    if "current_scene" in rule:
        expected = rule["current_scene"]
        return obs.app_state.current_view == expected

    if "source_visible" in rule:
        spec = rule["source_visible"]
        scene = spec["scene"]
        source = spec["source"]
        expected = spec["expected"]
        elem_id = f"scene:{scene}/source:{source}"
        elem = next((e for e in obs.interactive_elements if e.id == elem_id), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("visible") == expected
        return False

    if "response_has_field" in rule:
        # For OBS: just check that the observation has elements (the API responded)
        return len(obs.interactive_elements) > 0

    if "input_muted" in rule:
        spec = rule["input_muted"]
        name = spec["name"]
        expected = spec["expected"]
        elem = next((e for e in obs.interactive_elements if e.id == f"input:{name}"), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("muted") == expected
        return False

    if "input_volume_db" in rule:
        spec = rule["input_volume_db"]
        name = spec["name"]
        expected = spec["expected"]
        elem = next((e for e in obs.interactive_elements if e.id == f"input:{name}"), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            actual = elem.value.get("volumeDb", 0.0)
            return abs(float(actual) - float(expected)) < 0.1
        return False

    if "stream_active" in rule:
        expected = rule["stream_active"]
        elem = next((e for e in obs.interactive_elements if e.id == "stream_status"), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("active") == expected
        return False

    if "record_active" in rule:
        expected = rule["record_active"]
        elem = next((e for e in obs.interactive_elements if e.id == "record_status"), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("active") == expected
        return False

    if "scene_collection" in rule:
        expected = rule["scene_collection"]
        elem = next((e for e in obs.interactive_elements if e.id == "scene_collection"), None)
        if elem is None:
            return False
        if isinstance(elem.value, dict):
            return elem.value.get("name") == expected
        return elem.label == expected

    return True  # unknown rule key — pass


def _validate(obs, validation: dict[str, Any]) -> bool:
    """Compatibility wrapper for legacy validation tests."""
    return legacy_validate(obs, validation)


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def run_task(adapter: ASILAdapter, task: TaskDefinition) -> TaskResult:
    """Execute a task against an adapter with per-step recording.

    Follows OSWorld's loop: observe → act → observe → validate.
    Resolves {{placeholder}} in action specs using adapter.get_context().
    """
    start = time.monotonic()
    prepare_task = getattr(adapter, "prepare_task", None)
    if callable(prepare_task) and getattr(adapter, "_prepared_task_id", None) != task.id:
        prepare_task(task)
        try:
            setattr(adapter, "_prepared_task_id", task.id)
        except Exception:
            pass
    deadlocked = False
    last_obs_hash = ""
    repeat_count = 0
    step_results: list[StepResult] = []
    context = adapter.get_context() if hasattr(adapter, "get_context") else {}

    obs = adapter.observe()
    obs_element_count = len(obs.interactive_elements)

    for i, action_spec in enumerate(task.actions):
        resolved = resolve_placeholders(action_spec, context)
        action = Action(**resolved)
        step_start = time.monotonic()

        try:
            obs = adapter.execute(action)
            step_success = True
        except Exception:
            obs = adapter.observe()
            step_success = False

        latency = (time.monotonic() - step_start) * 1000

        # Deadlock detection
        obs_hash = str(sorted(e.id for e in obs.interactive_elements))
        if obs_hash == last_obs_hash:
            repeat_count += 1
            if repeat_count >= 5:
                deadlocked = True
        else:
            repeat_count = 0
            last_obs_hash = obs_hash

        step_results.append(StepResult(
            step_num=i + 1,
            action_type=action.action_type,
            target=action.target,
            params=action.params,
            success=step_success,
            latency_ms=latency,
            observation_element_count=len(obs.interactive_elements),
            observation_source=obs.meta.observation_source,
        ))

    e2e = time.monotonic() - start
    evaluation = evaluate_task_result(task, obs)

    return TaskResult(
        task_id=task.id,
        software=task.software,
        difficulty=task.difficulty,
        instruction=task.instruction,
        success=evaluation.success,
        score=evaluation.score,
        steps=len(task.actions),
        step_results=step_results,
        e2e_time_s=e2e,
        deadlocked=deadlocked,
        observation_element_count=obs_element_count,
        total_element_count=obs_element_count,
    )


def run_task_with_agent(
    adapter: ASILAdapter,
    task: TaskDefinition,
    llm_fn,
    max_steps: int = 20,
) -> TaskResult:
    """Execute a task using agent loop (observe → think → act).

    This mirrors OSWorld's agent-based evaluation where the agent
    decides actions based on observations, not predefined ground truth.
    """
    from asil.agent import ASILAgent, format_task_success_hint

    start = time.monotonic()
    agent = ASILAgent(adapter=adapter, llm_fn=llm_fn, max_steps=max_steps, software=task.software)
    success_hint = format_task_success_hint(task)
    step_results: list[StepResult] = []
    deadlocked = False
    last_obs_hash = ""
    repeat_count = 0

    obs = adapter.observe()
    obs_element_count = len(obs.interactive_elements)

    for i in range(max_steps):
        step_start = time.monotonic()

        try:
            _, action, new_obs, _trace = agent.step(task.instruction, obs, success_hint=success_hint)
            step_success = True
        except Exception:
            step_success = False
            break

        latency = (time.monotonic() - step_start) * 1000

        # Deadlock detection
        obs_hash = str(sorted(e.id for e in new_obs.interactive_elements))
        if obs_hash == last_obs_hash:
            repeat_count += 1
            if repeat_count >= 5:
                deadlocked = True
                break
        else:
            repeat_count = 0
            last_obs_hash = obs_hash

        step_results.append(StepResult(
            step_num=i + 1,
            action_type=action.action_type,
            target=action.target,
            params=action.params,
            success=step_success,
            latency_ms=latency,
            observation_element_count=len(new_obs.interactive_elements),
            observation_source=new_obs.meta.observation_source,
        ))

        obs = new_obs
        if action.action_type == "done":
            break

    e2e = time.monotonic() - start
    evaluation = evaluate_task_result(task, obs)

    return TaskResult(
        task_id=task.id,
        software=task.software,
        difficulty=task.difficulty,
        instruction=task.instruction,
        success=evaluation.success,
        score=evaluation.score,
        steps=len(step_results),
        step_results=step_results,
        e2e_time_s=e2e,
        deadlocked=deadlocked,
        observation_element_count=obs_element_count,
        total_element_count=obs_element_count,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    adapter: ASILAdapter,
    tasks: list[TaskDefinition],
    isolate_tasks: bool = False,
) -> list[TaskResult]:
    """Run a batch of tasks against an adapter and collect results.

    Args:
        adapter: The adapter to evaluate.
        tasks: List of task definitions to run.
        isolate_tasks: If True, restore the adapter's source file to its
            original content before each task. Ensures tasks are independent.
    """
    results = []

    # Save original file content for restoration
    original_bytes: bytes | None = None
    initial_state_bytes: dict[str, bytes] = {}
    src = getattr(adapter, "source_path", None)
    if isolate_tasks and src is not None and src.exists() and src.is_file():
        original_bytes = src.read_bytes()
        if hasattr(adapter, "setup_state"):
            needed_states = {
                getattr(task, "initial_state", "default") or "default"
                for task in tasks
            }
            for state_key in sorted(needed_states):
                adapter.setup_state(state_key)
                if src.exists():
                    initial_state_bytes[state_key] = src.read_bytes()
            if "default" in initial_state_bytes:
                src.write_bytes(initial_state_bytes["default"])
            elif original_bytes is not None:
                src.write_bytes(original_bytes)

    for task in tasks:
        prepare_task = getattr(adapter, "prepare_task", None)
        if callable(prepare_task):
            prepare_task(task)
            try:
                setattr(adapter, "_prepared_task_id", task.id)
            except Exception:
                pass
            result = run_task(adapter, task)
            results.append(result)
            continue
        if original_bytes is not None and src is not None:
            initial_state = getattr(task, "initial_state", "default") or "default"
            task_bytes = initial_state_bytes.get(initial_state, original_bytes)
            src.write_bytes(task_bytes)
        elif isolate_tasks and hasattr(adapter, "reset_state"):
            adapter.reset_state()
            initial_state = getattr(task, "initial_state", "default") or "default"
            if hasattr(adapter, "setup_state"):
                adapter.setup_state(initial_state)
        result = run_task(adapter, task)
        results.append(result)

    return results


def _clone_adapter_for_method(
    adapter: ASILAdapter,
    method: str,
    work_dir: Path,
) -> ASILAdapter:
    """Create an isolated copy of the adapter for a given comparison method.

    For file-based adapters, copies the source file into work_dir/method/.
    For service-based adapters (no source_path), returns the adapter as-is.
    """
    src = getattr(adapter, "source_path", None)
    if src is None:
        return adapter  # service-based — shared, no file to copy
    dest_dir = work_dir / method
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    return adapter.clone(dest)


def run_comparison(
    asil_adapter: ASILAdapter,
    cli_baseline: ASILAdapter,
    gui_agent: ASILAdapter,
    tasks: list[TaskDefinition],
    work_dir: Path | None = None,
) -> dict:
    """Run three-way comparison: ASIL vs CLI vs GUI Agent.

    Each method runs against its own isolated copy of the source file
    so that mutations from one method do not affect others.

    Returns OSWorld-style structured results with per-task trajectories.
    """
    _tmp_dir = None
    if work_dir is None:
        _tmp_dir = tempfile.mkdtemp()
        work_dir = Path(_tmp_dir)

    try:
        # Determine the underlying file adapter for each method.
        # CLIBaseline / GUIAgentBaseline wrap a delegate; clone the delegate.
        asil_iso = _clone_adapter_for_method(asil_adapter, "asil", work_dir)
        cli_iso = _clone_adapter_for_method(cli_baseline, "cli", work_dir)
        gui_iso = _clone_adapter_for_method(gui_agent, "gui", work_dir)

        asil_results = run_evaluation(asil_iso, tasks, isolate_tasks=True)
        cli_results = run_evaluation(cli_iso, tasks, isolate_tasks=True)
        gui_results = run_evaluation(gui_iso, tasks, isolate_tasks=True)
    finally:
        if _tmp_dir:
            shutil.rmtree(_tmp_dir, ignore_errors=True)

    return {
        "asil": {"results": asil_results, "metrics": compute_metrics(asil_results)},
        "cli": {"results": cli_results, "metrics": compute_metrics(cli_results)},
        "gui": {"results": gui_results, "metrics": compute_metrics(gui_results)},
    }


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def save_results(comparison: dict, output_path: Path) -> None:
    """Save comparison results to JSON with full trajectory data."""
    serializable = {}
    for method, data in comparison.items():
        metrics = data["metrics"]
        serializable[method] = {
            "aggregate": metrics.get("aggregate", {}),
            "per_software": metrics.get("per_software", {}),
            "tasks": [r.to_dict() for r in data["results"]],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))


def save_results_osworld_style(
    comparison: dict,
    output_dir: Path,
    method_name: str = "",
) -> None:
    """Save results in OSWorld-compatible directory structure.

    Output structure:
        {output_dir}/
            {method}/
                {software}/
                    {task_id}/
                        traj.jsonl
                        result.txt
            summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, data in comparison.items():
        for result in data["results"]:
            task_dir = output_dir / method / result.software / result.task_id
            result.save_trajectory(task_dir)
            result.save_result(task_dir)

    # Write summary
    summary = {}
    for method, data in comparison.items():
        metrics = data["metrics"]
        summary[method] = metrics.get("aggregate", {})
        summary[method]["per_software"] = metrics.get("per_software", {})

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
