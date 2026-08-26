"""Legacy simulated GUI baseline.

Kept only for historical compatibility tests. Public `participant=gui`
now refers to the real screenshot-driven GUI agent under `asil.gui_agent`.
"""

from __future__ import annotations
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from asil.adapter import ASILAdapter
from asil.protocol import Action, Element, Observation, Meta, AppState, Environment


class SimulatedGUIBaseline(ASILAdapter):
    """Simulates a screenshot-based GUI agent.

    Key limitations modeled (aligned with OSWorld benchmark findings):
    - observe() only captures ~40-60% of elements (visible on screen)
    - Values are approximated (visual estimation, not exact)
    - execute() has ~15% failure rate (coordinate grounding errors)
    - No background state (undo stack, processes, errors)

    Outputs OSWorld-compatible traj.jsonl via save_trajectory().
    """

    app_name = "GUI-Agent"
    supported_action_types = ["modify_file", "set_value", "invoke_function", "api_call", "navigate"]

    def __init__(
        self,
        delegate: ASILAdapter,
        visibility_ratio: float = 0.5,
        action_failure_rate: float = 0.15,
        seed: int = 42,
    ) -> None:
        self._delegate = delegate
        self._visibility_ratio = visibility_ratio
        self._failure_rate = action_failure_rate
        self._rng = random.Random(seed)
        self.app_name = f"GUI-{delegate.app_name}"
        # Per-step history for trajectory analysis
        self._step_log: list[dict[str, Any]] = []
        # OSWorld-compatible trajectory
        self._trajectory: list[dict[str, Any]] = []

    @property
    def source_path(self) -> Path | None:
        return self._delegate.source_path

    def clone(self, new_path: Path) -> "SimulatedGUIBaseline":
        return SimulatedGUIBaseline(
            self._delegate.clone(new_path),
            visibility_ratio=self._visibility_ratio,
            action_failure_rate=self._failure_rate,
            seed=self._rng.randint(0, 2**32),
        )

    def get_context(self) -> dict[str, str]:
        return self._delegate.get_context()

    @property
    def step_log(self) -> list[dict[str, Any]]:
        """Access per-step grounding error log."""
        return list(self._step_log)

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """Access per-step OSWorld-format trajectory."""
        return list(self._trajectory)

    def reset(self) -> None:
        """Reset trajectory and step log for a new task."""
        self._step_log.clear()
        self._trajectory.clear()

    def observe(self) -> Observation:
        full_obs = self._delegate.observe()
        return self._observe_from(full_obs)

    def execute(self, action: Action) -> Observation:
        pre_obs = self._delegate.observe()
        will_fail = self._rng.random() < self._failure_rate
        step_num = len(self._trajectory) + 1

        if will_fail:
            error_types = ["click_miss", "wrong_element", "timeout"]
            error_type = self._rng.choice(error_types)
            self._step_log.append({
                "action_type": action.action_type,
                "target": action.target,
                "grounding_error": True,
                "error_type": error_type,
            })
            result_obs = self._observe_from(pre_obs)
            self._trajectory.append(self._make_traj_entry(
                step_num, action, grounding_error=True, error_type=error_type,
                element_count=len(result_obs.interactive_elements),
            ))
            return result_obs

        self._step_log.append({
            "action_type": action.action_type,
            "target": action.target,
            "grounding_error": False,
            "error_type": "",
        })
        self._delegate.execute(action)
        result_obs = self._observe_from(self._delegate.observe())
        self._trajectory.append(self._make_traj_entry(
            step_num, action, grounding_error=False, error_type="",
            element_count=len(result_obs.interactive_elements),
        ))
        return result_obs

    def _make_traj_entry(
        self,
        step_num: int,
        action: Action,
        grounding_error: bool,
        error_type: str,
        element_count: int,
    ) -> dict[str, Any]:
        """Build an OSWorld-compatible traj.jsonl entry."""
        return {
            "step_num": step_num,
            "action_timestamp": datetime.now().isoformat(),
            "action": {
                "action_type": action.action_type,
                "target": action.target,
                "params": action.params,
            },
            "response": "",  # LLM response — empty in simulation
            "reward": 0.0,   # Unknown until final validation
            "done": False,
            "screenshot_file": "",  # No real screenshot in simulation
            "observation_source": "screenshot",
            "observation_element_count": element_count,
            "grounding_error": grounding_error,
            "grounding_error_type": error_type,
        }

    def _observe_from(self, full_obs: Observation) -> Observation:
        """Build a GUI-degraded observation from a given full observation.

        Models real screenshot-based agent limitations:
        1. Only visible elements captured (visibility_ratio)
        2. Dict values approximated (visual estimation)
        3. Environment state invisible (empty Environment)
        """
        visible_elements = []
        for e in full_obs.interactive_elements:
            if self._rng.random() < self._visibility_ratio:
                approx_value = e.value
                if isinstance(e.value, (int, float)):
                    approx_value = round(e.value * (1 + self._rng.uniform(-0.1, 0.1)), 1)
                elif isinstance(e.value, dict):
                    # Approximate numeric values in dicts
                    approx_value = {}
                    for k, v in e.value.items():
                        try:
                            num = float(v)
                            approx_value[k] = str(round(num * (1 + self._rng.uniform(-0.1, 0.1)), 1))
                        except (ValueError, TypeError):
                            approx_value[k] = v
                visible_elements.append(Element(
                    id=e.id,
                    type=e.type,
                    label=e.label,
                    value=approx_value,
                    editable=e.editable,
                    data_type=e.data_type,
                    constraints={},
                    actions=[],
                    metadata={},
                ))
        return Observation(
            meta=Meta(
                app_name=self.app_name,
                app_version=full_obs.meta.app_version,
                observation_source="screenshot",
            ),
            app_state=AppState(
                current_view=full_obs.app_state.current_view,
                active_document=full_obs.app_state.active_document,
            ),
            interactive_elements=visible_elements,
            environment=Environment(),  # No background state visible
            data_summary=f"[Screenshot] {len(visible_elements)}/{len(full_obs.interactive_elements)} elements visible",
        )

    def format_observation_as_a11y_tree(self, obs: Observation) -> str:
        """Format observation as OSWorld-style accessibility tree text.

        Mimics the linearized a11y tree format OSWorld agents receive:
            tag  name  text  value  position
        """
        lines = [
            f"Application: {obs.meta.app_name}",
            f"View: {obs.app_state.current_view}",
            f"Document: {obs.app_state.active_document}",
            "",
            "tag\tname\tvalue",
        ]
        for e in obs.interactive_elements:
            val_str = json.dumps(e.value) if isinstance(e.value, dict) else str(e.value)
            lines.append(f"{e.type}\t{e.label or e.id}\t{val_str}")
        return "\n".join(lines)

    def predict(self, instruction: str, obs: Observation) -> tuple[str, list[Action]]:
        """OSWorld-compatible agent interface.

        In a real GUI agent, this would send the screenshot to an LLM.
        In this simulation, it returns a text summary and an empty action list
        (actions are executed via execute() in the evaluation loop).

        Returns:
            response: Text description of what the agent "sees"
            actions: Empty list — actions are scheduled externally in simulation
        """
        a11y = self.format_observation_as_a11y_tree(obs)
        response = (
            f"[GUI Agent] Instruction: {instruction}\n"
            f"{a11y}\n"
            f"Visible elements: {len(obs.interactive_elements)}"
        )
        return response, []

    def save_trajectory(self, output_dir: Path) -> None:
        """Save per-step trajectory as OSWorld-compatible traj.jsonl."""
        output_dir.mkdir(parents=True, exist_ok=True)
        traj_path = output_dir / "traj.jsonl"
        with open(traj_path, "w") as f:
            for entry in self._trajectory:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def validate_action(self, action: Action) -> bool:
        return action.action_type in self.supported_action_types


# Backward-compatible alias for older tests/imports.
GUIAgentBaseline = SimulatedGUIBaseline
