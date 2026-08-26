"""ASIL evaluation metrics collection and computation.

Aligned with OSWorld's benchmark structure:
- Per-step trajectory recording (StepResult)
- Per-task result with trajectory (TaskResult)
- Aggregate metrics (compute_metrics)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    """Result of a single step within a task — mirrors OSWorld's traj.jsonl entry."""
    step_num: int
    action_type: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    latency_ms: float = 0.0
    observation_element_count: int = 0
    observation_source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # For GUI agent: grounding error details
    grounding_error: bool = False
    grounding_error_type: str = ""  # "click_miss" | "wrong_element" | "timeout"
    # Response text from agent (if agent-based execution)
    agent_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_num": self.step_num,
            "action_timestamp": self.timestamp,
            "action": {
                "action_type": self.action_type,
                "target": self.target,
                "params": self.params,
            },
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "observation_element_count": self.observation_element_count,
            "observation_source": self.observation_source,
            "grounding_error": self.grounding_error,
            "grounding_error_type": self.grounding_error_type,
            "agent_response": self.agent_response,
        }


@dataclass
class TaskResult:
    """Result of running a single task — includes full step trajectory."""
    task_id: str
    software: str = ""
    difficulty: str = ""
    instruction: str = ""
    success: bool = False
    steps: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    e2e_time_s: float = 0.0
    deadlocked: bool = False
    observation_element_count: int = 0
    total_element_count: int = 0
    score: float = 0.0  # 0.0 or 1.0 like OSWorld

    @property
    def coverage(self) -> float:
        if self.total_element_count == 0:
            return 0.0
        return self.observation_element_count / self.total_element_count

    @property
    def avg_latency_ms(self) -> float:
        if not self.step_results:
            return 0.0
        return sum(s.latency_ms for s in self.step_results) / len(self.step_results)

    @property
    def latencies_ms(self) -> list[float]:
        return [s.latency_ms for s in self.step_results]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "software": self.software,
            "difficulty": self.difficulty,
            "instruction": self.instruction,
            "success": self.success,
            "score": self.score,
            "steps": self.steps,
            "e2e_time_s": round(self.e2e_time_s, 3),
            "deadlocked": self.deadlocked,
            "coverage": round(self.coverage, 4),
            "trajectory": [s.to_dict() for s in self.step_results],
        }

    def save_trajectory(self, output_dir: Path) -> None:
        """Save trajectory as traj.jsonl (OSWorld-compatible format)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        traj_path = output_dir / "traj.jsonl"
        with open(traj_path, "w") as f:
            for step in self.step_results:
                f.write(json.dumps(step.to_dict(), ensure_ascii=False) + "\n")

    def save_result(self, output_dir: Path) -> None:
        """Save result.txt (OSWorld-compatible: single float score)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.txt").write_text(str(self.score))


def compute_metrics(results: list[TaskResult]) -> dict[str, Any]:
    """Compute aggregate metrics from a list of task results.

    Returns per-task breakdown + aggregates (OSWorld-style).
    """
    n = len(results)
    if n == 0:
        return {"tasks": [], "aggregate": {}}

    all_latencies = [s.latency_ms for r in results for s in r.step_results]

    # Per-software breakdown
    by_software: dict[str, list[TaskResult]] = {}
    for r in results:
        by_software.setdefault(r.software, []).append(r)

    per_software = {}
    for sw, sw_results in by_software.items():
        successes = sum(1 for r in sw_results if r.success)
        per_software[sw] = {
            "success_rate": successes / len(sw_results),
            "total": len(sw_results),
            "passed": successes,
        }

    successes = sum(1 for r in results if r.success)
    deadlocks = sum(1 for r in results if r.deadlocked)

    aggregate = {
        "success_rate": successes / n,
        "avg_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0.0,
        "avg_e2e_s": sum(r.e2e_time_s for r in results) / n,
        "deadlock_rate": deadlocks / n,
        "avg_coverage": sum(r.coverage for r in results) / n,
        "avg_steps": sum(r.steps for r in results) / n,
        "total_tasks": n,
        "passed": successes,
    }

    return {
        "tasks": [r.to_dict() for r in results],
        "per_software": per_software,
        "aggregate": aggregate,
    }
