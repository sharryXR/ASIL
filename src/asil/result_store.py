from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl


@dataclass(frozen=True)
class TaskKey:
    software: str
    task_id: str


@dataclass(frozen=True)
class TaskSelection:
    total_tasks: int
    skipped_tasks: int
    cleaned_incomplete_tasks: int
    pending_tasks: tuple[TaskKey, ...]


def flatten_task_mapping(task_mapping: dict[str, list[str]]) -> list[TaskKey]:
    flattened: list[TaskKey] = []
    for software, task_ids in task_mapping.items():
        for task_id in task_ids:
            flattened.append(TaskKey(software=software, task_id=task_id))
    return flattened


def group_task_keys(task_keys: list[TaskKey]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for task in task_keys:
        grouped.setdefault(task.software, []).append(task.task_id)
    return grouped


def task_dir_for_run(
    result_root: Path,
    task: TaskKey,
    *,
    run_mode: str,
    participant: str,
    comparison_participants: tuple[str, ...],
) -> Path | dict[str, Path]:
    if run_mode == "comparison":
        return {
            participant_name: result_root / participant_name / task.software / task.task_id
            for participant_name in comparison_participants
        }
    return result_root / task.software / task.task_id


def _read_result_score(task_dir: Path) -> float | None:
    result_path = task_dir / "result.txt"
    if not result_path.exists():
        return None
    try:
        return float(result_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _clear_task_dir(task_dir: Path) -> None:
    if task_dir.exists():
        shutil.rmtree(task_dir)


def _status_for_single_task_dir(
    task_dir: Path,
    *,
    force_rerun: bool,
    rerun_failed_only: bool,
) -> tuple[str, bool]:
    if force_rerun:
        return "pending", False

    result_path = task_dir / "result.txt"
    if result_path.exists():
        score = _read_result_score(task_dir)
        has_runtime_error = (task_dir / "runtime_error.txt").exists()
        if rerun_failed_only:
            if has_runtime_error or score is None or score < 1.0:
                return "pending", False
            return "skip", False
        return "skip", False

    if task_dir.exists():
        return "incomplete", True
    return "pending", False


def select_pending_tasks(
    task_mapping: dict[str, list[str]],
    *,
    result_root: Path,
    run_mode: str,
    participant: str,
    comparison_participants: tuple[str, ...],
    resume: bool,
    force_rerun: bool,
    rerun_failed_only: bool,
) -> TaskSelection:
    pending_tasks: list[TaskKey] = []
    skipped_tasks = 0
    cleaned_incomplete_tasks = 0
    total_tasks = 0

    for task in flatten_task_mapping(task_mapping):
        total_tasks += 1
        if not resume and not force_rerun and not rerun_failed_only:
            pending_tasks.append(task)
            continue

        task_target = task_dir_for_run(
            result_root,
            task,
            run_mode=run_mode,
            participant=participant,
            comparison_participants=comparison_participants,
        )

        if run_mode == "comparison":
            assert isinstance(task_target, dict)
            participant_statuses: list[str] = []
            should_clean = False
            for participant_name in comparison_participants:
                status, cleanable = _status_for_single_task_dir(
                    task_target[participant_name],
                    force_rerun=force_rerun,
                    rerun_failed_only=rerun_failed_only,
                )
                participant_statuses.append(status)
                should_clean = should_clean or cleanable

            if all(status == "skip" for status in participant_statuses):
                skipped_tasks += 1
                continue

            if should_clean:
                for task_dir in task_target.values():
                    _clear_task_dir(task_dir)
                cleaned_incomplete_tasks += 1

            pending_tasks.append(task)
            continue

        assert isinstance(task_target, Path)
        status, cleanable = _status_for_single_task_dir(
            task_target,
            force_rerun=force_rerun,
            rerun_failed_only=rerun_failed_only,
        )
        if status == "skip":
            skipped_tasks += 1
            continue
        if cleanable:
            _clear_task_dir(task_target)
            cleaned_incomplete_tasks += 1
        pending_tasks.append(task)

    return TaskSelection(
        total_tasks=total_tasks,
        skipped_tasks=skipped_tasks,
        cleaned_incomplete_tasks=cleaned_incomplete_tasks,
        pending_tasks=tuple(pending_tasks),
    )


def append_summary_entry(summary_dir: Path, entry: dict[str, Any]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    results_path = summary_dir / "results.json"
    with open(results_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            content = handle.read().strip()
            if content:
                try:
                    payload = json.loads(content)
                    if not isinstance(payload, list):
                        payload = []
                except json.JSONDecodeError:
                    payload = []
            else:
                payload = []
            payload.append(entry)
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_summary_entries(summary_dir: Path) -> list[dict[str, Any]]:
    results_path = summary_dir / "results.json"
    if not results_path.exists():
        return []
    content = results_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError(f"{results_path} must contain a JSON array.")
    return payload


def build_aggregate(entries: list[dict[str, Any]], *, is_comparison: bool) -> dict[str, Any]:
    if is_comparison:
        aggregate: dict[str, dict[str, dict[str, Any]]] = {}
        for entry in entries:
            participant_name = str(entry["participant"])
            software = str(entry["software"])
            bucket = aggregate.setdefault(participant_name, {}).setdefault(
                software,
                {
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_e2e_s": 0.0,
                    "deadlock_rate": 0.0,
                    "avg_steps": 0.0,
                    "total_tasks": 0,
                    "passed": 0,
                    "_latency_total": 0.0,
                    "_e2e_total": 0.0,
                    "_steps_total": 0.0,
                    "_deadlocks": 0,
                },
            )
            bucket["total_tasks"] += 1
            bucket["passed"] += 1 if entry.get("success", False) else 0
            bucket["_latency_total"] += float(entry.get("avg_latency_ms", 0.0) or 0.0)
            bucket["_e2e_total"] += float(entry.get("e2e_time_s", 0.0) or 0.0)
            bucket["_steps_total"] += float(entry.get("steps", 0.0) or 0.0)
            bucket["_deadlocks"] += 1 if entry.get("deadlocked", False) else 0
        for participant_payload in aggregate.values():
            for bucket in participant_payload.values():
                total = bucket["total_tasks"] or 1
                bucket["success_rate"] = bucket["passed"] / total
                bucket["avg_latency_ms"] = bucket["_latency_total"] / total
                bucket["avg_e2e_s"] = bucket["_e2e_total"] / total
                bucket["deadlock_rate"] = bucket["_deadlocks"] / total
                bucket["avg_steps"] = bucket["_steps_total"] / total
                bucket.pop("_latency_total", None)
                bucket.pop("_e2e_total", None)
                bucket.pop("_steps_total", None)
                bucket.pop("_deadlocks", None)
        return aggregate

    aggregate: dict[str, dict[str, Any]] = {}
    for entry in entries:
        software = str(entry["software"])
        bucket = aggregate.setdefault(
            software,
            {
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "avg_e2e_s": 0.0,
                "deadlock_rate": 0.0,
                "avg_steps": 0.0,
                "total_tasks": 0,
                "passed": 0,
                "_latency_total": 0.0,
                "_e2e_total": 0.0,
                "_steps_total": 0.0,
                "_deadlocks": 0,
            },
        )
        bucket["total_tasks"] += 1
        bucket["passed"] += 1 if entry.get("success", False) else 0
        bucket["_latency_total"] += float(entry.get("avg_latency_ms", 0.0) or 0.0)
        bucket["_e2e_total"] += float(entry.get("e2e_time_s", 0.0) or 0.0)
        bucket["_steps_total"] += float(entry.get("steps", 0.0) or 0.0)
        bucket["_deadlocks"] += 1 if entry.get("deadlocked", False) else 0

    for bucket in aggregate.values():
        total = bucket["total_tasks"] or 1
        bucket["success_rate"] = bucket["passed"] / total
        bucket["avg_latency_ms"] = bucket["_latency_total"] / total
        bucket["avg_e2e_s"] = bucket["_e2e_total"] / total
        bucket["deadlock_rate"] = bucket["_deadlocks"] / total
        bucket["avg_steps"] = bucket["_steps_total"] / total
        bucket.pop("_latency_total", None)
        bucket.pop("_e2e_total", None)
        bucket.pop("_steps_total", None)
        bucket.pop("_deadlocks", None)
    return aggregate


def write_aggregate(summary_dir: Path, entries: list[dict[str, Any]], *, is_comparison: bool) -> dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    aggregate = build_aggregate(entries, is_comparison=is_comparison)
    (summary_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate
