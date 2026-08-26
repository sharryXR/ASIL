#!/usr/bin/env python3
"""Re-run selected agent-evaluated tasks in fresh eval containers.

This is a task-level counterpart to rerun_agent_software_fresh.py. It is meant
for transient failures such as network hiccups where only a few tasks in an
otherwise healthy software run need to be replaced.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _agent_result_root(output_dir: Path, model: str) -> Path:
    return output_dir / "asil_protocol" / "structured_json" / model


def _summary_dir(output_dir: Path, model: str) -> Path:
    return _agent_result_root(output_dir, model) / "summary"


def _container_results_path(host_results_root: Path, host_path: Path) -> Path:
    return Path("/results") / host_path.resolve().relative_to(host_results_root.resolve())


def _compute_aggregate_from_task_dicts(software: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(tasks)
    if total == 0:
        return {
            "success_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_e2e_s": 0.0,
            "deadlock_rate": 0.0,
            "avg_coverage": 0.0,
            "avg_steps": 0.0,
            "total_tasks": 0,
            "passed": 0,
            "per_software": {software: {"success_rate": 0.0, "total": 0, "passed": 0}},
        }

    passed = sum(1 for task in tasks if task.get("success"))
    deadlocks = sum(1 for task in tasks if task.get("deadlocked"))
    avg_e2e = sum(float(task.get("e2e_time_s", 0.0)) for task in tasks) / total
    avg_coverage = sum(float(task.get("coverage", 0.0)) for task in tasks) / total
    avg_steps = sum(float(task.get("steps", 0.0)) for task in tasks) / total

    latencies: list[float] = []
    for task in tasks:
        for step in task.get("trajectory", []):
            if "latency_ms" in step:
                latencies.append(float(step["latency_ms"]))
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    success_rate = passed / total
    return {
        "success_rate": success_rate,
        "avg_latency_ms": avg_latency,
        "avg_e2e_s": avg_e2e,
        "deadlock_rate": deadlocks / total,
        "avg_coverage": avg_coverage,
        "avg_steps": avg_steps,
        "total_tasks": total,
        "passed": passed,
        "per_software": {
            software: {
                "success_rate": success_rate,
                "total": total,
                "passed": passed,
            }
        },
    }


def _build_compose_run_command(
    *,
    project_root: Path,
    env_file: Path,
    compose_file: Path,
    project_name: str,
    software: str,
    provider: str,
    model: str,
    max_steps: int,
    temp_index_name: str,
    host_results_root: Path,
    host_output_dir: Path,
    host_output_json: Path,
) -> list[str]:
    container_output_dir = _container_results_path(host_results_root, host_output_dir)
    container_output_json = _container_results_path(host_results_root, host_output_json)
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "-p",
        project_name,
        "--profile",
        "eval",
        "run",
        "--rm",
        "eval",
        "python",
        "scripts/run_benchmark.py",
        "--agent",
        "--provider",
        provider,
        "--model",
        model,
        "--software",
        software,
        "--max-steps",
        str(max_steps),
        "--test-config-base-dir",
        "evaluation_examples",
        "--task-set",
        temp_index_name,
        "--output-dir",
        str(container_output_dir),
        "--output",
        str(container_output_json),
    ]


def _merge_completed_task_rerun(
    *,
    target_output_dir: Path,
    target_output_json: Path,
    temp_output_dir: Path,
    temp_output_json: Path,
    software: str,
    task_ids: list[str],
    model: str,
) -> None:
    task_id_set = set(task_ids)
    target_result_root = _agent_result_root(target_output_dir, model)
    temp_result_root = _agent_result_root(temp_output_dir, model)
    target_summary_dir = _summary_dir(target_output_dir, model)
    temp_summary_dir = _summary_dir(temp_output_dir, model)

    target_software_dir = target_result_root / software
    temp_software_dir = temp_result_root / software
    if not temp_software_dir.exists():
        raise RuntimeError(f"Fresh rerun for `{software}` did not produce {temp_software_dir}")

    target_software_dir.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        src_task_dir = temp_software_dir / task_id
        if not src_task_dir.exists():
            raise RuntimeError(f"Fresh rerun for `{task_id}` did not produce {src_task_dir}")
        shutil.rmtree(target_software_dir / task_id, ignore_errors=True)
        shutil.copytree(src_task_dir, target_software_dir / task_id)

    existing_results = _load_json(target_summary_dir / "results.json", [])
    new_results = _load_json(temp_summary_dir / "results.json", [])
    kept_results = [
        row for row in existing_results
        if not (row.get("software") == software and row.get("task_id") in task_id_set)
    ]
    replaced_results = [
        row for row in new_results
        if row.get("software") == software and row.get("task_id") in task_id_set
    ]
    merged_results = kept_results + replaced_results
    _write_json(target_summary_dir / "results.json", merged_results)

    existing_output = _load_json(target_output_json, {})
    new_output = _load_json(temp_output_json, {})
    existing_software = existing_output.get(software, {})
    existing_tasks = existing_software.get("tasks", [])
    new_tasks = new_output.get(software, {}).get("tasks", [])

    kept_tasks = [task for task in existing_tasks if task.get("task_id") not in task_id_set]
    replaced_tasks = [task for task in new_tasks if task.get("task_id") in task_id_set]
    merged_tasks = sorted(kept_tasks + replaced_tasks, key=lambda task: task.get("task_id", ""))
    aggregate = _compute_aggregate_from_task_dicts(software, merged_tasks)

    existing_output[software] = {
        "aggregate": {k: v for k, v in aggregate.items() if k != "per_software"},
        "per_software": aggregate["per_software"],
        "tasks": merged_tasks,
    }
    _write_json(target_output_json, existing_output)

    existing_aggregate = _load_json(target_summary_dir / "aggregate.json", {})
    existing_aggregate[software] = aggregate
    _write_json(target_summary_dir / "aggregate.json", existing_aggregate)


def _run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-container rerun for selected agent-evaluated tasks.")
    parser.add_argument("--software", required=True, help="Software domain to rerun.")
    parser.add_argument("--task-id", nargs="+", required=True, help="Task IDs to rerun.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing result directory root to update.")
    parser.add_argument("--output", type=Path, required=True, help="Flat JSON output to merge/update.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--project-name", default="asil-expansion-gui-v2")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"
    host_results_root = project_root / "results"
    evaluation_examples_dir = project_root / "evaluation_examples"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="asil-rerun-task-", dir=host_results_root))
    temp_index_path = evaluation_examples_dir / f".tmp_{args.software}_{next(tempfile._get_candidate_names())}.json"
    try:
        temp_index_payload = {args.software: args.task_id}
        temp_index_path.write_text(
            json.dumps(temp_index_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        temp_output_dir = tmp_root / args.software
        temp_output_json = temp_output_dir / f"results_{args.software}.json"
        command = _build_compose_run_command(
            project_root=project_root,
            env_file=env_file,
            compose_file=compose_file,
            project_name=args.project_name,
            software=args.software,
            provider=args.provider,
            model=args.model,
            max_steps=args.max_steps,
            temp_index_name=temp_index_path.name,
            host_results_root=host_results_root,
            host_output_dir=temp_output_dir,
            host_output_json=temp_output_json,
        )
        print(f"\n=== Fresh task rerun: {args.software} {', '.join(args.task_id)} ===")
        _run_command(command, project_root)
        _merge_completed_task_rerun(
            target_output_dir=args.output_dir,
            target_output_json=args.output,
            temp_output_dir=temp_output_dir,
            temp_output_json=temp_output_json,
            software=args.software,
            task_ids=args.task_id,
            model=args.model,
        )
    finally:
        if temp_index_path.exists():
            temp_index_path.unlink()
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nUpdated result directory: {args.output_dir}")
    print(f"Updated flat output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
