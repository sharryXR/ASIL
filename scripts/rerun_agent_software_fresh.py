#!/usr/bin/env python3
"""Re-run selected agent-evaluated software in fresh eval containers.

This script is designed for targeted repairs inside an existing result directory.
It avoids long-lived eval-container contamination by launching a brand-new
`docker compose run --rm eval ...` process per software, then merges the new
software-specific results back into the target output directory.
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


def _delete_software_result_tree(result_root: Path, software: str) -> None:
    shutil.rmtree(result_root / software, ignore_errors=True)


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
    test_config_base_dir: str,
    test_all: str,
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
        test_config_base_dir,
        "--task-set",
        test_all,
        "--output-dir",
        str(container_output_dir),
        "--output",
        str(container_output_json),
    ]


def _merge_completed_software_run(
    *,
    target_output_dir: Path,
    target_output_json: Path,
    temp_output_dir: Path,
    temp_output_json: Path,
    software: str,
    model: str,
) -> None:
    target_result_root = _agent_result_root(target_output_dir, model)
    temp_result_root = _agent_result_root(temp_output_dir, model)
    target_summary_dir = _summary_dir(target_output_dir, model)
    temp_summary_dir = _summary_dir(temp_output_dir, model)

    src_software_dir = temp_result_root / software
    if not src_software_dir.exists():
        raise RuntimeError(f"Fresh rerun for `{software}` did not produce {src_software_dir}")

    _delete_software_result_tree(target_result_root, software)
    shutil.copytree(src_software_dir, target_result_root / software)

    existing_results = _load_json(target_summary_dir / "results.json", [])
    existing_results = [row for row in existing_results if row.get("software") != software]
    new_results = _load_json(temp_summary_dir / "results.json", [])
    new_results = [row for row in new_results if row.get("software") == software]
    _write_json(target_summary_dir / "results.json", existing_results + new_results)

    existing_aggregate = _load_json(target_summary_dir / "aggregate.json", {})
    new_aggregate = _load_json(temp_summary_dir / "aggregate.json", {})
    existing_aggregate.pop(software, None)
    if software in new_aggregate:
        existing_aggregate[software] = new_aggregate[software]
    _write_json(target_summary_dir / "aggregate.json", existing_aggregate)

    existing_output = _load_json(target_output_json, {})
    new_output = _load_json(temp_output_json, {})
    existing_output.pop(software, None)
    if software in new_output:
        existing_output[software] = new_output[software]
    _write_json(target_output_json, existing_output)


def _run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-container rerun for selected agent-evaluated software.")
    parser.add_argument("--software", nargs="+", required=True, help="Software domains to rerun.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing result directory root to update.")
    parser.add_argument("--output", type=Path, required=True, help="Flat JSON output to merge/update.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--test-config-base-dir", default="evaluation_examples")
    parser.add_argument("--test-all", default="test_expansion_all.json")
    parser.add_argument("--project-name", default="asil-expansion-gui-v2")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    compose_file = project_root / "docker" / "docker-compose.yml"
    host_results_root = project_root / "results"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="asil-rerun-", dir=host_results_root))
    try:
        for software in args.software:
            temp_output_dir = tmp_root / software
            temp_output_json = temp_output_dir / f"results_{software}.json"
            command = _build_compose_run_command(
                project_root=project_root,
                env_file=env_file,
                compose_file=compose_file,
                project_name=args.project_name,
                software=software,
                provider=args.provider,
                model=args.model,
                max_steps=args.max_steps,
                test_config_base_dir=args.test_config_base_dir,
                test_all=args.test_all,
                host_results_root=host_results_root,
                host_output_dir=temp_output_dir,
                host_output_json=temp_output_json,
            )
            print(f"\n=== Fresh rerun: {software} ===")
            _run_command(command, project_root)
            _merge_completed_software_run(
                target_output_dir=args.output_dir,
                target_output_json=args.output,
                temp_output_dir=temp_output_dir,
                temp_output_json=temp_output_json,
                software=software,
                model=args.model,
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nUpdated result directory: {args.output_dir}")
    print(f"Updated flat output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
