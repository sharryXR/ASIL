#!/usr/bin/env python3
"""Managed docker orchestrator for scripts/run_benchmark.py."""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asil.benchmark as _benchmark  # noqa: E402
from asil.result_store import TaskKey, group_task_keys, load_summary_entries, select_pending_tasks, write_aggregate  # noqa: E402


def _list_docker_names(kind: str) -> list[str]:
    command = ["docker", kind, "ls", "--format", "{{.Name}}"]
    if kind == "ps":
        command = ["docker", "ps", "-a", "--format", "{{.Names}}"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _remove_docker_resources(command: list[str], names: list[str]) -> None:
    if not names:
        return
    subprocess.run([*command, *names], check=False, capture_output=True)


def _cleanup_project_prefix(prefix: str) -> None:
    containers = [name for name in _list_docker_names("ps") if name.startswith(prefix)]
    networks = [name for name in _list_docker_names("network") if name.startswith(prefix)]
    volumes = [name for name in _list_docker_names("volume") if name.startswith(prefix)]
    _remove_docker_resources(["docker", "rm", "-f"], containers)
    _remove_docker_resources(["docker", "network", "rm"], networks)
    _remove_docker_resources(["docker", "volume", "rm"], volumes)


def _should_auto_cleanup_prefix(project_name: str) -> bool:
    return project_name.startswith("asil-jupyterlab-")


def _skip_prebuild() -> bool:
    return os.environ.get("ASIL_MANAGED_SKIP_BUILD", "").strip().lower() in {"1", "true", "yes"}


def _keep_generated_task_sets() -> bool:
    return os.environ.get("ASIL_MANAGED_KEEP_SHARDS", "").strip().lower() in {"1", "true", "yes"}


def _cleanup_generated_task_sets(shard_paths: list[Path]) -> int:
    cleaned = 0
    for shard_path in shard_paths:
        try:
            shard_path.unlink()
        except FileNotFoundError:
            continue
        cleaned += 1
    if cleaned:
        print(f"[managed] cleaned {cleaned} generated task-set shard(s)")
    return cleaned


def _map_result_path_for_container(path_arg: str, *, project_root: Path) -> str:
    """Rewrite project result paths onto the eval container's /results mount."""
    results_root = project_root / "results"
    path = Path(path_arg)

    if path.is_absolute():
        try:
            relative = path.relative_to(results_root)
        except ValueError:
            return path_arg
        return str(Path("/results") / relative)

    if path_arg == "results":
        return "/results"
    if path_arg.startswith("results/"):
        relative = Path(path_arg).relative_to("results")
        return str(Path("/results") / relative)

    return path_arg


def _rewrite_forwarded_result_paths(
    forwarded_args: list[str],
    *,
    project_root: Path,
) -> list[str]:
    rewritten: list[str] = []
    expects_path_for: str | None = None
    path_flags = {"--output", "--output-dir"}

    for arg in forwarded_args:
        if expects_path_for is not None:
            rewritten.append(_map_result_path_for_container(arg, project_root=project_root))
            expects_path_for = None
            continue

        matched_inline = False
        for flag in path_flags:
            prefix = f"{flag}="
            if arg.startswith(prefix):
                value = arg[len(prefix):]
                mapped = _map_result_path_for_container(value, project_root=project_root)
                rewritten.append(f"{flag}={mapped}")
                matched_inline = True
                break
        if matched_inline:
            continue

        rewritten.append(arg)
        if arg in path_flags:
            expects_path_for = arg

    return rewritten


def _build_eval_run_command(
    *,
    env_file: Path,
    compose_file: Path,
    project_name: str,
    forwarded_args: list[str],
) -> list[str]:
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
        *forwarded_args,
    ]


def _build_down_command(
    *,
    env_file: Path,
    compose_file: Path,
    project_name: str,
) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "-p",
        project_name,
        "down",
        "--remove-orphans",
    ]


def _build_compose_build_command(
    *,
    env_file: Path,
    compose_file: Path,
    project_name: str,
    services: list[str],
) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "-p",
        project_name,
        "build",
        *services,
    ]


def _build_worker_project_name(base_project_name: str, worker_index: int) -> str:
    return f"{base_project_name}-w{worker_index:02d}"


def _stable_round_robin_shards(tasks: list[TaskKey], num_envs: int) -> list[list[TaskKey]]:
    shards: list[list[TaskKey]] = [[] for _ in range(max(1, num_envs))]
    for index, task in enumerate(tasks):
        shards[index % len(shards)].append(task)
    return [shard for shard in shards if shard]


def _host_relative_to_project(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _strip_flag(
    args: list[str],
    flag_names: set[str],
    *,
    nargs_plus_flags: set[str] | None = None,
) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    skip_until_next_flag = False
    nargs_plus_flags = nargs_plus_flags or set()
    for arg in args:
        if skip_until_next_flag:
            if arg.startswith("-"):
                skip_until_next_flag = False
            else:
                continue
        if skip_next:
            skip_next = False
            continue
        matched = False
        for flag in flag_names:
            if arg == flag:
                if flag in nargs_plus_flags:
                    skip_until_next_flag = True
                else:
                    skip_next = True
                matched = True
                break
            if arg.startswith(f"{flag}="):
                matched = True
                break
        if not matched:
            stripped.append(arg)
    return stripped


def _replace_or_append_flag(args: list[str], flag: str, value: str) -> list[str]:
    updated = _strip_flag(args, {flag})
    updated.extend([flag, value])
    return updated


def _ensure_flag(args: list[str], flag: str) -> list[str]:
    if flag in args:
        return args
    return [*args, flag]


def _write_shard_task_set(
    *,
    evaluation_root: Path,
    base_name: str,
    worker_index: int,
    shard_tasks: list[TaskKey],
) -> Path:
    evaluation_root.mkdir(parents=True, exist_ok=True)
    shard_path = evaluation_root / f".generated-{base_name}.w{worker_index:02d}.json"
    shard_payload = group_task_keys(shard_tasks)
    shard_path.write_text(
        json.dumps(shard_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return shard_path


def _prepare_forwarded_args_for_worker(
    *,
    forwarded_args: list[str],
    shard_task_set: str,
    worker_output: str,
) -> list[str]:
    cleaned = _strip_flag(
        forwarded_args,
        {
            "--num-envs",
            "--software",
            "--software-filter",
            "--task-id-filter",
            "--managed-docker",
            "--output",
            "--task-set",
        },
        nargs_plus_flags={"--software", "--software-filter", "--task-id-filter"},
    )
    cleaned = _replace_or_append_flag(cleaned, "--task-set", shard_task_set)
    cleaned = _replace_or_append_flag(cleaned, "--output", worker_output)
    cleaned = _ensure_flag(cleaned, "--managed-docker")
    cleaned = _replace_or_append_flag(cleaned, "--num-envs", "1")
    return cleaned


def _materialize_benchmark_config(forwarded_args: list[str]):
    parser = _benchmark.build_arg_parser()
    namespace = parser.parse_args(forwarded_args)
    config = _benchmark.config_from_namespace(namespace)
    return namespace, config


def _dry_run_parallel_plan(
    *,
    task_mapping: dict[str, list[str]],
    pending_tasks: tuple[TaskKey, ...],
    skipped_tasks: int,
    num_envs: int,
) -> None:
    shards = _stable_round_robin_shards(list(pending_tasks), num_envs)
    print(f"Total tasks: {sum(len(task_ids) for task_ids in task_mapping.values())}")
    print(f"Skipped tasks: {skipped_tasks}")
    print(f"Unfinished tasks: {len(pending_tasks)}")
    for index, shard in enumerate(shards, start=1):
        print(f"  shard {index}: {len(shard)} tasks")


def _run_single_worker(
    *,
    env_file: Path,
    compose_file: Path,
    project_name: str,
    forwarded_args: list[str],
) -> int:
    if _should_auto_cleanup_prefix(project_name):
        _cleanup_project_prefix(project_name)

    run_command = _build_eval_run_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name=project_name,
        forwarded_args=forwarded_args,
    )
    down_command = _build_down_command(
        env_file=env_file,
        compose_file=compose_file,
        project_name=project_name,
    )

    exit_code = 0
    try:
        subprocess.run(run_command, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        exit_code = exc.returncode or 1
    finally:
        down_result = subprocess.run(down_command, cwd=str(PROJECT_ROOT), check=False)
        if down_result.returncode != 0 and exit_code == 0:
            exit_code = down_result.returncode
        if _should_auto_cleanup_prefix(project_name):
            _cleanup_project_prefix(project_name)
    return exit_code


def _rebuild_shared_outputs(config) -> None:
    args = _benchmark._namespace_from_config(config)
    args.software = list(config.software)
    args.software_filter = list(config.software)
    result_root = _benchmark._result_root_for_config(config)
    summary_dir = result_root / "summary"
    entries = load_summary_entries(summary_dir)
    write_aggregate(summary_dir, entries, is_comparison=config.run_mode == "comparison")
    _benchmark._rebuild_output_json(args, summary_dir, is_comparison=config.run_mode == "comparison")


def _restore_host_ownership(path: Path) -> None:
    """Make bind-mounted result files writable by the host user again."""
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return
    uid = os.getuid()
    gid = os.getgid()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{PROJECT_ROOT}:/workspace",
            "asil-eval:local",
            "chown",
            "-R",
            f"{uid}:{gid}",
            f"/workspace/{relative}",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_parallel_workers(
    *,
    env_file: Path,
    compose_file: Path,
    base_project_name: str,
    forwarded_args: list[str],
    config,
    task_mapping: dict[str, list[str]],
    pending_tasks: tuple[TaskKey, ...],
    keep_generated_task_sets: bool = False,
) -> int:
    evaluation_root = PROJECT_ROOT / "evaluation_examples"
    run_slug = f"{base_project_name}-{Path(config.task_index).stem}"
    shards = _stable_round_robin_shards(list(pending_tasks), config.num_envs)
    worker_specs: list[tuple[str, subprocess.Popen[bytes], list[str], Path]] = []
    shard_paths: list[Path] = []
    exit_code = 0
    workers_completed = False
    auto_cleanup = _should_auto_cleanup_prefix(base_project_name)

    if auto_cleanup:
        _cleanup_project_prefix(base_project_name)

    try:
        if _skip_prebuild():
            print("[managed] skipping prebuild because ASIL_MANAGED_SKIP_BUILD=1")
        else:
            prebuild_command = _build_compose_build_command(
                env_file=env_file,
                compose_file=compose_file,
                project_name=base_project_name,
                services=["obs-mock", "eval"],
            )
            subprocess.run(prebuild_command, cwd=str(PROJECT_ROOT), check=True)

        for worker_index, shard_tasks in enumerate(shards, start=1):
            project_name = _build_worker_project_name(base_project_name, worker_index)
            shard_path = _write_shard_task_set(
                evaluation_root=evaluation_root,
                base_name=run_slug,
                worker_index=worker_index,
                shard_tasks=shard_tasks,
            )
            shard_paths.append(shard_path)
            worker_output = f"results/.worker-results/{run_slug}.w{worker_index:02d}.json"
            worker_args = _prepare_forwarded_args_for_worker(
                forwarded_args=forwarded_args,
                shard_task_set=shard_path.name,
                worker_output=worker_output,
            )
            rewritten_args = _rewrite_forwarded_result_paths(
                worker_args,
                project_root=PROJECT_ROOT,
            )
            run_command = _build_eval_run_command(
                env_file=env_file,
                compose_file=compose_file,
                project_name=project_name,
                forwarded_args=rewritten_args,
            )
            process = subprocess.Popen(run_command, cwd=str(PROJECT_ROOT))
            worker_specs.append(
                (
                    project_name,
                    process,
                    _build_down_command(
                        env_file=env_file,
                        compose_file=compose_file,
                        project_name=project_name,
                    ),
                    shard_path,
                )
            )

        for project_name, process, _, _ in worker_specs:
            return_code = process.wait()
            if return_code != 0 and exit_code == 0:
                exit_code = return_code
            print(f"[managed] worker {project_name} finished with code {return_code}")
        workers_completed = True
    finally:
        for project_name, _, down_command, _ in worker_specs:
            down_result = subprocess.run(down_command, cwd=str(PROJECT_ROOT), check=False)
            if down_result.returncode != 0 and exit_code == 0:
                exit_code = down_result.returncode
        if auto_cleanup:
            _cleanup_project_prefix(base_project_name)
        _restore_host_ownership(_benchmark._result_root_for_config(config))
        _restore_host_ownership(PROJECT_ROOT / "results" / ".worker-results")
        _rebuild_shared_outputs(config)
        if workers_completed and exit_code == 0 and not keep_generated_task_sets:
            _cleanup_generated_task_sets(shard_paths)
        elif shard_paths and keep_generated_task_sets:
            print(
                "[managed] keeping generated task-set shards because "
                "ASIL_MANAGED_KEEP_SHARDS=1 or --keep-generated-task-sets was set"
            )
        elif shard_paths and exit_code != 0:
            print("[managed] keeping generated task-set shards because at least one worker failed")

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Managed docker wrapper/orchestrator for scripts/run_benchmark.py",
    )
    parser.add_argument("--compose-project-name", default="asil-managed")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--keep-generated-task-sets",
        action="store_true",
        default=_keep_generated_task_sets(),
        help="Keep managed worker .generated-* task-set shards after a successful parallel run.",
    )
    args, forwarded_args = parser.parse_known_args(argv)

    if not forwarded_args:
        parser.error(
            "Pass benchmark arguments, for example "
            "--task-set test_full15.json --participant asil --asil-execution agentic --provider openai"
        )

    benchmark_args, config = _materialize_benchmark_config(forwarded_args)
    num_envs = max(1, int(args.num_envs if args.num_envs is not None else config.num_envs))
    config = _benchmark.BenchmarkConfig(
        software=config.software,
        task_index=config.task_index,
        output_dir=config.output_dir,
        output_json=config.output_json,
        participant=config.participant,
        run_mode=config.run_mode,
        comparison_participants=config.comparison_participants,
        asil_execution=config.asil_execution,
        asil_success_hint=config.asil_success_hint,
        independent_raw_validation=config.independent_raw_validation,
        execution_mode=config.execution_mode,
        provider=config.provider,
        model=config.model,
        max_steps=config.max_steps,
        test_config_base_dir=config.test_config_base_dir,
        osworld_format=config.osworld_format,
        docker=config.docker,
        managed_docker=True,
        mock=config.mock,
        dry_run=config.dry_run,
        resume=config.resume,
        force_rerun=config.force_rerun,
        rerun_failed_only=config.rerun_failed_only,
        num_envs=num_envs,
    )

    task_mapping = _benchmark._load_task_index_mapping(
        task_index=benchmark_args.task_set,
        test_config_base_dir=benchmark_args.test_config_base_dir,
        software_filter=tuple(getattr(benchmark_args, "software_filter", ()) or ()),
        task_id_filter=tuple(getattr(benchmark_args, "task_id_filter", ()) or ()),
    )
    result_root = _benchmark._result_root_for_config(config)
    result_root.mkdir(parents=True, exist_ok=True)
    selection = select_pending_tasks(
        task_mapping,
        result_root=result_root,
        run_mode=config.run_mode,
        participant=config.participant,
        comparison_participants=config.comparison_participants,
        resume=config.resume,
        force_rerun=config.force_rerun,
        rerun_failed_only=config.rerun_failed_only,
    )

    if config.dry_run:
        _dry_run_parallel_plan(
            task_mapping=task_mapping,
            pending_tasks=selection.pending_tasks,
            skipped_tasks=selection.skipped_tasks,
            num_envs=num_envs,
        )
        return 0

    env_file = PROJECT_ROOT / ".env"
    compose_file = PROJECT_ROOT / "docker" / "docker-compose.yml"

    if num_envs == 1:
        rewritten_args = _rewrite_forwarded_result_paths(
            _ensure_flag(_replace_or_append_flag(forwarded_args, "--num-envs", "1"), "--managed-docker"),
            project_root=PROJECT_ROOT,
        )
        return _run_single_worker(
            env_file=env_file,
            compose_file=compose_file,
            project_name=args.compose_project_name,
            forwarded_args=rewritten_args,
        )

    if not selection.pending_tasks:
        _restore_host_ownership(_benchmark._result_root_for_config(config))
        _restore_host_ownership(PROJECT_ROOT / "results" / ".worker-results")
        _rebuild_shared_outputs(config)
        print(
            f"All selected tasks already complete under {result_root} "
            f"(skipped={selection.skipped_tasks}/{selection.total_tasks})."
        )
        return 0

    return _run_parallel_workers(
        env_file=env_file,
        compose_file=compose_file,
        base_project_name=args.compose_project_name,
        forwarded_args=forwarded_args,
        config=config,
        task_mapping=task_mapping,
        pending_tasks=selection.pending_tasks,
        keep_generated_task_sets=args.keep_generated_task_sets,
    )


if __name__ == "__main__":
    raise SystemExit(main())
