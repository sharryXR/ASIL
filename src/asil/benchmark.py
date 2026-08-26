"""Public benchmark entrypoint for ASIL.

Supports:
  - Per-software evaluation (deterministic, uses ground-truth actions)
  - Agent-based evaluation (LLM observe-think-act loop) with per-step artifacts
  - Three-way comparison (ASIL vs CLI vs GUI Agent)
  - OSWorld-style result directory output with visual rendering
  - Docker sandbox mode
"""

import argparse
import importlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from asil.sandbox import ASILSandbox
from asil.rendering import RenderArtifact
from asil.result_store import (
    TaskKey,
    append_summary_entry,
    build_aggregate,
    group_task_keys,
    load_summary_entries,
    select_pending_tasks,
    task_dir_for_run,
    write_aggregate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


_SVG_DEFAULT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"'
    ' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">'
    '<g id="layer1" inkscape:groupmode="layer" inkscape:label="Layer 1">'
    '<rect id="rect1" x="10" y="20" width="100" height="50" style="fill:#ff0000"/>'
    '<circle id="circle1" cx="200" cy="150" r="40" style="fill:#00ff00"/>'
    '<rect id="rect2" x="300" y="100" width="80" height="60" style="fill:#0000ff"/>'
    '<text id="text1" x="50" y="300" style="font-size:24px">Hello</text>'
    '</g></svg>'
)

_SVG_BLANK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"'
    ' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">'
    '<g id="layer1" inkscape:groupmode="layer" inkscape:label="Layer 1"/>'
    '</svg>'
)

SOFTWARE_CHOICES = [
    "inkscape",
    "libreoffice",
    "blender",
    "obs",
    "gitea",
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
    "multi_apps",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    software: tuple[str, ...] = ()
    task_id_filter: tuple[str, ...] = ()
    task_index: str = "test_full15.json"
    output_dir: Path = Path("results")
    output_json: Path = Path("results.json")
    participant: Literal["asil", "cli", "gui"] = "asil"
    run_mode: Literal["single", "comparison"] = "single"
    comparison_participants: tuple[str, ...] = ("asil", "cli", "gui")
    asil_execution: Literal["deterministic", "agentic"] = "deterministic"
    asil_success_hint: Literal["evaluator", "none"] = "evaluator"
    independent_raw_validation: bool = False
    execution_mode: str | None = None
    provider: str = "mock"
    model: str = ""
    max_steps: int = 20
    test_config_base_dir: Path | None = None
    osworld_format: bool = False
    docker: bool = False
    managed_docker: bool = False
    mock: bool = False
    dry_run: bool = False
    resume: bool = True
    force_rerun: bool = False
    rerun_failed_only: bool = False
    num_envs: int = 1


@dataclass(frozen=True)
class BenchmarkRunResult:
    exit_code: int
    config: BenchmarkConfig
    result_root: Path
    output_json: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASIL Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deterministic evaluation (ground-truth actions)
  python scripts/run_benchmark.py --output results.json

  # Canonical 15-software run
  python scripts/run_benchmark.py --task-set test_full15.json --output-dir results/full_run

  # ASIL agent with OpenAI (reads OPENAI_API_KEY from .env)
  python scripts/run_benchmark.py --task-set test_full15.json --participant asil --asil-execution agentic --provider openai --model gpt-5.4

  # Real GUI agent with screenshot observation + pyautogui-style actions
  python scripts/run_benchmark.py --task-set test_full15.json --participant gui --provider openai --model gpt-5.4

  # Managed docker wrapper still works and forwards into the same entrypoint
  python scripts/run_evaluation_managed.py --task-set test_full15.json --participant asil --asil-execution agentic --provider openai --model gpt-5.4
        """,
    )

    parser.add_argument(
        "--software-filter",
        nargs="+",
        default=None,
        choices=SOFTWARE_CHOICES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--software",
        nargs="+",
        default=None,
        choices=SOFTWARE_CHOICES,
        dest="software_filter",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--task-id-filter",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )

    legacy_group = parser.add_mutually_exclusive_group()
    legacy_group.add_argument(
        "--agent",
        action="store_true",
        help="Compatibility alias for --participant asil --run-mode single --asil-execution agentic.",
    )
    legacy_group.add_argument(
        "--comparison",
        action="store_true",
        help="Compatibility alias for --run-mode comparison.",
    )
    parser.add_argument(
        "--participant",
        choices=["asil", "cli", "gui"],
        default=None,
        help="Benchmark participant to run in single mode (default: asil).",
    )
    parser.add_argument(
        "--run-mode",
        choices=["single", "comparison"],
        default=None,
        help="Benchmark organization mode. `single` runs one participant; `comparison` runs ASIL vs CLI vs GUI.",
    )
    parser.add_argument(
        "--asil-execution",
        choices=["deterministic", "agentic"],
        default=None,
        help="ASIL execution submode. Applies to the ASIL branch in single/comparison runs.",
    )
    parser.add_argument(
        "--asil-success-hint",
        choices=["evaluator", "none"],
        default="evaluator",
        help="Evaluator-derived success criteria for agentic ASIL prompts (default: evaluator).",
    )
    parser.add_argument(
        "--independent-raw-validation",
        action="store_true",
        help="Write a separate final-state score from raw files or backing services.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["deterministic", "agent", "comparison"],
        default=None,
        help="Deprecated compatibility flag. Use --participant/--run-mode/--asil-execution instead.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "openai", "anthropic"],
        help="LLM provider for agent mode (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name (e.g. gpt-5.4, claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per task in agent mode (default: 20)",
    )

    parser.add_argument(
        "--test-config-base-dir",
        type=Path,
        default=None,
        help="Base directory for OSWorld-style task configs (default: {project_root}/evaluation_examples)",
    )
    parser.add_argument(
        "--task-set",
        dest="task_set",
        type=str,
        default="test_full15.json",
        help="Task-set index file name within --test-config-base-dir (default: test_full15.json)",
    )
    parser.add_argument(
        "--test-all",
        dest="task_set",
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--osworld-format",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. Result layout is OSWorld-style by default.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run in Docker sandbox",
    )
    parser.add_argument(
        "--managed-docker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock adapters for Blender/OBS where supported",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks with existing result.txt and resume unfinished runs (default: on).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing result.txt files and rerun every selected task.",
    )
    parser.add_argument(
        "--rerun-failed-only",
        action="store_true",
        help="Only rerun tasks whose result.txt is < 1.0 or that have runtime_error.txt.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel managed docker environments to use (default: 1).",
    )
    return parser


def _resolve_benchmark_axes(
    *,
    participant: str | None,
    run_mode: str | None,
    asil_execution: str | None,
    execution_mode: str | None,
    agent: bool,
    comparison: bool,
) -> tuple[str, str, str]:
    explicit_participant = participant
    explicit_run_mode = run_mode
    explicit_asil_execution = asil_execution

    normalized_participant = participant or "asil"
    normalized_run_mode = run_mode or "single"
    normalized_asil_execution = asil_execution or "deterministic"

    implied_execution_mode = "agent" if agent else ("comparison" if comparison else None)
    if execution_mode and implied_execution_mode and execution_mode != implied_execution_mode:
        raise ValueError(
            f"Conflicting execution mode flags: --execution-mode={execution_mode} and "
            f"{'--agent' if agent else '--comparison'}."
        )

    compatibility_values: list[tuple[str, str, str]] = []
    if execution_mode == "deterministic":
        compatibility_values.append(("asil", "single", "deterministic"))
    elif execution_mode == "agent":
        compatibility_values.append(("asil", "single", "agentic"))
    elif execution_mode == "comparison":
        compatibility_values.append(("asil", "comparison", "deterministic"))

    if agent:
        compatibility_values.append(("asil", "single", "agentic"))
    if comparison:
        compatibility_values.append(("asil", "comparison", "deterministic"))

    for compat_participant, compat_run_mode, compat_asil_execution in compatibility_values:
        if explicit_participant is not None and explicit_participant != compat_participant:
            raise ValueError(
                f"Conflicting participant flags: --participant={explicit_participant} is incompatible with "
                f"legacy mode selecting participant={compat_participant}."
            )
        if explicit_run_mode is not None and explicit_run_mode != compat_run_mode:
            raise ValueError(
                f"Conflicting run-mode flags: --run-mode={explicit_run_mode} is incompatible with "
                f"legacy mode selecting run_mode={compat_run_mode}."
            )
        if explicit_asil_execution is not None and explicit_asil_execution != compat_asil_execution:
            raise ValueError(
                f"Conflicting ASIL execution flags: --asil-execution={explicit_asil_execution} is incompatible with "
                f"legacy mode selecting asil_execution={compat_asil_execution}."
            )
        normalized_participant = compat_participant
        normalized_run_mode = compat_run_mode
        normalized_asil_execution = compat_asil_execution

    if normalized_run_mode == "comparison":
        if explicit_participant is not None and explicit_participant != "asil":
            raise ValueError("--participant only applies to --run-mode single; comparison always anchors on ASIL vs CLI vs GUI.")
        normalized_participant = "asil"
        normalized_asil_execution = explicit_asil_execution or normalized_asil_execution or "deterministic"

    if normalized_participant in {"cli", "gui"}:
        if explicit_asil_execution is not None and explicit_asil_execution != "deterministic":
            raise ValueError("--asil-execution only applies to --participant asil.")
        normalized_asil_execution = "deterministic"

    return normalized_participant, normalized_run_mode, normalized_asil_execution


def config_from_namespace(args: argparse.Namespace) -> BenchmarkConfig:
    participant, run_mode, asil_execution = _resolve_benchmark_axes(
        participant=getattr(args, "participant", None),
        run_mode=getattr(args, "run_mode", None),
        asil_execution=getattr(args, "asil_execution", None),
        execution_mode=getattr(args, "execution_mode", None),
        agent=bool(getattr(args, "agent", False)),
        comparison=bool(getattr(args, "comparison", False)),
    )
    resolved_software = _resolve_software_selection(
        task_index=args.task_set,
        test_config_base_dir=args.test_config_base_dir,
        software_filter=tuple(getattr(args, "software_filter", ()) or ()),
    )
    return BenchmarkConfig(
        software=resolved_software,
        task_id_filter=tuple(getattr(args, "task_id_filter", ()) or ()),
        task_index=args.task_set,
        output_dir=args.output_dir,
        output_json=args.output,
        participant=participant,
        run_mode=run_mode,
        comparison_participants=("asil", "cli", "gui"),
        asil_execution=asil_execution,
        asil_success_hint=getattr(args, "asil_success_hint", "evaluator"),
        independent_raw_validation=bool(
            getattr(args, "independent_raw_validation", False)
        ),
        execution_mode=getattr(args, "execution_mode", None),
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        test_config_base_dir=args.test_config_base_dir,
        osworld_format=bool(args.osworld_format),
        docker=bool(args.docker),
        managed_docker=bool(getattr(args, "managed_docker", False)),
        mock=bool(args.mock),
        dry_run=bool(args.dry_run),
        resume=bool(getattr(args, "resume", True)),
        force_rerun=bool(getattr(args, "force_rerun", False)),
        rerun_failed_only=bool(getattr(args, "rerun_failed_only", False)),
        num_envs=max(1, int(getattr(args, "num_envs", 1) or 1)),
    )


def _namespace_from_config(config: BenchmarkConfig) -> argparse.Namespace:
    resolved_software = config.software or _resolve_software_selection(
        task_index=config.task_index,
        test_config_base_dir=config.test_config_base_dir,
        software_filter=(),
    )
    agent = (
        config.participant == "asil"
        and config.run_mode == "single"
        and config.asil_execution == "agentic"
    )
    comparison = config.run_mode == "comparison"
    return argparse.Namespace(
        software=list(resolved_software),
        software_filter=list(resolved_software),
        task_id_filter=list(config.task_id_filter),
        output=config.output_json,
        output_dir=config.output_dir,
        agent=agent,
        comparison=comparison,
        participant=config.participant,
        run_mode=config.run_mode,
        comparison_participants=list(config.comparison_participants),
        asil_execution=config.asil_execution,
        asil_success_hint=config.asil_success_hint,
        independent_raw_validation=config.independent_raw_validation,
        execution_mode=config.execution_mode,
        provider=config.provider,
        model=config.model,
        max_steps=config.max_steps,
        test_config_base_dir=config.test_config_base_dir,
        task_set=config.task_index,
        test_all=config.task_index,
        osworld_format=config.osworld_format,
        docker=config.docker,
        managed_docker=config.managed_docker,
        mock=config.mock,
        dry_run=config.dry_run,
        resume=config.resume,
        force_rerun=config.force_rerun,
        rerun_failed_only=config.rerun_failed_only,
        num_envs=config.num_envs,
    )


def _result_root_for_config(config: BenchmarkConfig) -> Path:
    if config.run_mode == "comparison":
        label = "deterministic"
        if config.asil_execution == "agentic":
            model_label = config.model or config.provider or "default"
            label = f"agentic__{model_label}"
        return config.output_dir / "comparison" / label
    if config.participant == "gui" and config.run_mode == "single":
        model_name = config.model or config.provider or "default"
        return config.output_dir / "pyautogui" / "screenshot" / model_name
    if config.participant == "cli" and config.run_mode == "single":
        return config.output_dir / "shell" / "terminal" / "cli-baseline"
    if (
        config.participant == "asil"
        and config.run_mode == "single"
        and config.asil_execution == "agentic"
    ):
        model_name = config.model or "default"
        return config.output_dir / "semantic" / "structured" / f"asil-{model_name}"
    return config.output_dir / "semantic" / "structured" / "asil-deterministic"


def _task_index_path(task_index: str, test_config_base_dir: Path | None) -> Path:
    config_base = test_config_base_dir or (PROJECT_ROOT / "evaluation_examples")
    return config_base / task_index


def _load_task_index_mapping(
    *,
    task_index: str,
    test_config_base_dir: Path | None,
    software_filter: tuple[str, ...] = (),
    task_id_filter: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    index_file = _task_index_path(task_index, test_config_base_dir)
    if not index_file.exists():
        raise ValueError(f"Task-set index not found: {index_file}")

    payload = json.loads(index_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"Task-set index {index_file} must be a non-empty object mapping software names to task ids."
        )

    mapping: dict[str, list[str]] = {}
    allowed_task_ids = set(task_id_filter)

    for software, task_ids in payload.items():
        if software_filter and software not in software_filter:
            continue
        if not isinstance(task_ids, list):
            raise ValueError(f"Task-set entry for `{software}` must be a list of task ids.")
        filtered_task_ids = [
            str(task_id)
            for task_id in task_ids
            if not allowed_task_ids or str(task_id) in allowed_task_ids
        ]
        if filtered_task_ids:
            mapping[software] = filtered_task_ids

    if software_filter:
        missing = [software for software in software_filter if software not in payload]
        if missing:
            raise ValueError(
                f"Software filter {missing} not present in task set `{task_index}`."
            )

    if task_id_filter:
        seen_task_ids = {task_id for task_ids in mapping.values() for task_id in task_ids}
        missing_ids = [task_id for task_id in task_id_filter if task_id not in seen_task_ids]
        if missing_ids:
            raise ValueError(
                f"Task id filter {missing_ids} not present in task set `{task_index}`."
            )

    if not mapping:
        raise ValueError(f"Task-set `{task_index}` does not contain any tasks after filtering.")

    return mapping


def _resolve_software_selection(
    *,
    task_index: str,
    test_config_base_dir: Path | None,
    software_filter: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        _load_task_index_mapping(
            task_index=task_index,
            test_config_base_dir=test_config_base_dir,
            software_filter=software_filter,
        ).keys()
    )


def _adapter_class_name(software: str) -> str:
    explicit_names = {
        "celluloid": "CelluloidAdapter",
        "jupyterlab": "JupyterLabAdapter",
    }
    if software in explicit_names:
        return explicit_names[software]
    if software.startswith("libreoffice_"):
        suffix = software.removeprefix("libreoffice_")
        return "LibreOffice" + "".join(part.capitalize() for part in suffix.split("_")) + "Adapter"
    return "".join(part.capitalize() for part in software.split("_")) + "Adapter"


def _create_expansion_adapter(software: str, tmp: Path, sandbox=None, mock: bool = False):
    """Create one of the expansion-program adapters.

    Software-owned workers are expected to implement their adapter modules under
    `asil.adapters.<software>` with class names following `<Software>Adapter`.
    """
    module_name = f"asil.adapters.{software}"
    class_name = _adapter_class_name(software)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Software `{software}` is registered but its adapter module `{module_name}` "
            "has not been implemented yet."
        ) from exc

    adapter_cls = getattr(module, class_name, None)
    if adapter_cls is None:
        raise RuntimeError(
            f"Software `{software}` is registered but class `{class_name}` was not found in "
            f"`{module_name}`."
        )

    if hasattr(adapter_cls, "from_evaluation_context"):
        return adapter_cls.from_evaluation_context(tmp=tmp, sandbox=sandbox, mock=mock)

    try:
        return adapter_cls()
    except TypeError as exc:
        raise RuntimeError(
            f"Software `{software}` adapter `{class_name}` must either support a zero-argument "
            "constructor or define `from_evaluation_context(tmp, sandbox, mock)`."
        ) from exc


def _resolve_gitea_token(sandbox=None) -> str:
    """Resolve a Gitea token from sandbox, env, or a mounted token file."""
    import os

    if sandbox:
        return sandbox.gitea_token

    token = os.environ.get("GITEA_TOKEN", "").strip()
    if token:
        return token

    token_file = os.environ.get("GITEA_TOKEN_FILE", "").strip()
    if token_file:
        path = Path(token_file)
        if path.exists():
            file_token = path.read_text().strip()
            if file_token:
                return file_token

    if os.environ.get("ASIL_SANDBOX", "").lower() == "true":
        shared_token = Path("/shared/gitea_token.txt")
        if shared_token.exists():
            file_token = shared_token.read_text().strip()
            if file_token:
                return file_token
        try:
            import requests

            base_url = os.environ.get("GITEA_URL", "http://gitea:3000").rstrip("/")
            admin = os.environ.get("GITEA_ADMIN", "asil_admin")
            password = os.environ.get("GITEA_PASSWORD", "asil_password")
            api = f"{base_url}/api/v1"
            auth = (admin, password)
            no_proxy = {"http": None, "https": None}

            requests.delete(
                f"{api}/users/{admin}/tokens/asil-eval-token",
                auth=auth,
                timeout=5,
                proxies=no_proxy,
            )
            resp = requests.post(
                f"{api}/users/{admin}/tokens",
                auth=auth,
                json={"name": "asil-eval-token", "scopes": ["all"]},
                timeout=5,
                proxies=no_proxy,
            )
            if resp.status_code in (200, 201):
                token = resp.json().get("sha1", "").strip()
                if token:
                    return token
        except Exception:
            pass

    return ""


def _create_adapter(software: str, tmp: Path, sandbox=None, mock: bool = False):
    """Create an adapter for the given software.

    When sandbox is provided (--docker mode), OBS uses SimpleOBSWSClient
    to connect to the obs-mock WebSocket server, and Blender uses MockBlenderAdapter
    (real Blender binary is only available inside the eval container).
    """
    if software == "multi_apps":
        from asil.adapters.multi_apps import MultiAppAdapter

        def _factory(child_software: str, child_tmp: Path, child_sandbox=None, child_mock: bool = False):
            return _create_adapter(
                child_software,
                child_tmp,
                sandbox=sandbox if child_sandbox is None else child_sandbox,
                mock=child_mock,
            )

        return MultiAppAdapter(tmp=tmp, sandbox=sandbox, mock=mock, adapter_factory=_factory)

    if software == "inkscape":
        from asil.adapters.inkscape import InkscapeAdapter
        svg = tmp / "test.svg"
        svg.write_text(_SVG_DEFAULT)
        return InkscapeAdapter(svg_path=svg)

    elif software == "libreoffice":
        from asil.adapters.libreoffice import LibreOfficeAdapter
        ods = tmp / "test.ods"
        import zipfile
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<office:document-content'
            ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
            ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
            ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
            ' office:version="1.2">'
            '<office:body><office:spreadsheet>'
            '<table:table table:name="Sheet1">'
            '<table:table-row>'
            '<table:table-cell office:value-type="string"><text:p>Revenue</text:p></table:table-cell>'
            '<table:table-cell office:value-type="float" office:value="100"><text:p>100</text:p></table:table-cell>'
            '</table:table-row>'
            '<table:table-row>'
            '<table:table-cell office:value-type="string"><text:p>Cost</text:p></table:table-cell>'
            '<table:table-cell office:value-type="float" office:value="60"><text:p>60</text:p></table:table-cell>'
            '</table:table-row>'
            '</table:table>'
            '</office:spreadsheet></office:body></office:document-content>'
        )
        manifest = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"'
            ' manifest:version="1.2">'
            '<manifest:file-entry manifest:full-path="/" manifest:version="1.2"'
            ' manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>'
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
            '</manifest:manifest>'
        )
        with zipfile.ZipFile(ods, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('mimetype', 'application/vnd.oasis.opendocument.spreadsheet',
                         compress_type=zipfile.ZIP_STORED)
            zf.writestr('META-INF/manifest.xml', manifest)
            zf.writestr('content.xml', content)
        return LibreOfficeAdapter(ods_path=ods)

    elif software == "blender":
        if mock or sandbox:
            # Docker host mode: no Blender binary, use mock adapter
            from asil.adapters.blender import MockBlenderAdapter
            return MockBlenderAdapter()
        from asil.adapters.blender import BlenderAdapter
        return BlenderAdapter()

    elif software == "obs":
        import os
        if mock:
            # Pure deterministic mock: in-process, no network, stable across host/container.
            from asil.adapters.obs import MockOBSWSClient, OBSAdapter
            return OBSAdapter(ws=MockOBSWSClient())
        use_real_gui = os.environ.get("OBS_REAL_GUI", "").lower() in {"1", "true", "yes"}
        if sandbox:
            from asil.adapters.obs import OBSAdapter
            if use_real_gui:
                # Real OBS launched inside the eval container exposes its websocket locally.
                return OBSAdapter(
                    host="localhost",
                    port=4444,
                    password=os.environ.get("OBS_WS_PASSWORD", ""),
                    use_real_gui=True,
                    ws_protocol=os.environ.get("OBS_WS_PROTOCOL", "auto"),
                )
            # Mock/docker mode: connect to obs-mock WebSocket server
            return OBSAdapter(
                host=sandbox.obs_ws_host,
                port=sandbox.obs_ws_port,
                password=os.environ.get("OBS_WS_PASSWORD", ""),
                use_real_gui=False,
                ws_protocol=os.environ.get("OBS_WS_PROTOCOL", "auto"),
            )
        from asil.adapters.obs import OBSAdapter
        if use_real_gui and os.environ.get("ASIL_SANDBOX", "").lower() == "true":
            # Managed eval runs execute inside the eval container without passing --docker,
            # so real OBS still needs to target its local websocket rather than obs-mock.
            return OBSAdapter(
                host="localhost",
                port=4444,
                password=os.environ.get("OBS_WS_PASSWORD", ""),
                use_real_gui=True,
                ws_protocol=os.environ.get("OBS_WS_PROTOCOL", "auto"),
            )
        return OBSAdapter(
            host=os.environ.get("OBS_WS_HOST", "localhost"),
            port=int(os.environ.get("OBS_WS_PORT", "4455")),
            password=os.environ.get("OBS_WS_PASSWORD", ""),
            use_real_gui=use_real_gui,
            ws_protocol=os.environ.get("OBS_WS_PROTOCOL", "auto"),
        )

    elif software == "gitea":
        import os
        if mock:
            from asil.adapters.gitea import MockGiteaAdapter

            return MockGiteaAdapter()
        from asil.adapters.gitea import GiteaAdapter
        base_url = os.environ.get("GITEA_URL", "http://localhost:3000")
        if sandbox:
            base_url = sandbox.gitea_url
        token = _resolve_gitea_token(sandbox)
        owner = os.environ.get("GITEA_OWNER", "asil_admin")
        repo = os.environ.get("GITEA_REPO", "test-repo")
        if not token:
            raise RuntimeError(
                "Gitea token not found. Set GITEA_TOKEN, point GITEA_TOKEN_FILE to a token file, "
                "or use --docker so the sandbox can provision one."
            )
        return GiteaAdapter(base_url=base_url, token=token, owner=owner, repo=repo)

    elif software in {
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
    }:
        return _create_expansion_adapter(software, tmp, sandbox=sandbox, mock=mock)

    else:
        raise ValueError(f"Unknown software: {software}")


# ---------------------------------------------------------------------------
# Per-step rendering
# ---------------------------------------------------------------------------

def _render_step(adapter, task_dir: Path, step_num: int) -> RenderArtifact | None:
    """Render current adapter state to a PNG page artifact plus metadata."""
    output = task_dir / f"step_{step_num}.png"
    meta_path = task_dir / f"step_{step_num}.render.json"
    artifact = adapter.describe_rendering()
    artifact.filename = output.name
    try:
        setattr(adapter, "_last_capture_complete", artifact.capture_complete)
        if hasattr(adapter, "render_to_png"):
            adapter.render_to_png(output_path=output)
        else:
            raise RuntimeError(f"{type(adapter).__name__} does not implement render_to_png()")
        artifact.capture_complete = bool(getattr(adapter, "_last_capture_complete", artifact.capture_complete))
        meta_path.write_text(
            json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return artifact
    except Exception as e:
        print(f"    [render] step_{step_num}.png failed: {e}")
        return None


def _apply_task_render_target(adapter, task) -> None:
    """Pass task-level render targeting into adapters that support it."""
    render_target = getattr(task, "render_target", {}) or {}
    setter = getattr(adapter, "set_render_target", None)
    if callable(setter):
        setter(render_target)


def _write_task_info(
    task,
    task_dir: Path,
    *,
    asil_success_hint: str = "evaluator",
    independent_raw_validation: bool = False,
) -> None:
    """Persist human-readable task metadata in each result directory."""
    task_info = {
        "task_id": task.id,
        "task_name": task.description,
        "instruction": task.instruction,
        "software": task.software,
        "snapshot": getattr(task, "snapshot", ""),
        "initial_state": getattr(task, "initial_state", "default"),
        "validation": getattr(task, "validation", {}),
        "evaluator": getattr(task, "evaluator", {}),
        "gui_expectations": getattr(task, "gui_expectations", {}),
        "render_target": getattr(task, "render_target", {}),
        "asil_success_hint": asil_success_hint,
        "independent_raw_validation": independent_raw_validation,
    }
    (task_dir / "task_info.json").write_text(
        json.dumps(task_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_step_action(
    task,
    task_dir: Path,
    step_num: int,
    trace,
    *,
    asil_success_hint: str = "evaluator",
    independent_raw_validation: bool = False,
) -> None:
    """Persist instruction, formatted thought, and action for each agent step."""
    payload = {
        "instruction": task.instruction,
        "thought": trace.thought,
        "action": trace.action.model_dump(mode="json"),
        "model_latency_ms": round(float(getattr(trace, "model_latency_ms", 0.0) or 0.0), 2),
        "action_execution_latency_ms": round(float(getattr(trace, "action_execution_latency_ms", 0.0) or 0.0), 2),
        "render_latency_ms": round(float(getattr(trace, "render_latency_ms", 0.0) or 0.0), 2),
        "evaluation_latency_ms": round(float(getattr(trace, "evaluation_latency_ms", 0.0) or 0.0), 2),
        "step_total_latency_ms": round(float(getattr(trace, "step_total_latency_ms", 0.0) or 0.0), 2),
        "asil_success_hint": asil_success_hint,
        "independent_raw_validation": independent_raw_validation,
    }
    (task_dir / f"step_{step_num}_action.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_asil_agentic_run(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "agent", False)):
        return True
    return (
        getattr(args, "participant", "asil") == "asil"
        and getattr(args, "run_mode", "single") == "single"
        and getattr(args, "asil_execution", "deterministic") == "agentic"
    )


def _is_comparison_run(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "comparison", False)):
        return True
    return getattr(args, "run_mode", "single") == "comparison"


def _is_single_cli_run(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "participant", "asil") == "cli"
        and getattr(args, "run_mode", "single") == "single"
    )


def _is_single_gui_run(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "participant", "asil") == "gui"
        and getattr(args, "run_mode", "single") == "single"
    )


def _result_root_from_args(args: argparse.Namespace) -> Path:
    task_index = getattr(args, "task_set", None) or getattr(args, "test_all", "test_full15.json")
    return _result_root_for_config(
        BenchmarkConfig(
            software=tuple(getattr(args, "software", ()) or getattr(args, "software_filter", ()) or ()),
            task_id_filter=tuple(getattr(args, "task_id_filter", ()) or ()),
            task_index=task_index,
            output_dir=args.output_dir,
            output_json=args.output,
            participant=getattr(args, "participant", "asil"),
            run_mode=getattr(args, "run_mode", "single"),
            comparison_participants=tuple(getattr(args, "comparison_participants", ["asil", "cli", "gui"])),
            asil_execution=getattr(args, "asil_execution", "deterministic"),
            asil_success_hint=getattr(args, "asil_success_hint", "evaluator"),
            independent_raw_validation=bool(
                getattr(args, "independent_raw_validation", False)
            ),
            execution_mode=getattr(args, "execution_mode", None),
            provider=args.provider,
            model=args.model,
            max_steps=args.max_steps,
            test_config_base_dir=args.test_config_base_dir,
            osworld_format=bool(args.osworld_format),
            docker=bool(args.docker),
            managed_docker=bool(getattr(args, "managed_docker", False)),
            mock=bool(args.mock),
            dry_run=bool(args.dry_run),
        )
    )


def _write_args_json(args: argparse.Namespace, result_root: Path) -> None:
    is_asil_agentic = _is_asil_agentic_run(args)
    is_single_gui = _is_single_gui_run(args)
    is_comparison = _is_comparison_run(args)
    args_data = {
        "action_space": (
            "pyautogui"
            if is_single_gui
            else ("mixed" if is_comparison else ("shell" if _is_single_cli_run(args) else "semantic"))
        ),
        "observation_type": (
            "screenshot"
            if is_single_gui
            else ("mixed" if is_comparison else ("terminal" if _is_single_cli_run(args) else "structured"))
        ),
        "observation_source": (
            "screenshot"
            if is_single_gui
            else ("mixed" if is_comparison else ("terminal" if _is_single_cli_run(args) else "structured"))
        ),
        "gui_action_space": "pyautogui" if (is_single_gui or is_comparison) else "",
        "model": args.model or "(default)",
        "provider": args.provider if (is_asil_agentic or is_single_gui or is_comparison) else "n/a",
        "max_steps": args.max_steps,
        "software": getattr(args, "software", []),
        "task_set": getattr(args, "task_set", None) or getattr(args, "test_all", "test_full15.json"),
        "task_set_name": Path(getattr(args, "task_set", None) or getattr(args, "test_all", "test_full15.json")).stem,
        "mode": (
            "comparison"
            if is_comparison
            else ("agent" if is_asil_agentic else ("gui" if is_single_gui else "deterministic"))
        ),
        "participant": getattr(args, "participant", "asil"),
        "run_mode": getattr(args, "run_mode", "single"),
        "asil_execution": getattr(args, "asil_execution", "deterministic"),
        "asil_success_hint": getattr(args, "asil_success_hint", "evaluator"),
        "independent_raw_validation": bool(
            getattr(args, "independent_raw_validation", False)
        ),
        "comparison_participants": getattr(args, "comparison_participants", ["asil", "cli", "gui"]),
        "resume": getattr(args, "resume", True),
        "force_rerun": getattr(args, "force_rerun", False),
        "rerun_failed_only": getattr(args, "rerun_failed_only", False),
        "num_envs": getattr(args, "num_envs", 1),
        "timestamp": datetime.now().isoformat(),
    }
    (result_root / "args.json").write_text(
        json.dumps(args_data, indent=2, ensure_ascii=False)
    )


def _load_tasks_for_software(args: argparse.Namespace, sw: str) -> list:
    from asil.eval.runner import TaskDefinition

    project_root = PROJECT_ROOT
    config_base = args.test_config_base_dir or (project_root / "evaluation_examples")
    index_name = getattr(args, "task_set", None) or getattr(args, "test_all", "test_full15.json")
    index_file = config_base / index_name
    filtered_task_ids = set(_selected_task_mapping_from_args(args).get(sw, []))

    if index_file.exists():
        tasks = TaskDefinition.from_index(index_file, domain=sw)
    else:
        tasks_file = project_root / "src" / "asil" / "eval" / "tasks" / f"{sw}.json"
        if not tasks_file.exists():
            return []
        tasks = TaskDefinition.from_json(tasks_file)
    return [t for t in tasks if t.actions and (not filtered_task_ids or t.id in filtered_task_ids)]


def _selected_task_mapping_from_args(args: argparse.Namespace) -> dict[str, list[str]]:
    return _load_task_index_mapping(
        task_index=getattr(args, "task_set", None) or getattr(args, "test_all", "test_full15.json"),
        test_config_base_dir=args.test_config_base_dir,
        software_filter=tuple(getattr(args, "software_filter", ()) or getattr(args, "software", ()) or ()),
        task_id_filter=tuple(getattr(args, "task_id_filter", ()) or ()),
    )


def _comparison_label(args: argparse.Namespace) -> str:
    label = "deterministic"
    if getattr(args, "asil_execution", "deterministic") == "agentic":
        model_label = args.model or args.provider or "default"
        label = f"agentic__{model_label}"
    return label


def _rebuild_output_json(args: argparse.Namespace, summary_dir: Path, *, is_comparison: bool) -> dict:
    entries = load_summary_entries(summary_dir)
    aggregate = build_aggregate(entries, is_comparison=is_comparison)
    if is_comparison:
        payload: dict[str, dict[str, dict[str, object]]] = {}
        for entry in entries:
            software = str(entry["software"])
            participant_name = str(entry["participant"])
            software_bucket = payload.setdefault(software, {})
            participant_bucket = software_bucket.setdefault(
                participant_name,
                {
                    "aggregate": aggregate.get(participant_name, {}).get(software, {}),
                    "per_software": {software: aggregate.get(participant_name, {}).get(software, {})},
                    "tasks": [],
                },
            )
            participant_bucket["tasks"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload

    payload = {}
    for entry in entries:
        software = str(entry["software"])
        bucket = payload.setdefault(
            software,
            {
                "aggregate": aggregate.get(software, {}),
                "per_software": {software: aggregate.get(software, {})},
                "tasks": [],
            },
        )
        bucket["tasks"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _task_state_snapshot(adapter, tasks: list, sw: str) -> tuple[Path | None, bytes | None, dict[str, bytes]]:
    if callable(getattr(adapter, "prepare_task", None)):
        return None, None, {}
    src = getattr(adapter, "source_path", None)
    if callable(src):
        src = src()
    original_bytes = src.read_bytes() if src and src.exists() and src.is_file() else None
    initial_state_cache: dict[str, bytes] = {}
    needed_states = {
        (getattr(task, "initial_state", "default") or "default")
        for task in tasks
    }
    if src and sw == "inkscape":
        src.write_text(_SVG_DEFAULT)
        initial_state_cache["default"] = src.read_bytes()
        src.write_text(_SVG_BLANK)
        initial_state_cache["blank"] = src.read_bytes()
        src.write_bytes(initial_state_cache["default"])
    elif src and src.is_file() and hasattr(adapter, "setup_state") and callable(adapter.setup_state):
        for state_key in sorted(needed_states):
            if hasattr(adapter, "clear_gui_shadow_state"):
                adapter.clear_gui_shadow_state()
            adapter.setup_state(state_key)
            if src.exists():
                initial_state_cache[state_key] = src.read_bytes()
        if "default" in initial_state_cache:
            src.write_bytes(initial_state_cache["default"])
        elif initial_state_cache:
            src.write_bytes(next(iter(initial_state_cache.values())))
    return src, original_bytes, initial_state_cache


def _restore_task_state(adapter, task, src: Path | None, original_bytes: bytes | None, initial_state_cache: dict[str, bytes]) -> None:
    prepare_task = getattr(adapter, "prepare_task", None)
    if callable(prepare_task):
        prepare_task(task)
        return
    if src is not None and src.is_file():
        state_key = getattr(task, "initial_state", "default") or "default"
        task_bytes = initial_state_cache.get(state_key, original_bytes)
        if task_bytes is not None:
            src.write_bytes(task_bytes)
        if hasattr(adapter, "clear_gui_shadow_state"):
            adapter.clear_gui_shadow_state()
    elif hasattr(adapter, "reset_state"):
        adapter.reset_state()
        initial_state = getattr(task, "initial_state", "default") or "default"
        if hasattr(adapter, "setup_state"):
            adapter.setup_state(initial_state)
        if hasattr(adapter, "clear_gui_shadow_state"):
            adapter.clear_gui_shadow_state()


def _record_task_summary(all_task_summaries: list[dict], *, task_result, participant: str) -> None:
    all_task_summaries.append({
        "participant": participant,
        "software": task_result.software,
        "task_id": task_result.task_id,
        "difficulty": task_result.difficulty,
        "status": "success" if task_result.success else "failed",
        "success": bool(task_result.success),
        "score": task_result.score,
        "steps": task_result.steps,
        "e2e_time_s": round(task_result.e2e_time_s, 3),
        "avg_latency_ms": round(float(getattr(task_result, "avg_latency_ms", 0.0) or 0.0), 3),
        "deadlocked": bool(getattr(task_result, "deadlocked", False)),
        "timestamp": datetime.now().isoformat(),
    })


def _save_task_result_artifacts(result, task, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_task_info(task, task_dir)
    result.save_trajectory(task_dir)
    result.save_result(task_dir)
    (task_dir / "evaluation.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    )


def _summarize_metrics(all_results: dict, *, is_comparison: bool) -> dict:
    if is_comparison:
        aggregate: dict[str, dict[str, dict]] = {}
        for sw, comparison in all_results.items():
            for participant, payload in comparison.items():
                aggregate.setdefault(participant, {})
                aggregate[participant][sw] = payload["metrics"].get("aggregate", {})
                aggregate[participant][sw]["per_software"] = payload["metrics"].get("per_software", {})
        return aggregate

    agg_summary = {}
    for sw, metrics in all_results.items():
        agg_summary[sw] = metrics.get("aggregate", {})
        agg_summary[sw]["per_software"] = metrics.get("per_software", {})
    return agg_summary


# ---------------------------------------------------------------------------
# Agent mode with per-step artifacts
# ---------------------------------------------------------------------------

def _run_agent_with_artifacts(
    adapter,
    task,
    llm_fn,
    max_steps,
    task_dir,
    *,
    success_hint_policy: str = "evaluator",
    independent_raw_validation: bool = False,
):
    """Run agent loop with per-step observation/action recording and visual rendering.

    Produces OSWorld-compatible output:
      - traj.jsonl  (per-step trajectory with full observation + action)
      - result.txt  (final score)
      - step_N.png + step_N.render.json  (page rendering per step)
    """
    from asil.agent import ASILAgent, format_task_success_hint
    from asil.eval.metrics import TaskResult, StepResult
    from asil.eval.evaluator import evaluate_task_result

    task_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale visual artifacts from previous runs to avoid confusion
    for old_file in task_dir.glob("step_*.png"):
        old_file.unlink()
    for old_file in task_dir.glob("step_*.pdf"):
        old_file.unlink()
    for old_file in task_dir.glob("step_*.render.json"):
        old_file.unlink()
    for old_file in task_dir.glob("step_*_action.json"):
        old_file.unlink()

    if success_hint_policy not in {"evaluator", "none"}:
        raise ValueError(f"Unknown ASIL success-hint policy: {success_hint_policy}")
    _write_task_info(
        task,
        task_dir,
        asil_success_hint=success_hint_policy,
        independent_raw_validation=independent_raw_validation,
    )

    agent = ASILAgent(adapter=adapter, llm_fn=llm_fn, max_steps=max_steps, software=task.software,
                      hybrid_vision=os.environ.get("ASIL_HYBRID_VISION", "").strip().lower() in {"1", "true", "yes", "on"})
    success_hint = (
        format_task_success_hint(task)
        if success_hint_policy == "evaluator"
        else ""
    )

    start = time.monotonic()
    traj_entries = []
    step_results = []
    deadlocked = False
    last_obs_hash = ""
    repeat_count = 0

    # Initial observation + render step_0
    obs = adapter.observe()
    obs_element_count = len(obs.interactive_elements)
    _apply_task_render_target(adapter, task)
    render_artifact = _render_step(adapter, task_dir, 0)
    render_file = render_artifact.filename if render_artifact else None
    print(f"    step_0: initial state ({obs_element_count} elements) -> {render_file or 'no render'}")

    for i in range(max_steps):
        step_start = time.monotonic()

        try:
            _, action, new_obs, trace = agent.step(task.instruction, obs, success_hint=success_hint)
            step_success = True
        except Exception as e:
            print(f"    step_{i+1}: agent error — {e}")
            step_success = False
            break

        # Render after this step
        render_start = time.monotonic()
        render_artifact = _render_step(adapter, task_dir, i + 1)
        render_latency = (time.monotonic() - render_start) * 1000
        render_file = render_artifact.filename if render_artifact else None

        evaluation_start = time.monotonic()
        step_evaluation = evaluate_task_result(task, new_obs)
        evaluation_latency = (time.monotonic() - evaluation_start) * 1000
        latency = (time.monotonic() - step_start) * 1000
        computed_total_latency = max(
            latency,
            float(getattr(trace, "model_latency_ms", 0.0) or 0.0)
            + float(getattr(trace, "action_execution_latency_ms", 0.0) or 0.0)
            + render_latency
            + evaluation_latency,
        )

        trace.render_latency_ms = round(render_latency, 2)
        trace.evaluation_latency_ms = round(evaluation_latency, 2)
        trace.step_total_latency_ms = round(computed_total_latency, 2)
        _write_step_action(
            task,
            task_dir,
            i + 1,
            trace,
            asil_success_hint=success_hint_policy,
            independent_raw_validation=independent_raw_validation,
        )

        # Deadlock detection
        obs_hash = str(sorted(e.id for e in new_obs.interactive_elements))
        if obs_hash == last_obs_hash:
            repeat_count += 1
            if repeat_count >= 5:
                deadlocked = True
        else:
            repeat_count = 0
            last_obs_hash = obs_hash

        # Determine render filename for this step
        # Build traj entry with full observation + action
        traj_entries.append({
            "step_num": i + 1,
            "action_timestamp": datetime.now().isoformat(),
            "instruction": task.instruction,
            "thought": trace.thought,
            "observation": obs.model_dump(mode="json"),
            "action": action.model_dump(mode="json"),
            "step_action_file": f"step_{i+1}_action.json",
            "screenshot_file": render_file or "",
            "render_metadata_file": f"step_{i+1}.render.json" if render_file else "",
            "render_kind": render_artifact.kind if render_artifact else "",
            "render_backend": render_artifact.backend if render_artifact else "",
            "render_actual_page": render_artifact.actual_page if render_artifact else False,
            "render_capture_complete": render_artifact.capture_complete if render_artifact else False,
            "latency_ms": round(latency, 2),
            "model_latency_ms": getattr(trace, "model_latency_ms", 0.0),
            "action_execution_latency_ms": getattr(trace, "action_execution_latency_ms", 0.0),
            "render_latency_ms": round(render_latency, 2),
            "evaluation_latency_ms": round(evaluation_latency, 2),
            "step_total_latency_ms": round(computed_total_latency, 2),
            "step_score": step_evaluation.score,
            "step_success_snapshot": step_evaluation.success,
            "success": step_success,
            "done": action.action_type == "done",
            "asil_success_hint": success_hint_policy,
            "independent_raw_validation": independent_raw_validation,
        })

        # Build StepResult for metrics
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

        print(f"    step_{i+1}: {action.action_type} -> {action.target[:50]}  "
              f"({latency:.0f}ms, {render_file or 'no render'})")

        obs = new_obs
        if action.action_type == "done":
            break
        if deadlocked:
            print(f"    deadlocked after {i+1} steps")
            break

    e2e = time.monotonic() - start
    evaluation = evaluate_task_result(task, obs)

    # Write traj.jsonl
    with open(task_dir / "traj.jsonl", "w") as f:
        for entry in traj_entries:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    (task_dir / "evaluation.json").write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if independent_raw_validation:
        from asil.eval.raw_validation import validate_raw_final_state

        independent_report = validate_raw_final_state(
            adapter,
            task,
            evaluator_score=evaluation.score,
        )
        (task_dir / "independent_evaluation.json").write_text(
            json.dumps(independent_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Write result.txt
    score = evaluation.score
    (task_dir / "result.txt").write_text(str(score))

    return TaskResult(
        task_id=task.id,
        software=task.software,
        difficulty=task.difficulty,
        instruction=task.instruction,
        success=evaluation.success,
        score=score,
        steps=len(step_results),
        step_results=step_results,
        e2e_time_s=e2e,
        deadlocked=deadlocked,
        observation_element_count=obs_element_count,
        total_element_count=obs_element_count,
    )


def _execute_from_namespace(args: argparse.Namespace) -> int:
    is_asil_agentic = _is_asil_agentic_run(args)
    is_single_gui = _is_single_gui_run(args)
    is_comparison = _is_comparison_run(args)

    if args.dry_run:
        if args.docker:
            sandbox = ASILSandbox()
            if not sandbox.is_docker_available:
                print("ERROR: Docker not available.")
                return 1
            print("Docker sandbox available.")
        else:
            print("Local eval mode. Software:", ", ".join(args.software))
            if is_asil_agentic:
                print(f"ASIL agent mode: provider={args.provider}, model={args.model or '(default)'}")
            elif is_single_gui:
                print(f"GUI agent mode: provider={args.provider}, model={args.model or '(default)'}")
            elif is_comparison:
                print(
                    "Comparison mode: "
                    f"participants={','.join(getattr(args, 'comparison_participants', ['asil', 'cli', 'gui']))}, "
                    f"asil_execution={getattr(args, 'asil_execution', 'deterministic')}"
                )
            # Check rendering tools
            import shutil as _s
            for tool, name in [("inkscape", "Inkscape"), ("libreoffice", "LibreOffice"), ("xdotool", "xdotool")]:
                path = _s.which(tool) or _s.which("soffice" if tool == "libreoffice" else "")
                print(f"  {name}: {'OK — ' + path if path else 'NOT INSTALLED'}")
        return 0

    sandbox = None
    if args.docker:
        sandbox = ASILSandbox()
        if not sandbox.is_docker_available:
            print("ERROR: Docker not available.")
            return 1
        print("Starting Docker sandbox...")
        sandbox.__enter__()

    try:
        _run_evaluation(args, sandbox)
    finally:
        if sandbox:
            sandbox.__exit__(None, None, None)
    return 0


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRunResult:
    materialized_software = config.software or _resolve_software_selection(
        task_index=config.task_index,
        test_config_base_dir=config.test_config_base_dir,
        software_filter=(),
    )
    materialized_config = BenchmarkConfig(
        software=materialized_software,
        task_id_filter=config.task_id_filter,
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
        managed_docker=config.managed_docker,
        mock=config.mock,
        dry_run=config.dry_run,
        resume=config.resume,
        force_rerun=config.force_rerun,
        rerun_failed_only=config.rerun_failed_only,
        num_envs=config.num_envs,
    )
    args = _namespace_from_config(materialized_config)
    exit_code = _execute_from_namespace(args)
    return BenchmarkRunResult(
        exit_code=exit_code,
        config=materialized_config,
        result_root=_result_root_for_config(materialized_config),
        output_json=materialized_config.output_json,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_namespace(args)
    except ValueError as exc:
        parser.error(str(exc))
    return run_benchmark(config).exit_code


def _task_result_from_gui_exception(adapter, task, exc: Exception, task_dir: Path):
    from asil.eval.metrics import StepResult, TaskResult
    from asil.gui_agent.session import GUISessionStartupError

    task_dir.mkdir(parents=True, exist_ok=True)
    error_payload = {
        "score": 0.0,
        "success": False,
        "error": str(exc),
    }
    runtime_error_text = str(exc)
    action_type = "FAIL"
    params = {"error": str(exc)}
    steps = 1

    if isinstance(exc, GUISessionStartupError):
        error_payload["error_category"] = exc.category
        runtime_error_text = f"{exc.category}: {exc}"
        action_type = "STARTUP_FAIL"
        params["error_category"] = exc.category
        steps = 0

    (task_dir / "evaluation.json").write_text(
        json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (task_dir / "runtime_error.txt").write_text(runtime_error_text + "\n", encoding="utf-8")
    result = TaskResult(
        task_id=task.id,
        software=task.software,
        difficulty=task.difficulty,
        instruction=task.instruction,
        success=False,
        score=0.0,
        steps=steps,
        step_results=[
            StepResult(
                step_num=1,
                action_type=action_type,
                target=adapter.app_name,
                params=params,
                success=False,
                observation_source="screenshot",
            )
        ],
    )
    setattr(result, "_skip_summary", True)
    return result


def _run_evaluation(args, sandbox=None):
    """Run evaluation for all requested software."""
    import tempfile
    from asil.eval.runner import run_evaluation
    from asil.eval.metrics import StepResult, TaskResult, compute_metrics
    from asil.baselines.cli_baseline import CLIBaseline
    from asil.gui_agent import GUISessionStartupError, create_gui_llm_fn, run_gui_agent_task

    is_asil_agentic = _is_asil_agentic_run(args)
    is_comparison = _is_comparison_run(args)
    is_single_cli = _is_single_cli_run(args)
    is_single_gui = _is_single_gui_run(args)

    asil_llm_fn = None
    if is_asil_agentic or (is_comparison and getattr(args, "asil_execution", "deterministic") == "agentic"):
        from asil.agent import create_llm_fn
        asil_llm_fn = create_llm_fn(provider=args.provider, model=args.model)
        print(f"ASIL agent mode: provider={args.provider}, model={args.model or '(default)'}")

    if is_single_gui or is_comparison:
        if is_single_gui:
            print(f"GUI agent mode: provider={args.provider}, model={args.model or '(default)'}")

    tmp = Path(tempfile.mkdtemp())
    all_results = {}
    result_root = _result_root_from_args(args)
    result_root.mkdir(parents=True, exist_ok=True)
    task_mapping = _selected_task_mapping_from_args(args)
    args.software = list(task_mapping.keys())
    _write_args_json(args, result_root)
    summary_dir = result_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    selection = select_pending_tasks(
        task_mapping,
        result_root=result_root,
        run_mode=getattr(args, "run_mode", "single"),
        participant=getattr(args, "participant", "asil"),
        comparison_participants=tuple(getattr(args, "comparison_participants", ["asil", "cli", "gui"])),
        resume=bool(getattr(args, "resume", True)),
        force_rerun=bool(getattr(args, "force_rerun", False)),
        rerun_failed_only=bool(getattr(args, "rerun_failed_only", False)),
    )
    pending_mapping = group_task_keys(list(selection.pending_tasks))

    existing_entries = load_summary_entries(summary_dir)
    if existing_entries:
        pending_key_set = {(task.software, task.task_id) for task in selection.pending_tasks}
        existing_entries = [
            entry
            for entry in existing_entries
            if (str(entry.get("software", "")), str(entry.get("task_id", ""))) not in pending_key_set
        ]
        (summary_dir / "results.json").write_text(
            json.dumps(existing_entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    all_task_summaries = list(existing_entries)

    if not selection.pending_tasks:
        write_aggregate(summary_dir, all_task_summaries, is_comparison=is_comparison)
        _rebuild_output_json(args, summary_dir, is_comparison=is_comparison)
        print(
            f"All selected tasks already complete under {result_root} "
            f"(skipped={selection.skipped_tasks}/{selection.total_tasks})."
        )
        return

    def _participant_uses_mock(sw: str, participant: str) -> bool:
        use_mock = bool(getattr(args, "mock", False))
        if sw == "obs" and participant != "gui":
            use_mock = True
        return use_mock

    def _persist_task_summary(task_result, participant_name: str) -> None:
        if bool(getattr(task_result, "_skip_summary", False)):
            return
        _record_task_summary(all_task_summaries, task_result=task_result, participant=participant_name)
        append_summary_entry(summary_dir, all_task_summaries[-1])

    def _run_agentic_tasks(adapter, runnable_tasks, sw: str, base_root: Path, participant_name: str):
        task_results = []
        src, original_bytes, initial_state_cache = _task_state_snapshot(adapter, runnable_tasks, sw)
        for task in runnable_tasks:
            _restore_task_state(adapter, task, src, original_bytes, initial_state_cache)
            _apply_task_render_target(adapter, task)
            task_dir = base_root / sw / task.id
            print(f"\n  [{participant_name}:{task.id}] {task.description}")
            result = _run_agent_with_artifacts(
                adapter,
                task,
                asil_llm_fn,
                args.max_steps,
                task_dir,
                success_hint_policy=getattr(args, "asil_success_hint", "evaluator"),
                independent_raw_validation=bool(
                    getattr(args, "independent_raw_validation", False)
                ),
            )
            task_results.append(result)
            _persist_task_summary(result, participant_name)
            status = "PASS" if result.success else "FAIL"
            print(f"    => {status} (score={result.score}, steps={result.steps}, e2e={result.e2e_time_s:.2f}s)")
        return {
            "results": task_results,
            "metrics": compute_metrics(task_results),
        }

    def _run_gui_tasks(adapter, runnable_tasks, sw: str, base_root: Path, participant_name: str):
        task_results = []
        src, original_bytes, initial_state_cache = _task_state_snapshot(adapter, runnable_tasks, sw)
        for task in runnable_tasks:
            _restore_task_state(adapter, task, src, original_bytes, initial_state_cache)
            _apply_task_render_target(adapter, task)
            task_dir = base_root / sw / task.id
            print(f"\n  [{participant_name}:{task.id}] {task.description}")
            # Give every task its own conversation object. A timed-out model
            # request runs in a daemon thread that cannot be cancelled; if that
            # thread returns late, it must not be able to mutate the next task's
            # computer-use history.
            task_gui_llm_fn = create_gui_llm_fn(provider=args.provider, model=args.model)
            try:
                result = run_gui_agent_task(
                    adapter,
                    task,
                    task_gui_llm_fn,
                    max_steps=args.max_steps,
                    task_dir=task_dir,
                )
            except Exception as exc:
                result = _task_result_from_gui_exception(adapter, task, exc, task_dir)
            task_results.append(result)
            _persist_task_summary(result, participant_name)
            status = "PASS" if result.success else "FAIL"
            print(f"    => {status} (score={result.score}, steps={result.steps}, e2e={result.e2e_time_s:.2f}s)")
        return {
            "results": task_results,
        "metrics": compute_metrics(task_results),
    }

    def _run_deterministic_tasks(participant_adapter, runnable_tasks, sw: str, participant_name: str, base_root: Path | None):
        task_results = run_evaluation(participant_adapter, runnable_tasks, isolate_tasks=True)
        metrics = compute_metrics(task_results)
        task_lookup = {task.id: task for task in runnable_tasks}
        if base_root is not None:
            for result in task_results:
                _save_task_result_artifacts(result, task_lookup[result.task_id], base_root / sw / result.task_id)
                _persist_task_summary(result, participant_name)
        return {
            "results": task_results,
            "metrics": metrics,
        }

    for sw, selected_task_ids in pending_mapping.items():
        print(f"\n{'='*60}")
        print(f"  {sw.upper()}")
        print(f"{'='*60}")
        sw_tmp = tmp / sw
        sw_tmp.mkdir(parents=True, exist_ok=True)
        runnable_tasks = [
            task for task in _load_tasks_for_software(args, sw)
            if task.id in set(selected_task_ids)
        ]
        if not runnable_tasks:
            print(f"  No task definitions found for {sw}")
            continue

        if is_asil_agentic:
            adapter = _create_adapter(sw, sw_tmp, sandbox, mock=_participant_uses_mock(sw, "asil"))
            payload = _run_agentic_tasks(adapter, runnable_tasks, sw, result_root, "asil")
            all_results[sw] = payload["metrics"]
            agg = payload["metrics"].get("aggregate", {})
            print(f"\n  [Summary] SR={agg.get('success_rate', 0):.1%}, "
                  f"passed={agg.get('passed', 0)}/{agg.get('total_tasks', 0)}, "
                  f"avg_steps={agg.get('avg_steps', 0):.1f}, "
                  f"avg_latency={agg.get('avg_latency_ms', 0):.0f}ms")

        elif is_comparison:
            comparison = {}
            for participant_name in getattr(args, "comparison_participants", ["asil", "cli", "gui"]):
                participant_tmp = sw_tmp / participant_name
                participant_tmp.mkdir(parents=True, exist_ok=True)
                adapter = _create_adapter(
                    sw,
                    participant_tmp,
                    sandbox,
                    mock=_participant_uses_mock(sw, participant_name),
                )
                participant_root = result_root / participant_name
                if participant_name == "asil":
                    if getattr(args, "asil_execution", "deterministic") == "agentic":
                        payload = _run_agentic_tasks(adapter, runnable_tasks, sw, participant_root, "asil")
                    else:
                        payload = _run_deterministic_tasks(adapter, runnable_tasks, sw, "asil", participant_root)
                elif participant_name == "cli":
                    payload = _run_deterministic_tasks(CLIBaseline(adapter), runnable_tasks, sw, "cli", participant_root)
                elif participant_name == "gui":
                    payload = _run_gui_tasks(adapter, runnable_tasks, sw, participant_root, "gui")
                else:
                    raise ValueError(f"Unknown comparison participant: {participant_name}")
                comparison[participant_name] = payload
                agg = payload["metrics"].get("aggregate", {})
                print(f"  {participant_name}: SR={agg.get('success_rate', 0):.1%}, "
                      f"avg_latency={agg.get('avg_latency_ms', 0):.1f}ms, "
                      f"avg_steps={agg.get('avg_steps', 0):.1f}")
            all_results[sw] = comparison

        else:
            participant_name = getattr(args, "participant", "asil")
            adapter = _create_adapter(sw, sw_tmp, sandbox, mock=_participant_uses_mock(sw, participant_name))
            if is_single_gui:
                payload = _run_gui_tasks(adapter, runnable_tasks, sw, result_root, "gui")
            else:
                participant_adapter = CLIBaseline(adapter) if is_single_cli else adapter
                artifacts_root = result_root
                payload = _run_deterministic_tasks(participant_adapter, runnable_tasks, sw, participant_name, artifacts_root)
            all_results[sw] = payload["metrics"]
            agg = payload["metrics"].get("aggregate", {})
            print(f"  SR={agg.get('success_rate', 0):.1%}, "
                  f"passed={agg.get('passed', 0)}/{agg.get('total_tasks', 0)}, "
                  f"avg_steps={agg.get('avg_steps', 0):.1f}")

    if all_task_summaries:
        write_aggregate(summary_dir, all_task_summaries, is_comparison=is_comparison)
        _rebuild_output_json(args, summary_dir, is_comparison=is_comparison)
        print(f"\nSummary saved to {summary_dir}/")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
