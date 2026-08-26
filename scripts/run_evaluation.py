#!/usr/bin/env python3
"""Compatibility wrapper around the public ASIL benchmark entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asil.benchmark as _benchmark  # noqa: E402
from asil.benchmark import (  # noqa: E402
    SOFTWARE_CHOICES,
    BenchmarkConfig,
    BenchmarkRunResult,
    _adapter_class_name,
    _apply_task_render_target,
    _create_adapter,
    _create_expansion_adapter,
    _render_step,
    _resolve_gitea_token,
    _run_agent_with_artifacts,
    _write_step_action,
    _write_task_info,
    build_arg_parser,
    config_from_namespace,
    main,
    run_benchmark,
)


def _run_evaluation(args, sandbox=None):
    """Compatibility bridge that preserves script-level monkeypatching in tests."""
    overrides = {
        "_create_adapter": _create_adapter,
        "_apply_task_render_target": _apply_task_render_target,
        "_run_agent_with_artifacts": _run_agent_with_artifacts,
        "_write_task_info": _write_task_info,
        "_write_step_action": _write_step_action,
        "_render_step": _render_step,
    }
    originals = {name: getattr(_benchmark, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(_benchmark, name, value)
        return _benchmark._run_evaluation(args, sandbox=sandbox)
    finally:
        for name, value in originals.items():
            setattr(_benchmark, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
