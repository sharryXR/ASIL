"""Command-line interface for grounded ASIL software onboarding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from asil.protocol import Action
from asil.softwaregen.audit import audit_bundle
from asil.softwaregen.generator import (
    generate_extension,
    load_extension_bundle,
    load_onboarding_profile,
    write_extension_bundle,
)
from asil.softwaregen.evidence import build_deployment_evidence_report
from asil.softwaregen.models import InterfacePlan
from asil.softwaregen.provider import DeterministicSoftwareGenProvider, OpenAISoftwareGenProvider
from asil.softwaregen.qualification import load_reference_catalog, qualify_profile
from asil.softwaregen.validation import docker_probe_extension, probe_extension


def _emit(payload: Any, output: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate and audit an extension bundle.")
    generate.add_argument("profile", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--provider", choices=["openai", "deterministic"], default="openai")
    generate.add_argument("--plan-file", type=Path)
    generate.add_argument("--model", default="gpt-5.4")
    generate.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    generate.add_argument("--max-retries", type=int, default=2)
    generate.add_argument("--overwrite", action="store_true")

    audit = subparsers.add_parser("audit", help="Run static grounding and safety checks.")
    audit.add_argument("bundle", type=Path)
    audit.add_argument("--output", type=Path)

    probe = subparsers.add_parser("probe", help="Execute read probes and an optional explicit smoke action.")
    probe.add_argument("bundle", type=Path)
    probe.add_argument("--action", type=Path)
    probe.add_argument("--allow-actions", action="store_true")
    probe.add_argument("--output", type=Path)
    probe.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    docker_probe = subparsers.add_parser("docker-probe", help="Run the public probe command in Docker.")
    docker_probe.add_argument("bundle", type=Path)
    docker_probe.add_argument("--image", required=True)
    docker_probe.add_argument("--network", default="")
    docker_probe.add_argument("--env", action="append", default=[], dest="env_names")
    docker_probe.add_argument("--action", type=Path)
    docker_probe.add_argument("--allow-actions", action="store_true")
    docker_probe.add_argument("--output", type=Path)

    catalog = subparsers.add_parser("catalog", help="Show the audited fifteen-application access-path catalog.")
    catalog.add_argument("--catalog", type=Path, dest="catalog_path")
    catalog.add_argument("--output", type=Path)

    qualify = subparsers.add_parser(
        "qualify", help="Assess whether a profile is eligible for assisted ASIL onboarding."
    )
    qualify.add_argument("profile", type=Path)
    qualify.add_argument("--output", type=Path)

    evidence = subparsers.add_parser("evidence-report", help="Assemble fail-closed onboarding evidence.")
    evidence.add_argument("profile", type=Path)
    evidence.add_argument("bundle", type=Path)
    evidence.add_argument("--generation-report", type=Path, required=True)
    evidence.add_argument("--host-report", type=Path, action="append", required=True, dest="host_reports")
    evidence.add_argument("--docker-report", type=Path, action="append", required=True, dest="docker_reports")
    evidence.add_argument("--output", type=Path)
    return parser


def _load_action(path: Path | None) -> Action | None:
    if path is None:
        return None
    return Action.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "generate":
            profile = load_onboarding_profile(args.profile)
            if args.provider == "deterministic":
                if args.plan_file is None:
                    raise ValueError("--plan-file is required with --provider deterministic.")
                plan = InterfacePlan.model_validate_json(args.plan_file.read_text(encoding="utf-8"))
                provider = DeterministicSoftwareGenProvider(plan, model=args.model)
            else:
                provider = OpenAISoftwareGenProvider(
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    max_retries=args.max_retries,
                    trace_dir=args.output / "api_traces",
                )
            result = generate_extension(profile, provider)
            report = write_extension_bundle(result, args.output, overwrite=args.overwrite)
            _emit(report)
            return 0 if report["ok"] else 2

        if args.command == "audit":
            report = audit_bundle(load_extension_bundle(args.bundle))
            payload = report.model_dump(mode="json")
            _emit(payload, args.output)
            return 0 if report.ok else 2

        if args.command == "probe":
            report = probe_extension(
                load_extension_bundle(args.bundle),
                action=_load_action(args.action),
                allow_actions=args.allow_actions,
            )
            _emit(report, args.output)
            return 0

        if args.command == "docker-probe":
            report = docker_probe_extension(
                args.bundle,
                image=args.image,
                network=args.network,
                env_names=args.env_names,
                action_path=args.action,
                allow_actions=args.allow_actions,
            )
            _emit(report, args.output)
            return 0
        if args.command == "catalog":
            catalog = load_reference_catalog(args.catalog_path)
            _emit(catalog.model_dump(mode="json"), args.output)
            return 0
        if args.command == "qualify":
            report = qualify_profile(load_onboarding_profile(args.profile))
            _emit(report.model_dump(mode="json"), args.output)
            return 0 if report.eligible else 2
        if args.command == "evidence-report":
            generation_report = json.loads(args.generation_report.read_text(encoding="utf-8"))
            host_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.host_reports]
            docker_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.docker_reports]
            report = build_deployment_evidence_report(
                profile=load_onboarding_profile(args.profile),
                bundle=load_extension_bundle(args.bundle),
                generation_report=generation_report,
                artifact_dir=args.generation_report.parent,
                host_reports=host_reports,
                docker_reports=docker_reports,
            )
            _emit(report.model_dump(mode="json"), args.output)
            return 0 if report.ready else 2
        raise ValueError(f"Unsupported softwaregen command: {args.command}")
    except Exception as exc:
        error_output = getattr(args, "output", None)
        if args.command == "generate" and error_output is not None:
            error_output = error_output / "generation_error.json"
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, error_output)
        return 2


if __name__ == "__main__":
    sys.exit(main())
