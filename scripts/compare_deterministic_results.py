#!/usr/bin/env python3
"""Compare deterministic task-level results between two benchmark directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare deterministic results against an official baseline.")
    parser.add_argument("--baseline-root", type=Path, required=True, help="Official benchmark result root.")
    parser.add_argument("--candidate-root", type=Path, required=True, help="Candidate benchmark result root.")
    parser.add_argument("--output", type=Path, required=True, help="Comparison report JSON path.")
    return parser


def _results_path(root: Path) -> Path:
    return root / "asil_protocol" / "deterministic" / "ground_truth" / "summary" / "results.json"


def _load_results(root: Path) -> list[dict]:
    summary_path = _results_path(root)
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload

    flat_path = root / "results.json"
    payload = json.loads(flat_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows: list[dict] = []
        for software, software_payload in payload.items():
            for task in software_payload.get("tasks", []):
                rows.append(
                    {
                        "software": software,
                        "task_id": task["task_id"],
                        "score": task.get("score"),
                        "success": bool(task.get("success")),
                    }
                )
        return rows
    raise ValueError(f"Unsupported results payload shape under {root}")


def main() -> int:
    args = build_parser().parse_args()
    baseline = _load_results(args.baseline_root)
    candidate = _load_results(args.candidate_root)

    baseline_by_key = {(row["software"], row["task_id"]): row for row in baseline}
    candidate_by_key = {(row["software"], row["task_id"]): row for row in candidate}

    drifts = []
    for key, base_row in baseline_by_key.items():
        cand_row = candidate_by_key.get(key)
        if cand_row is None:
            drifts.append(
                {
                    "software": key[0],
                    "task_id": key[1],
                    "issue": "missing_in_candidate",
                }
            )
            continue
        baseline_success = base_row.get("success")
        if baseline_success is None:
            baseline_success = base_row.get("status") == "success"
        candidate_success = cand_row.get("success")
        if candidate_success is None:
            candidate_success = cand_row.get("status") == "success"
        if (
            base_row.get("score") != cand_row.get("score")
            or baseline_success != candidate_success
        ):
            drifts.append(
                {
                    "software": key[0],
                    "task_id": key[1],
                    "issue": "score_or_success_changed",
                    "baseline": {"score": base_row.get("score"), "success": baseline_success},
                    "candidate": {"score": cand_row.get("score"), "success": candidate_success},
                }
            )

    payload = {
        "baseline_root": str(args.baseline_root),
        "candidate_root": str(args.candidate_root),
        "total_tasks": len(baseline_by_key),
        "drift_count": len(drifts),
        "drifts": drifts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Deterministic comparison report saved to {args.output}")
    print(f"drift_count={len(drifts)} / total={len(baseline_by_key)}")
    return 0 if not drifts else 1


if __name__ == "__main__":
    raise SystemExit(main())
