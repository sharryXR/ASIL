#!/usr/bin/env python
"""Gate experiment results using visual-delta and latency audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visual_audit_json", type=Path, help="Path to visual_audit.json produced by audit_visual_deltas.py")
    parser.add_argument("--max-identical-per-software", type=int, default=0, help="Fail if any software exceeds this many all_identical tasks")
    parser.add_argument("--max-gap-seconds", type=float, default=120.0, help="Warn/fail threshold for render->next action gaps")
    parser.add_argument("--hard-gap-seconds", type=float, default=180.0, help="Hard failure threshold for render->next action gaps")
    parser.add_argument("--max-cropped-per-software", type=int, default=0, help="Fail if any software exceeds this many cropped_or_letterboxed tasks")
    args = parser.parse_args()

    payload = json.loads(args.visual_audit_json.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []

    for software, summary in sorted(payload.get("per_software", {}).items()):
        identical = int(summary.get("all_identical", 0))
        if identical > args.max_identical_per_software:
            failures.append(
                f"{software}: all_identical={identical} exceeds allowed {args.max_identical_per_software}"
            )
        max_gap = float(summary.get("max_render_to_action_gap_s", 0.0) or 0.0)
        if max_gap > args.hard_gap_seconds:
            failures.append(
                f"{software}: max_render_to_action_gap_s={max_gap:.3f}s exceeds hard limit {args.hard_gap_seconds:.1f}s"
            )
        elif max_gap > args.max_gap_seconds:
            warnings.append(
                f"{software}: max_render_to_action_gap_s={max_gap:.3f}s exceeds warning threshold {args.max_gap_seconds:.1f}s"
            )
        actual_page = int(summary.get("actual_page_all_true", 0))
        total = int(summary.get("total_tasks", 0))
        if actual_page != total:
            failures.append(f"{software}: only {actual_page}/{total} tasks have actual_page=true across step renders")
        capture_complete = int(summary.get("capture_complete_all_true", total))
        if capture_complete != total:
            failures.append(
                f"{software}: only {capture_complete}/{total} tasks have capture_complete=true across step renders"
            )
        cropped = int(summary.get("cropped_or_letterboxed", 0))
        if cropped > args.max_cropped_per_software:
            failures.append(
                f"{software}: cropped_or_letterboxed={cropped} exceeds allowed {args.max_cropped_per_software}"
            )

    print(f"Evaluated visual audit: {args.visual_audit_json}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nVisual preflight gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
