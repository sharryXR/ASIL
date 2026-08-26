#!/usr/bin/env python
"""Audit ASIL task definitions for migrated evaluator and GUI visibility rules."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from asil.eval.task_audit import audit_task_tree


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("evaluation_examples/examples"),
        help="Task root directory or single task JSON file.",
    )
    args = parser.parse_args()

    reports = audit_task_tree(args.root)
    total = len(reports)
    passed = sum(report.ok for report in reports)
    failed = total - passed

    per_software = Counter()
    for report in reports:
        if report.path and len(report.path.parts) >= 2:
            per_software[report.path.parent.name] += 0 if report.ok else 1

    print(f"Audited {total} task files")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if per_software:
        print("\nPer-software failures:")
        for software, count in sorted(per_software.items()):
            print(f"  {software}: {count}")

    if failed:
        print("\nFailures:")
        for report in reports:
            if report.ok:
                continue
            location = str(report.path) if report.path else report.task_id
            print(f"- {location}")
            for error in report.errors:
                print(f"    ERROR: {error}")
            for warning in report.warnings:
                print(f"    WARN: {warning}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
