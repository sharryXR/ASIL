#!/usr/bin/env python3
"""Remove Docker resources for managed benchmark projects by prefix."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _list_names(kind: str) -> list[str]:
    command = ["docker", kind, "ls", "--format", "{{.Name}}"]
    if kind == "ps":
        command = ["docker", "ps", "-a", "--format", "{{.Names}}"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _remove_many(command: list[str], names: list[str]) -> None:
    if not names:
        return
    subprocess.run([*command, *names], check=False, capture_output=True)


def cleanup_project_prefix(prefix: str) -> None:
    containers = [name for name in _list_names("ps") if name.startswith(prefix)]
    networks = [name for name in _list_names("network") if name.startswith(prefix)]
    volumes = [name for name in _list_names("volume") if name.startswith(prefix)]

    _remove_many(["docker", "rm", "-f"], containers)
    _remove_many(["docker", "network", "rm"], networks)
    _remove_many(["docker", "volume", "rm"], volumes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-prefix", required=True, help="Docker compose project prefix to clean up.")
    args = parser.parse_args(argv)
    cleanup_project_prefix(args.project_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
