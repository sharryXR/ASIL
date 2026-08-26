#!/usr/bin/env python3
"""Fail closed when an ASIL checkout contains non-public release material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable


REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "DATA_LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
    "evaluation_examples/test_full15.json",
    "evaluation_examples/test_multi_apps_80.json",
    "evaluation_examples/test_full15_multi_apps_380.json",
    "evaluation_examples/test_full15_realwork_hard.json",
    "evaluation_examples/test_easy60.json",
)
REQUIRED_DIRECTORIES = (
    "src/asil",
    "scripts",
    "tests",
    "evaluation_examples/examples",
)
ALLOWED_RELEASE_ARTIFACTS = {
    "docs/ASIL_EMNLP_2026_Findings.pdf",
}
FORBIDDEN_DIRECTORY_NAMES = {
    "build",
    "dist",
    "paper",
    "rebuttal",
    "results",
    "rl",
    "sft",
    "sftgen",
    "taskgen",
}
LOCAL_STATE_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "results",
}
FORBIDDEN_PATH_PREFIXES = (
    "configs/rl/",
    "configs/sft/",
    "docs/superpowers/",
)
FORBIDDEN_SUFFIXES = (
    ".aux",
    ".bin",
    ".ckpt",
    ".log",
    ".out",
    ".pdf",
    ".pt",
    ".pth",
    ".safetensors",
    ".sif",
    ".sif.gz",
    ".tar.gz",
    ".zip",
)
TEXT_SCAN_EXEMPTIONS = {
    "scripts/validate_public_release.py",
    "tests/test_public_release.py",
    "tests/test_rebuttal_docker_verifier.py",
}
FORBIDDEN_TEXT = (
    ("wrong contributor login", re.compile(r"\bshzirui\b", re.IGNORECASE)),
    ("wrong contributor email", re.compile(r"609146431@qq\.com", re.IGNORECASE)),
    ("local macOS path", re.compile(r"/Users/[^\s)\]}`'\"]+")),
    ("internal public path", re.compile(r"/public/(?:home|LLM_model_dataset)/")),
    ("GitHub token", re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}")),
    ("OpenAI-shaped key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("release placeholder", re.compile(r"\b(?:REPLACE_ME|TBD_ZENODO|ZENODO_PLACEHOLDER)\b")),
)
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _finding(code: str, path: str, detail: str) -> dict[str, str]:
    return {"code": code, "path": path, "detail": detail}


def _release_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if set(relative.parts) & LOCAL_STATE_DIRECTORY_NAMES:
            continue
        if path.is_file() or path.is_symlink():
            yield path


def _validate_inventory(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            findings.append(_finding("missing_required", relative, "required file is absent"))
    for relative in REQUIRED_DIRECTORIES:
        if not (root / relative).is_dir():
            findings.append(_finding("missing_required", relative, "required directory is absent"))
    return findings


def _validate_paths(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in _release_files(root):
        relative = path.relative_to(root).as_posix()
        if (
            relative in ALLOWED_RELEASE_ARTIFACTS
            and path.is_file()
            and not path.is_symlink()
        ):
            continue
        parts = set(path.relative_to(root).parts)
        lower = relative.lower()
        if path.is_symlink():
            findings.append(_finding("forbidden_path", relative, "symbolic links are not released"))
        if parts & FORBIDDEN_DIRECTORY_NAMES or any(
            relative.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES
        ):
            findings.append(_finding("forbidden_path", relative, "path belongs to an excluded tree"))
        if lower == ".env" or lower.endswith(FORBIDDEN_SUFFIXES):
            findings.append(_finding("forbidden_path", relative, "file type is excluded from GitHub"))
    return findings


def _validate_text(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in _release_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in TEXT_SCAN_EXEMPTIONS or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in FORBIDDEN_TEXT:
            if pattern.search(text):
                findings.append(_finding("forbidden_text", relative, label))
    return findings


def _validate_task_indexes(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    index_root = root / "evaluation_examples"
    if not index_root.is_dir():
        return findings
    for index_path in sorted(index_root.glob("test_*.json")):
        relative = index_path.relative_to(root).as_posix()
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(_finding("invalid_index", relative, type(exc).__name__))
            continue
        if not isinstance(payload, dict):
            findings.append(_finding("invalid_index", relative, "index must be a JSON object"))
            continue
        for software, task_ids in payload.items():
            if not isinstance(software, str) or not isinstance(task_ids, list):
                findings.append(_finding("invalid_index", relative, "software entries must contain lists"))
                continue
            for task_id in task_ids:
                if not isinstance(task_id, str):
                    findings.append(_finding("invalid_index", relative, "task IDs must be strings"))
                    continue
                task_path = index_root / "examples" / software / f"{task_id}.json"
                if not task_path.is_file():
                    findings.append(
                        _finding(
                            "missing_task",
                            relative,
                            f"{software}/{task_id}.json is not present",
                        )
                    )
    return findings


def validate_public_release(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    findings = [
        *_validate_inventory(resolved),
        *_validate_paths(resolved),
        *_validate_text(resolved),
        *_validate_task_indexes(resolved),
    ]
    findings.sort(key=lambda item: (item["path"], item["code"], item["detail"]))
    return {"ok": not findings, "root": str(resolved), "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = validate_public_release(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
