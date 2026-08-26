from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_public_release import validate_public_release


PUBLIC_INDEXES = (
    "test_full15.json",
    "test_multi_apps_80.json",
    "test_full15_multi_apps_380.json",
    "test_full15_realwork_hard.json",
    "test_easy60.json",
)


def _write_minimal_public_tree(root: Path) -> None:
    for filename in (
        "README.md",
        "LICENSE",
        "DATA_LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "pyproject.toml",
    ):
        (root / filename).write_text(f"public {filename}\n", encoding="utf-8")

    for directory in ("src/asil", "scripts", "tests"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    task_dir = root / "evaluation_examples/examples/demo"
    task_dir.mkdir(parents=True)
    (task_dir / "demo_01.json").write_text(
        json.dumps({"id": "demo_01", "software": "demo"}),
        encoding="utf-8",
    )
    for filename in PUBLIC_INDEXES:
        (root / "evaluation_examples" / filename).write_text(
            json.dumps({"demo": ["demo_01"]}),
            encoding="utf-8",
        )


def _codes(report: dict[str, object]) -> set[str]:
    return {str(finding["code"]) for finding in report["findings"]}


def test_rejects_training_tree(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    path = tmp_path / "src/asil/sft/trainer.py"
    path.parent.mkdir(parents=True)
    path.write_text("pass\n", encoding="utf-8")

    report = validate_public_release(tmp_path)

    assert report["ok"] is False
    assert "forbidden_path" in _codes(report)


def test_rejects_internal_path(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    path = tmp_path / "rebuttal/private_notes.md"
    path.parent.mkdir(parents=True)
    path.write_text("internal\n", encoding="utf-8")

    report = validate_public_release(tmp_path)

    assert report["ok"] is False
    assert "forbidden_path" in _codes(report)


def test_accepts_versioned_public_paper_pdf(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    path = tmp_path / "docs/ASIL_EMNLP_2026_Findings.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF-1.7\n")

    report = validate_public_release(tmp_path)

    assert report["ok"] is True


def test_rejects_unapproved_pdf(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    path = tmp_path / "docs/internal_results.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF-1.7\n")

    report = validate_public_release(tmp_path)

    assert report["ok"] is False
    assert "forbidden_path" in _codes(report)


def test_rejects_wrong_contributor_marker(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    (tmp_path / "README.md").write_text(
        "historical author: shzirui <609146431@qq.com>\n",
        encoding="utf-8",
    )

    report = validate_public_release(tmp_path)

    assert report["ok"] is False
    assert "forbidden_text" in _codes(report)


def test_rejects_missing_task_reference(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    (tmp_path / "evaluation_examples/test_easy60.json").write_text(
        json.dumps({"demo": ["demo_missing"]}),
        encoding="utf-8",
    )

    report = validate_public_release(tmp_path)

    assert report["ok"] is False
    assert "missing_task" in _codes(report)


def test_accepts_minimal_public_tree(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)

    report = validate_public_release(tmp_path)

    assert report == {"ok": True, "root": str(tmp_path.resolve()), "findings": []}


def test_ignores_local_generated_state(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    for relative in (
        ".pytest_cache/v/cache/nodeids",
        ".venv/lib/python/site.py",
        "results/smoke/results.json",
        "src/asil/__pycache__/module.pyc",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")

    report = validate_public_release(tmp_path)

    assert report["ok"] is True


def test_validator_source_does_not_match_its_pattern_definitions(tmp_path: Path) -> None:
    _write_minimal_public_tree(tmp_path)
    source = Path(validate_public_release.__code__.co_filename)
    (tmp_path / "scripts/validate_public_release.py").write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = validate_public_release(tmp_path)

    assert report["ok"] is True


def test_repository_readme_has_publication_links_and_citation() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    required_fragments = (
        "ASIL: Replacing Screenshot-and-Click with Structured State and Semantic Actions",
        "Findings of EMNLP 2026",
        "docs/ASIL_EMNLP_2026_Findings.pdf",
        "https://huggingface.co/datasets/sharryXR/asil-benchmark",
        "https://huggingface.co/datasets/sharryXR/asil-training-data",
        "https://huggingface.co/datasets/sharryXR/asil-benchmark-images",
        "https://huggingface.co/sharryXR/asil-qwen35-2b-sft",
        "https://huggingface.co/sharryXR/asil-qwen35-2b-rl",
        "https://huggingface.co/sharryXR/asil-qwen35-9b-sft",
        "https://huggingface.co/sharryXR/asil-qwen35-9b-rl",
        "https://huggingface.co/collections/sharryXR/asil-models-6a1e9faf39fe6ce4eb4626e1",
        "@inproceedings{xie2026asil",
    )

    missing = [fragment for fragment in required_fragments if fragment not in readme]

    assert missing == []
    assert "Models%20%26%20Data" not in readme
    for resource_label in (
        "Benchmark tasks",
        "SFT and RL training data",
        "Runtime images",
        "Model collection",
    ):
        assert resource_label in readme
    assert (root / "docs/ASIL_EMNLP_2026_Findings.pdf").is_file()
