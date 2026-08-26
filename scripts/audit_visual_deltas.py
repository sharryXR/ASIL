"""Audit step-level screenshot deltas and timing gaps for experiment results."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class TaskVisualAudit:
    software: str
    task_id: str
    screenshot_count: int
    unique_png_count: int
    all_identical: bool
    changed_once: bool
    rich_change: bool
    actual_page_all_true: bool
    capture_complete_all_true: bool
    edge_black_ratio: float
    interior_black_ratio: float
    cropped_or_letterboxed: bool
    max_render_to_action_gap_s: float
    max_step_total_latency_ms: float


def _png_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _step_number(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Unable to determine step number from {path.name}")
    return int(match.group(1))


def _edge_black_metrics(path: Path) -> dict[str, float | bool]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            return {
                "edge_black_ratio": 0.0,
                "interior_black_ratio": 0.0,
                "cropped_or_letterboxed": False,
            }

        right_strip = max(width // 8, 24)
        bottom_strip = max(height // 8, 24)
        interior_margin_x = max(width // 4, 1)
        interior_margin_y = max(height // 4, 1)

        def _black_ratio(box: tuple[int, int, int, int]) -> float:
            region = rgb.crop(box)
            width_px, height_px = region.size
            if width_px <= 0 or height_px <= 0:
                return 0.0
            pixels = region.load()
            black = 0
            for y in range(height_px):
                for x in range(width_px):
                    if all(channel <= 8 for channel in pixels[x, y]):
                        black += 1
            return black / float(width_px * height_px)

        right_ratio = _black_ratio((max(width - right_strip, 0), 0, width, height))
        bottom_ratio = _black_ratio((0, max(height - bottom_strip, 0), width, height))
        if interior_margin_x * 2 >= width or interior_margin_y * 2 >= height:
            interior_ratio = 0.0
        else:
            interior_ratio = _black_ratio(
                (interior_margin_x, interior_margin_y, width - interior_margin_x, height - interior_margin_y)
            )
        edge_ratio = max(right_ratio, bottom_ratio)
        cropped = edge_ratio >= 0.85 and interior_ratio <= 0.35
        return {
            "edge_black_ratio": round(edge_ratio, 4),
            "interior_black_ratio": round(interior_ratio, 4),
            "cropped_or_letterboxed": cropped,
        }


def _audit_task(task_dir: Path, software: str) -> TaskVisualAudit:
    screenshots = sorted(task_dir.glob("step_*.png"), key=_step_number)
    render_metas = sorted(task_dir.glob("step_*.render.json"), key=_step_number)
    hashes = [_png_hash(path) for path in screenshots]
    unique_png_count = len(set(hashes))
    actual_page_all_true = bool(render_metas)
    capture_complete_all_true = bool(render_metas)
    max_step_total_latency_ms = 0.0
    max_render_to_action_gap_s = 0.0
    edge_black_ratio = 0.0
    interior_black_ratio = 0.0
    cropped_or_letterboxed = False

    for screenshot in screenshots:
        metrics = _edge_black_metrics(screenshot)
        edge_black_ratio = max(edge_black_ratio, float(metrics["edge_black_ratio"]))
        interior_black_ratio = max(interior_black_ratio, float(metrics["interior_black_ratio"]))
        cropped_or_letterboxed = cropped_or_letterboxed or bool(metrics["cropped_or_letterboxed"])

    for render_meta in render_metas:
        payload = json.loads(render_meta.read_text(encoding="utf-8"))
        actual_page_all_true = actual_page_all_true and bool(payload.get("actual_page"))
        capture_complete_all_true = capture_complete_all_true and bool(payload.get("capture_complete", True))

    action_files = sorted(task_dir.glob("step_*_action.json"), key=_step_number)
    for action_file in action_files:
        payload = json.loads(action_file.read_text(encoding="utf-8"))
        max_step_total_latency_ms = max(max_step_total_latency_ms, float(payload.get("step_total_latency_ms", 0.0) or 0.0))
        step_num = _step_number(action_file)
        if step_num <= 1:
            continue
        prev_render = task_dir / f"step_{step_num - 1}.render.json"
        if prev_render.exists():
            gap = action_file.stat().st_mtime - prev_render.stat().st_mtime
            max_render_to_action_gap_s = max(max_render_to_action_gap_s, gap)

    return TaskVisualAudit(
        software=software,
        task_id=task_dir.name,
        screenshot_count=len(screenshots),
        unique_png_count=unique_png_count,
        all_identical=bool(screenshots) and unique_png_count == 1,
        changed_once=unique_png_count == 2,
        rich_change=unique_png_count >= 3,
        actual_page_all_true=actual_page_all_true,
        capture_complete_all_true=capture_complete_all_true,
        edge_black_ratio=round(edge_black_ratio, 4),
        interior_black_ratio=round(interior_black_ratio, 4),
        cropped_or_letterboxed=cropped_or_letterboxed,
        max_render_to_action_gap_s=round(max_render_to_action_gap_s, 3),
        max_step_total_latency_ms=round(max_step_total_latency_ms, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GUI screenshot deltas for an experiment result tree.")
    parser.add_argument("result_root", type=Path, help="Path to asil_protocol/structured_json/<model> root")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    audits: list[TaskVisualAudit] = []
    software_summary: dict[str, dict[str, Any]] = {}

    for software_dir in sorted(path for path in args.result_root.iterdir() if path.is_dir() and path.name != "summary"):
        task_dirs = sorted(path for path in software_dir.iterdir() if path.is_dir())
        per_software = []
        for task_dir in task_dirs:
            audit = _audit_task(task_dir, software_dir.name)
            audits.append(audit)
            per_software.append(audit)
        software_summary[software_dir.name] = {
            "total_tasks": len(per_software),
            "all_identical": sum(1 for item in per_software if item.all_identical),
            "changed_once": sum(1 for item in per_software if item.changed_once),
            "rich_change": sum(1 for item in per_software if item.rich_change),
            "actual_page_all_true": sum(1 for item in per_software if item.actual_page_all_true),
            "capture_complete_all_true": sum(1 for item in per_software if item.capture_complete_all_true),
            "cropped_or_letterboxed": sum(1 for item in per_software if item.cropped_or_letterboxed),
            "max_edge_black_ratio": max((item.edge_black_ratio for item in per_software), default=0.0),
            "max_render_to_action_gap_s": max((item.max_render_to_action_gap_s for item in per_software), default=0.0),
            "max_step_total_latency_ms": max((item.max_step_total_latency_ms for item in per_software), default=0.0),
        }

    payload = {
        "result_root": str(args.result_root),
        "per_software": software_summary,
        "tasks": [asdict(item) for item in audits],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
