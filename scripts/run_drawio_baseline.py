#!/usr/bin/env python3
"""draw.io MCP native-interface baseline (#3, reviewer WgDr).

Mirrors the LibreOffice-UNO baseline but for draw.io: the model drives draw.io
through its *own* agent-oriented interface — the draw.io MCP representation, which
is mxGraph XML (draw.io's native diagram format, exactly what the official
draw.io MCP server consumes/produces). Same drawio tasks, same ASIL evaluator,
same model, same step budget as the ASIL runs, so the only variable is the
interface (ASIL's normalized JSON observation + semantic diagram actions vs.
draw.io's raw mxGraph XML contract).

Runs headless in API mode (no DRAWIO_URL / no browser): each turn the model sees
the current diagram as mxGraph XML and returns the complete updated mxGraph XML;
the harness round-trips it through the adapter (XML -> state) so adapter.observe()
+ the SAME evaluator score the result exactly as for ASIL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from asil.benchmark import _create_adapter
from asil.eval.runner import TaskDefinition
from asil.eval.evaluator import evaluate_task_result
from asil.eval.metrics import TaskResult, StepResult, compute_metrics

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_uno_baseline import call_model  # dual-protocol (gpt chat / claude native) gateway call

SYSTEM_PROMPT = """You control draw.io (diagrams.net) through its MCP interface. draw.io's \
native representation of a diagram is mxGraph XML (the format the draw.io MCP server \
consumes and produces). You do NOT use a GUI, mouse, or screenshots.

Each turn you receive the CURRENT diagram as mxGraph XML and the result of your last \
edit. You then return the COMPLETE updated mxGraph XML for the whole diagram, or the \
single token FINISH if the task is already satisfied.

mxGraph XML structure:
  <mxfile><diagram id="..." name="Page-1"><mxGraphModel ...><root>
    <mxCell id="0"/><mxCell id="1" parent="0"/>
    <mxCell id="review" value="Review" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="300" y="140" width="150" height="80" as="geometry"/></mxCell>
    <mxCell id="flow_1" value="submit" style="edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1" source="start" target="review">
      <mxGeometry relative="1" as="geometry"/></mxCell>
  </root></mxGraphModel></diagram></mxfile>

Rules:
- Each shape/connector is an <mxCell>. A shape has vertex="1"; a connector has edge="1"
  with source/target referencing shape ids. The cell's `value` is its visible label.
- The mxCell `id` is the element's stable identity. PRESERVE existing ids. Give each NEW
  cell a concise, human-meaningful id derived from its label (lowercase, like the existing
  ids), e.g. a "Deploy" step -> id "deploy", a new edge -> id "flow_2".
- Keep the surrounding <mxfile>/<diagram>/<mxGraphModel>/<root> wrapper and cells 0 and 1.

Respond with a brief Thought, then either one ```xml ...``` block (the full diagram) or FINISH."""

USER_TEMPLATE = """Task: {instruction}

Current diagram (mxGraph XML):
{xml}

{feedback}Return the complete updated mxGraph XML, or FINISH if the task is already done."""

XML_RE = re.compile(r"```(?:xml)?\s*(.*?)```", re.DOTALL)
MXFILE_RE = re.compile(r"(<mxfile[\s\S]*?</mxfile>)", re.IGNORECASE)


def extract_xml(text: str) -> tuple[str, bool]:
    if re.search(r"\bFINISH\b", text or "") and "<mxfile" not in (text or "") and "```" not in (text or ""):
        return "", True
    m = XML_RE.search(text or "")
    if m:
        inner = m.group(1).strip()
        mm = MXFILE_RE.search(inner)
        return (mm.group(1) if mm else inner), False
    mm = MXFILE_RE.search(text or "")
    if mm:
        return mm.group(1), False
    if re.search(r"\bFINISH\b", text or ""):
        return "", True
    return "", False


def current_xml(adapter) -> str:
    try:
        return adapter._state_to_drawio_xml(adapter._read_state())
    except Exception:
        try:
            return Path(str(adapter.source_path).replace(".json", "")).read_text()
        except Exception:
            return "<mxfile><diagram><mxGraphModel><root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/></root></mxGraphModel></diagram></mxfile>"


def apply_xml(adapter, xml: str) -> str | None:
    """Round-trip model XML into adapter state. Returns an error string or None."""
    try:
        state = adapter._state_from_drawio_xml(xml)
    except Exception as exc:  # noqa: BLE001
        return f"XML did not parse: {exc}"
    try:
        adapter._write_state(state)
    except Exception as exc:  # noqa: BLE001
        return f"could not apply diagram: {exc}"
    return None


def run_task(task: TaskDefinition, adapter, model: str, max_steps: int) -> TaskResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps: list[StepResult] = []
    feedback = ""
    t0 = time.monotonic()
    for step_num in range(1, max_steps + 1):
        xml = current_xml(adapter)
        messages.append({"role": "user", "content": USER_TEMPLATE.format(
            instruction=task.instruction, xml=xml, feedback=feedback)})
        t_llm = time.monotonic()
        reply = call_model(messages, model)
        latency = (time.monotonic() - t_llm) * 1000.0
        messages.append({"role": "assistant", "content": reply})
        new_xml, done = extract_xml(reply)
        if done:
            steps.append(StepResult(step_num=step_num, action_type="finish", target="",
                                    success=True, latency_ms=latency, agent_response=reply))
            break
        if not new_xml:
            feedback = "Your reply had no ```xml``` diagram block and no FINISH. "
            steps.append(StepResult(step_num=step_num, action_type="noop", target="",
                                    success=False, latency_ms=latency, agent_response=reply))
            continue
        err = apply_xml(adapter, new_xml)
        if err:
            feedback = f"Previous diagram FAILED to apply: {err}\n"
            ok = False
        else:
            n = len(adapter.observe().interactive_elements)
            feedback = f"Diagram applied ({n} elements).\n"
            ok = True
        steps.append(StepResult(step_num=step_num, action_type="mxgraph_xml", target="",
                                success=ok, latency_ms=latency, agent_response=reply))
    obs = adapter.observe()
    report = evaluate_task_result(task, obs)
    return TaskResult(
        task_id=task.id, software=task.software, difficulty=task.difficulty,
        instruction=task.instruction, success=report.success, score=report.score,
        steps=len(steps), step_results=steps, e2e_time_s=time.monotonic() - t0,
        observation_element_count=len(obs.interactive_elements),
        total_element_count=len(obs.interactive_elements))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-set", default="test_full15_multi_apps_380.json")
    ap.add_argument("--test-config-base-dir", default="evaluation_examples")
    ap.add_argument("--task-id-filter", nargs="*", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.test_config_base_dir) / args.task_set
    id_filter = set()
    for x in (args.task_id_filter or []):
        id_filter.update(p for p in re.split(r"[,\s]+", x) if p)
    out_root = Path(args.output_dir)

    tasks = TaskDefinition.from_index(index_path, domain="drawio")
    if id_filter:
        tasks = [t for t in tasks if t.id in id_filter]

    results: list[TaskResult] = []
    tmp = out_root / "_work"
    tmp.mkdir(parents=True, exist_ok=True)
    # DRAWIO_URL must be unset so the adapter runs in API/file mode (no browser).
    os.environ.pop("DRAWIO_URL", None)
    adapter = _create_adapter("drawio", tmp)
    init_bytes = Path(adapter.source_path).read_bytes()

    for task in tasks:
        task_dir = out_root / "drawio" / task.id
        if args.resume and (task_dir / "result.txt").exists():
            score = float((task_dir / "result.txt").read_text().strip() or "0")
            results.append(TaskResult(task_id=task.id, software="drawio", difficulty=task.difficulty,
                                      instruction=task.instruction, success=score >= 1.0, score=score))
            print(f"[skip] {task.id} (resumed)", flush=True)
            continue
        Path(adapter.source_path).write_bytes(init_bytes)  # reset to initial diagram
        try:
            tr = run_task(task, adapter, args.model, args.max_steps)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {task.id}: {exc}", flush=True)
            tr = TaskResult(task_id=task.id, software="drawio", difficulty=task.difficulty,
                            instruction=task.instruction, success=False, score=0.0)
        tr.save_result(task_dir)
        tr.save_trajectory(task_dir)
        results.append(tr)
        print(f"[done] {task.id} score={tr.score} steps={tr.steps} t={tr.e2e_time_s:.1f}s", flush=True)

    metrics = compute_metrics(results)
    out_path = Path(args.output) if args.output else (out_root / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    agg = metrics.get("aggregate", {})
    print(f"[drawio-mcp] tasks={agg.get('total_tasks')} pass={agg.get('passed')} "
          f"success_rate={agg.get('success_rate')}", flush=True)


if __name__ == "__main__":
    main()
