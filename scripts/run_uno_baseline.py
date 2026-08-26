#!/usr/bin/env python3
"""Matched native-interface baseline (#3, reviewer WgDr): drive LibreOffice through
its *own* native programmatic API (UNO) with the same model, same tasks, same step
budget, and the SAME ASIL evaluator used for the ASIL runs — so the only variable is
the interface (ASIL's curated semantic-action contract + normalized observation vs.
LibreOffice's raw native UNO API + native state observation).

This is deliberately a strong baseline: UNO exposes the app's full automation surface
(any edit the GUI can do), so if ASIL still wins it is because the curated contract +
normalized observation make the model *more reliable* than raw API power — which is
exactly the claim under test.

Architecture (see uno_worker.py for the other half):
  py3.11 orchestrator (this file)          py3.10 + python3-uno worker (subprocess)
  - imports the ASIL package               - holds the soffice UNO connection
  - loads tasks / adapters / evaluator     - holds one live document
  - calls the model (gateway chat)         - observe() / exec(model code) / save()
  - scores final ODF via adapter.observe() ── JSON lines over stdin/stdout ──

No compose services are needed: LibreOffice tasks are pure file manipulation, so this
runs under a plain `docker run asil-eval:uno ...`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ASIL package (py3.11 venv) — reuse the *exact* task defs, adapters, evaluator, metrics.
from asil.benchmark import _create_adapter
from asil.eval.runner import TaskDefinition
from asil.eval.evaluator import evaluate_task_result
from asil.eval.metrics import TaskResult, StepResult, compute_metrics

LO_SOFTWARES = ["libreoffice", "libreoffice_writer", "libreoffice_impress"]

SYSTEM_PROMPT = """You operate LibreOffice through its native UNO scripting API — the \
application's own programmatic automation interface (the same API macros use). You are \
NOT using a GUI; there are no screenshots and no mouse. You write Python code that runs \
against the live document.

Each turn you receive the current document state (serialized from the live document) and \
the result of your previous code. You then either:
  1. Emit exactly one Python code block to run against the document, or
  2. Emit the single token FINISH when the task is complete.

Environment available to your code (already bound, do not re-import or re-connect):
  document      the live UNO document component (also aliased ThisComponent)
  uno           the uno module
  desktop, smgr, ctx   UNO desktop / service manager / component context

The document is AUTOMATICALLY SAVED after each of your code blocks — do not call store*.
Use print(...) to inspect values; printed output is returned to you next turn.

UNO quick reference:
  Calc:   sheet = document.Sheets.getByIndex(0)          # or .getByName("Sheet1")
          cell  = sheet.getCellByPosition(col, row)       # 0-based; A1 -> (0,0)
          cell.setString("text")  /  cell.setValue(3.14)  /  cell.setFormula("=A1+A2")
  Writer: text = document.Text; cur = text.createTextCursor()
          text.insertString(cur, "hello", False)
          # enumerate paragraphs via document.Text.createEnumeration()
  Impress:to set a slide's TITLE or BODY you must write into the presentation
          PLACEHOLDER shapes (not a plain text box) or the change will not register.
          If a slide has no title/content placeholder yet, first set its autolayout:
            slide = document.DrawPages.getByIndex(0)
            slide.Layout = 1                      # instantiates Title + Content placeholders
          Then locate placeholders by service and set their text:
            for i in range(slide.Count):
                sh = slide.getByIndex(i)
                if sh.supportsService('com.sun.star.presentation.TitleTextShape'): sh.setString('My Title')
                elif sh.supportsService('com.sun.star.presentation.OutlineTextShape') or \\
                     sh.supportsService('com.sun.star.presentation.SubtitleTextShape'): sh.setString('Body')
          (Layout 1 = title+content, 2 = title+outline.)

Respond with a brief Thought, then either one ```python ...``` block or FINISH."""

USER_TEMPLATE = """Task: {instruction}

Current document state:
{state}

{feedback}Write the next Python code block to make progress, or FINISH if the task is \
already satisfied."""

CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Model (gateway chat) — same key/endpoint as the ASIL/GUI runs
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def call_model(messages: list[dict], model: str) -> str:
    base = (os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    if not base or not key:
        raise RuntimeError("gateway base/key not set")
    url = f"{base}/chat/completions"
    effort = os.environ.get("ASIL_GUI_REASONING_EFFORT", "high") or "high"
    timeout = float(_env_int("ASIL_GUI_LLM_TIMEOUT_S", 900))
    retries = _env_int("ASIL_GUI_LLM_RETRIES", 100)
    is_claude = "claude" in model.lower()
    max_tokens = _env_int("ASIL_UNO_MAX_TOKENS", 4000)

    def build(attempt):
        if is_claude:
            # Anthropic native protocol via the DashScope compatible-mode passthrough.
            system = "\n".join(m["content"] for m in messages if m["role"] == "system")
            conv = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
            return {"model": model, "dashscope_extend_params": {"using_native_protocol": "true"},
                    "max_tokens": max_tokens, "system": system, "messages": conv}
        body = {"model": model, "messages": messages}
        if effort and attempt < 2:  # some gateway builds reject reasoning_effort on chat
            body["reasoning_effort"] = effort
        return body

    def parse(data):
        if is_claude:
            blocks = data.get("content", []) if isinstance(data, dict) else []
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        return (data["choices"][0]["message"]["content"] or "").strip()

    last = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(build(attempt)).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return parse(json.loads(r.read().decode("utf-8")))
        except Exception as exc:  # noqa: BLE001 — retry storm handling
            last = exc
            time.sleep(min(30.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"model call failed after {retries} retries: {last}")


def parse_response(text: str) -> tuple[str, bool]:
    """Return (code, done). done=True when the model signals FINISH with no code."""
    m = CODE_RE.search(text or "")
    if m:
        return m.group(1).strip(), False
    if re.search(r"\bFINISH\b", text or "", re.IGNORECASE):
        return "", True
    return "", False  # no code, no finish -> treat as a wasted step


# ---------------------------------------------------------------------------
# soffice + UNO worker lifecycle
# ---------------------------------------------------------------------------
class UnoBackend:
    def __init__(self, port: int, worker_py: str, worker_script: str, profile: str):
        self.port = port
        self.worker_py = worker_py
        self.worker_script = worker_script
        self.profile = profile
        self.soffice = None
        self.worker = None

    def _port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self):
        self.stop()
        env = dict(os.environ)
        env["HOME"] = "/tmp"
        self.soffice = subprocess.Popen(
            [
                "soffice", "--headless", "--invisible", "--nologo", "--nodefault",
                "--norestore", "--nofirststartwizard", "--nocrashreport",
                f"-env:UserInstallation=file://{self.profile}",
                f"--accept=socket,host=127.0.0.1,port={self.port};urp;StarOffice.ComponentContext",
            ],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        t0 = time.monotonic()
        while time.monotonic() - t0 < 90:
            if self._port_open():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("soffice UNO socket did not open within 90s")
        self.worker = subprocess.Popen(
            [self.worker_py, self.worker_script, str(self.port)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        ready = self._read(timeout=30)
        if not (ready and ready.get("ok")):
            raise RuntimeError(f"worker failed to start: {ready}")

    def stop(self):
        for p in (self.worker, self.soffice):
            if p is not None:
                try:
                    p.kill()
                except Exception:
                    pass
        self.worker = self.soffice = None

    def _read(self, timeout: float) -> dict | None:
        r, _, _ = select.select([self.worker.stdout], [], [], timeout)
        if not r:
            return None
        line = self.worker.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line)
        except Exception:
            return {"ok": False, "error": f"bad worker line: {line[:200]}"}

    def rpc(self, req: dict, timeout: float) -> dict:
        self.worker.stdin.write(json.dumps(req) + "\n")
        self.worker.stdin.flush()
        resp = self._read(timeout)
        if resp is None:
            raise TimeoutError(f"worker rpc timeout on {req.get('cmd')}")
        return resp


# ---------------------------------------------------------------------------
# Task loop
# ---------------------------------------------------------------------------
def run_task(task: TaskDefinition, adapter, backend: UnoBackend, model: str,
             max_steps: int, step_timeout: float) -> TaskResult:
    url = "file://" + str(Path(adapter.source_path).resolve())
    backend.rpc({"cmd": "load", "url": url}, timeout=step_timeout)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps: list[StepResult] = []
    feedback = ""
    t0 = time.monotonic()
    for step_num in range(1, max_steps + 1):
        state = backend.rpc({"cmd": "observe"}, timeout=step_timeout).get("text", "(no state)")
        user = USER_TEMPLATE.format(instruction=task.instruction, state=state, feedback=feedback)
        messages.append({"role": "user", "content": user})
        t_llm = time.monotonic()
        reply = call_model(messages, model)
        latency = (time.monotonic() - t_llm) * 1000.0
        messages.append({"role": "assistant", "content": reply})
        code, done = parse_response(reply)
        if done:
            steps.append(StepResult(step_num=step_num, action_type="finish", target="",
                                    success=True, latency_ms=latency, agent_response=reply))
            break
        if not code:
            feedback = "Your previous reply had no runnable code block and no FINISH. "
            steps.append(StepResult(step_num=step_num, action_type="noop", target="",
                                    success=False, latency_ms=latency, agent_response=reply))
            continue
        try:
            ex = backend.rpc({"cmd": "exec", "code": code, "save": True}, timeout=step_timeout)
        except TimeoutError:
            backend.start()  # soffice hung — restart and keep going from saved state
            backend.rpc({"cmd": "load", "url": url}, timeout=step_timeout)
            ex = {"ok": False, "error": "execution timed out; soffice restarted"}
        ok = bool(ex.get("ok"))
        out = (ex.get("stdout") or "").strip()
        err = (ex.get("error") or "").strip()
        if ok:
            feedback = f"Previous code ran OK.{(' Output: ' + out) if out else ''}\n"
        else:
            feedback = f"Previous code FAILED:\n{err[:1500]}\n"
        steps.append(StepResult(step_num=step_num, action_type="uno_exec", target="",
                                params={"code": code}, success=ok, latency_ms=latency,
                                agent_response=reply))
    # Final state saved on the worker; score the real ODF via the SAME evaluator.
    backend.rpc({"cmd": "save"}, timeout=step_timeout)
    obs = adapter.observe()
    report = evaluate_task_result(task, obs)
    return TaskResult(
        task_id=task.id, software=task.software, difficulty=task.difficulty,
        instruction=task.instruction, success=report.success, score=report.score,
        steps=len(steps), step_results=steps, e2e_time_s=time.monotonic() - t0,
        observation_element_count=len(obs.interactive_elements),
        total_element_count=len(obs.interactive_elements),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-set", default="test_full15_multi_apps_380.json")
    ap.add_argument("--test-config-base-dir", default="evaluation_examples")
    ap.add_argument("--task-id-filter", nargs="*", default=None)
    ap.add_argument("--software", nargs="*", default=None,
                    help="subset of %s; default all three" % LO_SOFTWARES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="openai")  # accepted for symmetry; gateway chat only
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--uno-port", type=int, default=2002)
    ap.add_argument("--uno-python", default="/usr/bin/python3.10")
    ap.add_argument("--step-timeout", type=float, default=120.0)
    args = ap.parse_args()

    index_path = Path(args.test_config_base_dir) / args.task_set
    softwares = args.software or LO_SOFTWARES
    id_filter = set()
    for x in (args.task_id_filter or []):
        id_filter.update(p for p in re.split(r"[,\s]+", x) if p)

    out_root = Path(args.output_dir)
    worker_script = str(Path(__file__).resolve().parent / "uno_worker.py")
    backend = UnoBackend(args.uno_port, args.uno_python, worker_script, "/tmp/lo_uno_profile")
    backend.start()

    all_results: list[TaskResult] = []
    try:
        for sw in softwares:
            tmp = out_root / "_work" / sw
            tmp.mkdir(parents=True, exist_ok=True)
            adapter = _create_adapter(sw, tmp)
            init_bytes = Path(adapter.source_path).read_bytes()
            tasks = TaskDefinition.from_index(index_path, domain=sw)
            if id_filter:
                tasks = [t for t in tasks if t.id in id_filter]
            for task in tasks:
                task_dir = out_root / sw / task.id
                if args.resume and (task_dir / "result.txt").exists():
                    try:
                        all_results.append(_load_prior(task_dir, task))
                    except Exception:
                        pass
                    print(f"[skip] {task.id} (resumed)", flush=True)
                    continue
                Path(adapter.source_path).write_bytes(init_bytes)  # reset to initial artifact
                try:
                    tr = run_task(task, adapter, backend, args.model, args.max_steps, args.step_timeout)
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] {task.id}: {exc}", flush=True)
                    backend.start()  # recover the backend for subsequent tasks
                    tr = TaskResult(task_id=task.id, software=sw, difficulty=task.difficulty,
                                    instruction=task.instruction, success=False, score=0.0)
                tr.save_result(task_dir)
                tr.save_trajectory(task_dir)
                all_results.append(tr)
                print(f"[done] {task.id} score={tr.score} steps={tr.steps} "
                      f"t={tr.e2e_time_s:.1f}s", flush=True)
    finally:
        backend.stop()

    metrics = compute_metrics(all_results)
    out_path = Path(args.output) if args.output else (out_root / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    agg = metrics.get("aggregate", {})
    print(f"[uno-baseline] tasks={agg.get('total_tasks')} pass={agg.get('passed')} "
          f"success_rate={agg.get('success_rate')}", flush=True)


def _load_prior(task_dir: Path, task: TaskDefinition) -> TaskResult:
    score = float((task_dir / "result.txt").read_text().strip() or "0")
    return TaskResult(task_id=task.id, software=task.software, difficulty=task.difficulty,
                      instruction=task.instruction, success=score >= 1.0, score=score)


if __name__ == "__main__":
    main()
