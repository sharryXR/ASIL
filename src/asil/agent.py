"""ASIL Agent — the observe-think-act loop."""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from asil.adapter import ASILAdapter
from asil.protocol import Action, Observation

load_dotenv()

@dataclass
class AgentModelOutput:
    """Normalized provider output before action parsing."""

    text: str
    reasoning_summary: str = ""
    provider: str = ""
    model: str = ""


@dataclass
class AgentTrace:
    """Structured trace for one model decision."""

    instruction: str
    thought: str
    action: Action
    raw_text: str = ""
    reasoning_summary: str = ""
    provider: str = ""
    model: str = ""
    model_latency_ms: float = 0.0
    action_execution_latency_ms: float = 0.0


LLMFunction = Callable[[str], str | AgentModelOutput]

SYSTEM_PROMPT = """\
You are an ASIL agent that controls desktop software through structured JSON actions.
You receive a structured observation of the software state and must return ONE next-step action.

Output format:
Thought: <brief structured reasoning with current-state check, next-step plan, and target summary>
Action: <raw JSON action object>

Rules:
1. Always return both `Thought:` and `Action:` sections.
2. `Thought` must be concise and task-relevant.
3. `Action` must have exactly three keys: action_type, target, params.
4. Use ONLY the action format specified in the Action Schema section.
5. Use EXACTLY the element IDs and values specified in the task and success criteria.
6. After EACH action, check the "Current State" — if the success criteria are met, return the done action immediately.
7. Never repeat a failed action more than once — try a different approach or return done.
"""

PROMPT_VARIANT_HINTS = {
    "default": "",
    "verify_first": (
        "Before choosing an action, explicitly compare the Current State with the "
        "Success Criteria. If a criterion is already satisfied, preserve it. If all "
        "criteria are satisfied, choose done immediately."
    ),
    "state_delta": (
        "Reason as a minimal state-delta editor: identify the smallest file, field, "
        "object, cell, or setting that differs from the Success Criteria, then change "
        "only that necessary part."
    ),
    "schema_grounded": (
        "Ground every decision in the Action Schema and visible element IDs. Prefer "
        "one well-formed schema-valid action over broad edits, and copy exact IDs, "
        "paths, keys, and expected values from the prompt."
    ),
    "recovery_aware": (
        "If the Current State or Recent Errors show a failed or redundant edit, avoid "
        "repeating it. Choose a more direct schema-valid operation or stop with done "
        "when the evaluator-facing state is already correct."
    ),
}

_SCHEMA_DIR = Path(__file__).parent / "action_schemas"
_schema_cache: dict[str, dict] = {}


def available_prompt_variants() -> list[str]:
    return sorted(PROMPT_VARIANT_HINTS)


def format_prompt_variant_hint(prompt_variant: str) -> str:
    return PROMPT_VARIANT_HINTS.get(prompt_variant, "")


def load_action_schema(software: str) -> dict | None:
    """Load the action schema JSON for a given software name.

    Looks up src/asil/action_schemas/{software}.json.
    Returns the parsed dict, or None if no schema file exists.
    """
    if software in _schema_cache:
        return _schema_cache[software]

    schema_path = _SCHEMA_DIR / f"{software}.json"
    if not schema_path.exists():
        return None

    with open(schema_path) as f:
        schema = json.load(f)
    _schema_cache[software] = schema
    return schema


def format_validation_hint(validation: dict) -> str:
    """Convert an ASIL validation dict into a human-readable success criteria hint."""
    if not validation:
        return ""
    parts = []
    for key, val in validation.items():
        if key == "element_exists":
            parts.append(f"An element with id='{val}' must exist in the document.")
        elif key == "element_not_exists":
            parts.append(f"The element with id='{val}' must NOT exist in the document.")
        elif key == "element_value":
            eid = val.get("id", "?")
            k = val.get("key")
            exp = val.get("expected", "?")
            attr = f"['{k}']" if k else ""
            parts.append(f"Element '{eid}'{attr} must have value='{exp}'.")
        elif key == "element_contains":
            eid = val.get("id", "?")
            k = val.get("key")
            exp = val.get("expected", "?")
            attr = f"['{k}']" if k else ""
            parts.append(f"Element '{eid}'{attr} must contain '{exp}'.")
        elif key == "count_equals":
            parts.append(f"The document must contain exactly {val} interactive elements.")
        elif key == "count_at_least":
            parts.append(f"The document must contain at least {val} interactive elements.")
        else:
            parts.append(f"{key}: {json.dumps(val)}")
    return "\n".join(f"- {p}" for p in parts)


def _format_expected(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return repr(value)


def _format_evaluator_rule(rule: dict) -> str:
    if "element_value" in rule:
        spec = rule["element_value"]
        eid = spec.get("id", "?")
        key = spec.get("key")
        exp = _format_expected(spec.get("expected"))
        attr = f"['{key}']" if key else ""
        return f"Element '{eid}'{attr} must equal {exp}."
    if "element_contains" in rule:
        spec = rule["element_contains"]
        eid = spec.get("id", "?")
        key = spec.get("key")
        exp = _format_expected(spec.get("expected"))
        attr = f"['{key}']" if key else ""
        return f"Element '{eid}'{attr} must contain {exp}."
    if "image_size" in rule:
        spec = rule["image_size"]
        return f"Image size must be {spec.get('width', '?')}x{spec.get('height', '?')}."
    if "image_region_stat" in rule:
        spec = rule["image_region_stat"]
        bounds = []
        if "min" in spec:
            bounds.append(f">= {spec['min']}")
        if "max" in spec:
            bounds.append(f"<= {spec['max']}")
        return (
            f"Image region {spec.get('box', '?')} {spec.get('metric', 'metric')} "
            f"must be {' and '.join(bounds) or 'within the requested range'}."
        )
    if "any_element_matches" in rule:
        spec = rule["any_element_matches"]
        return f"At least one {spec.get('type', 'element')} must match {_format_expected(spec.get('value', {}))}."
    if "no_element_matches" in rule:
        spec = rule["no_element_matches"]
        return f"No {spec.get('type', 'element')} may match {_format_expected(spec.get('value', {}))}."
    if "any_element_value" in rule:
        spec = rule["any_element_value"]
        return (
            f"At least one {spec.get('type', 'element')} element key "
            f"{spec.get('key', '?')!r} must equal {_format_expected(spec.get('expected'))}."
        )
    if "any_element_contains" in rule:
        spec = rule["any_element_contains"]
        return (
            f"At least one {spec.get('type', 'element')} element key "
            f"{spec.get('key', '?')!r} must contain {_format_expected(spec.get('expected'))}."
        )
    if "count_elements_matching" in rule:
        spec = rule["count_elements_matching"]
        return f"Count of matching {spec.get('type', 'element')} elements must satisfy {_format_expected(spec)}."
    return json.dumps(rule, ensure_ascii=False, sort_keys=True)


def format_evaluator_hint(evaluator: dict, *, max_checkpoints: int = 60) -> str:
    """Convert path/checkpoint evaluator specs into concise success criteria."""
    if not evaluator or not isinstance(evaluator.get("paths"), list):
        return ""
    lines: list[str] = []
    emitted = 0
    for path in evaluator.get("paths", []):
        if not isinstance(path, dict):
            continue
        path_id = path.get("path_id") or path.get("id") or "path"
        checkpoints = path.get("checkpoints") or path.get("conditions") or []
        if not isinstance(checkpoints, list):
            continue
        lines.append(f"Path '{path_id}' succeeds when these checkpoints pass:")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            rule = checkpoint.get("rule") if "rule" in checkpoint else checkpoint
            if not isinstance(rule, dict):
                continue
            checkpoint_id = checkpoint.get("id") or checkpoint.get("checkpoint_id") or f"checkpoint_{emitted + 1}"
            required = "required" if checkpoint.get("required", True) else "optional"
            lines.append(f"- [{required}] {checkpoint_id}: {_format_evaluator_rule(rule)}")
            emitted += 1
            if emitted >= max_checkpoints:
                lines.append(f"- ... {max_checkpoints}+ checkpoints total; satisfy all remaining listed evaluator requirements.")
                return "\n".join(lines)
    return "\n".join(lines)


def format_task_success_hint(task) -> str:
    """Format all known success criteria for a task.

    Prefer path/checkpoint evaluator hints when present because generated
    OSWorld-style tasks often leave ``_asil.validation`` empty.
    """
    parts = []
    evaluator_hint = format_evaluator_hint(getattr(task, "evaluator", {}) or {})
    if evaluator_hint:
        parts.append(evaluator_hint)
    validation_hint = format_validation_hint(getattr(task, "validation", {}) or {})
    if validation_hint:
        parts.append(validation_hint)
    return "\n".join(parts)


def format_action_schema(schema: dict) -> str:
    """Format an action schema dict into a concise prompt section."""
    lines = []
    lines.append(f"# Action Schema — {schema['software']}")
    lines.append(f"{schema['description']}")
    lines.append(f"Supported action_type(s): {', '.join(schema['supported_action_types'])}")
    lines.append(f"target: {schema['target']}")
    lines.append("")

    for action in schema["actions"]:
        lines.append(f"## {action['name']}")
        lines.append(action["description"])
        lines.append(f"params format: ```json\n{json.dumps(action['params_schema'], indent=2)}\n```")
        # Show a small bounded set of examples by default. Composite schemas can
        # opt into more examples because they need to teach nested action shapes.
        example_limit = int(schema.get("example_limit", 2))
        for ex in action.get("examples", [])[:example_limit]:
            lines.append(f"Example — {ex['description']}:")
            lines.append(f"```json\n{json.dumps(ex['action'], indent=2)}\n```")
        lines.append("")

    if schema.get("done_action"):
        lines.append(f"## done")
        lines.append(f"{schema['done_action']['description']}")
        lines.append(f"```json\n{json.dumps(schema['done_action']['example'], indent=2)}\n```")
        lines.append("")

    if schema.get("tips"):
        lines.append("## Tips")
        for tip in schema["tips"]:
            lines.append(f"- {tip}")

    return "\n".join(lines)


def _split_thought_action(text: str) -> tuple[str, str]:
    """Split model output into `Thought:` and `Action:` sections."""
    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", text, re.DOTALL)
    action_match = re.search(r"Action:\s*(.+)$", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    action_text = action_match.group(1).strip() if action_match else text.strip()
    return thought, action_text


def _response_output_items(response) -> list:
    output = getattr(response, "output", None)
    if output is None and hasattr(response, "model_dump"):
        output = response.model_dump().get("output", [])
    return output or []


def _message_text(item) -> str:
    content = getattr(item, "content", None)
    if content is None and isinstance(item, dict):
        content = item.get("content", [])
    if not content:
        return ""

    parts = []
    for part in content:
        part_type = getattr(part, "type", None)
        if part_type is None and isinstance(part, dict):
            part_type = part.get("type")
        if part_type == "output_text":
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _reasoning_text(item) -> str:
    summary = getattr(item, "summary", None)
    if summary is None and isinstance(item, dict):
        summary = item.get("summary", [])
    if not summary:
        return ""
    if isinstance(summary, list):
        parts = []
        for part in summary:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text", "")
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(summary)


def _extract_openai_output(response, model_name: str) -> AgentModelOutput:
    """Extract text and reasoning summary from an OpenAI Responses API reply."""
    texts = []
    summaries = []

    for item in _response_output_items(response):
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type == "message":
            text = _message_text(item)
            if text:
                texts.append(text)
        elif item_type == "reasoning":
            summary = _reasoning_text(item)
            if summary:
                summaries.append(summary)

    return AgentModelOutput(
        text="\n".join(part for part in texts if part),
        reasoning_summary="\n".join(part for part in summaries if part),
        provider="openai",
        model=model_name,
    )
class ASILAgent:
    def __init__(
        self,
        adapter: ASILAdapter,
        llm_fn: LLMFunction,
        max_steps: int = 20,
        software: str = "",
        prompt_variant: str = "default",
        hybrid_vision: bool = False,
    ) -> None:
        self.adapter = adapter
        self.llm_fn = llm_fn
        self.max_steps = max_steps
        self.software = software
        self.prompt_variant = prompt_variant
        self.hybrid_vision = hybrid_vision
        self._action_schema_text = ""

        # Load per-software action schema
        if software:
            schema = load_action_schema(software)
            if schema:
                self._action_schema_text = format_action_schema(schema)

    def format_observation(
        self,
        obs: Observation,
        task_description: str,
        success_hint: str = "",
        prompt_variant: str | None = None,
        reference_action_hint: str = "",
    ) -> str:
        elements_summary = []
        for e in obs.interactive_elements[:50]:
            elements_summary.append({
                "id": e.id, "type": e.type, "label": e.label,
                "value": e.value, "actions": e.actions,
            })

        parts = [f"# Task\n{task_description}\n"]

        # Inject action schema if available
        if self._action_schema_text:
            parts.append(self._action_schema_text)
            parts.append("")

        # Inject success criteria derived from task validation
        if success_hint:
            parts.append(f"# Success Criteria\nThe task is complete when ALL of the following are true:\n{success_hint}")
            parts.append("Once the Current State satisfies ALL criteria, return the done action immediately.\n")

        if reference_action_hint:
            parts.append(
                "# Reference Action Plan\n"
                "The following comes from a known-correct trajectory for this task. "
                "Use it as guidance for the next action, but still inspect the Current State, "
                "emit only ONE next action, and return done once the Success Criteria are met.\n"
                f"{reference_action_hint}\n"
            )

        variant = self.prompt_variant if prompt_variant is None else prompt_variant
        variant_hint = format_prompt_variant_hint(variant)
        if variant_hint:
            parts.append(f"# Decision Style\n{variant_hint}\n")

        # Document path for target field
        doc_path = obs.app_state.__dict__.get("document_path", "") or obs.app_state.active_document

        parts.append(
            f"# Current State — {obs.meta.app_name}\n"
            f"Document path (use as target): {doc_path}\n"
            f"Summary: {obs.data_summary}\n\n"
            f"## Interactive Elements ({len(obs.interactive_elements)} total):\n"
            f"```json\n{json.dumps(elements_summary, indent=2, default=str)}\n```"
        )

        # Show recent errors if any
        errors = obs.environment.recent_errors if obs.environment else []
        if errors:
            parts.append(f"\n## Recent Errors\n{json.dumps(errors, indent=2)}")

        parts.append("\nReturn ONE JSON action object.")
        return "\n".join(parts)

    @staticmethod
    def _extract_first_json(text: str) -> str:
        """Extract the first complete JSON object from text by brace counting."""
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return text  # unbalanced — return as-is, json.loads will raise

    def parse_action(self, llm_output: str) -> Action:
        # Try to extract content from a markdown code block first
        code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", llm_output, re.DOTALL)
        if code_match:
            candidate = code_match.group(1).strip()
            # If the code block isn't empty, prefer it; otherwise fall through to raw text
            if candidate:
                json_str = self._extract_first_json(candidate)
                data = json.loads(json_str)
                return Action(**data)

        # Fall back to raw text (handles cases where model skips markdown wrapping)
        json_str = self._extract_first_json(llm_output.strip())
        data = json.loads(json_str)
        return Action(**data)

    def parse_trace(
        self,
        model_output: str | AgentModelOutput,
        instruction: str = "",
    ) -> AgentTrace:
        if isinstance(model_output, AgentModelOutput):
            raw_text = model_output.text
            reasoning_summary = model_output.reasoning_summary
            provider = model_output.provider
            model = model_output.model
        else:
            raw_text = model_output
            reasoning_summary = ""
            provider = ""
            model = ""

        thought, action_text = _split_thought_action(raw_text)
        action = self.parse_action(action_text)
        return AgentTrace(
            instruction=instruction,
            thought=thought,
            action=action,
            raw_text=raw_text,
            reasoning_summary=reasoning_summary,
            provider=provider,
            model=model,
        )

    def _render_hybrid_screenshot(self) -> bytes | None:
        """Render the current software state to PNG bytes for hybrid visual observation."""
        render = getattr(self.adapter, "render_to_png", None)
        if render is None:
            return None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                path = tmp.name
            out = render(output_path=path)
            data = open(out if out else path, "rb").read()
            try:
                os.unlink(path)
            except OSError:
                pass
            return data or None
        except Exception:
            return None

    def step(
        self,
        task_description: str,
        obs: Observation | None = None,
        success_hint: str = "",
        prompt_variant: str | None = None,
        reference_action_hint: str = "",
    ) -> tuple[Observation, Action, Observation, AgentTrace]:
        if obs is None:
            obs = self.adapter.observe()

        prompt = self.format_observation(
            obs,
            task_description,
            success_hint=success_hint,
            prompt_variant=prompt_variant,
            reference_action_hint=reference_action_hint,
        )
        model_start = time.monotonic()
        if self.hybrid_vision:
            shot = self._render_hybrid_screenshot()
            try:
                model_output = self.llm_fn(prompt, shot)
            except TypeError:
                model_output = self.llm_fn(prompt)
        else:
            model_output = self.llm_fn(prompt)
        model_latency_ms = (time.monotonic() - model_start) * 1000
        trace = self.parse_trace(model_output, instruction=task_description)
        trace.model_latency_ms = round(model_latency_ms, 2)
        if trace.action.action_type == "done":
            trace.action_execution_latency_ms = 0.0
            return obs, trace.action, obs, trace
        execute_start = time.monotonic()
        new_obs = self.adapter.execute(trace.action)
        trace.action_execution_latency_ms = round((time.monotonic() - execute_start) * 1000, 2)
        return obs, trace.action, new_obs, trace

    def run(self, task_description: str) -> list[tuple[Observation, Action, Observation, AgentTrace]]:
        history = []
        obs = self.adapter.observe()
        for _ in range(self.max_steps):
            obs, action, new_obs, trace = self.step(task_description, obs)
            history.append((obs, action, new_obs, trace))
            obs = new_obs
            if action.action_type == "done":
                break
        return history


def create_llm_fn(
    provider: str = "mock",
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    reasoning_effort: str = "medium",
) -> LLMFunction:
    """Create an LLM function for the agent.

    Args:
        provider: "openai", "anthropic", or "mock" (default)
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        api_key: API key for the provider. If empty, reads from
                 OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable.
        base_url: Custom API base URL (OpenAI-compatible). If empty, reads
                  from OPENAI_API_BASE environment variable.

    Returns:
        A callable that takes prompt string and returns response string.
    """
    if provider == "mock":
        def mock_fn(prompt: str) -> AgentModelOutput:
            # Default mock: immediately terminate so screenshot-chain smokes
            # can validate rendering without relying on software-specific actions.
            return AgentModelOutput(
                text='Thought: Rendering smoke only; no software mutation is required.\nAction: {"action_type": "done", "target": "", "params": {}}',
                provider="mock",
                model="mock",
            )
        return mock_fn

    elif provider == "openai":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env or pass api_key argument.")

        resolved_base = base_url or os.environ.get("OPENAI_API_BASE", "")

        # Hybrid visual observation (mJNV #5 ablation): ASIL structured prompt PLUS
        # a rendered screenshot, sent as one multimodal chat/completions message.
        # Enabled with ASIL_HYBRID_VISION=1; the agent passes the screenshot bytes.
        if os.environ.get("ASIL_HYBRID_VISION", "").strip().lower() in {"1", "true", "yes", "on"}:
            def openai_hybrid_fn(prompt: str, image_bytes: bytes | None = None) -> AgentModelOutput:
                import base64 as _b64
                import openai

                client_kwargs = {"api_key": resolved_key}
                if resolved_base:
                    client_kwargs["base_url"] = resolved_base
                client = openai.OpenAI(**client_kwargs)
                model_name = model or "gpt-4o"
                content: list[dict] = [{"type": "text", "text": prompt}]
                if image_bytes:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _b64.b64encode(image_bytes).decode("ascii")},
                    })
                request: dict = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    "max_tokens": int(os.environ.get("ASIL_HYBRID_MAX_TOKENS", "3000") or 3000),
                }
                last_err = None
                for attempt in range(int(os.environ.get("ASIL_GUI_LLM_RETRIES", "100") or 100)):
                    try:
                        response = client.chat.completions.create(**request)
                        return AgentModelOutput(
                            text=(response.choices[0].message.content or "") if response.choices else "",
                            provider="openai-hybrid-vision",
                            model=model_name,
                        )
                    except Exception as exc:  # noqa: BLE001 - backend owns retry
                        last_err = exc
                        time.sleep(min(10.0, 2.0 * (attempt + 1)))
                raise RuntimeError(f"hybrid vision chat/completions failed: {last_err}")
            return openai_hybrid_fn

        def openai_fn(prompt: str) -> AgentModelOutput:
            import openai
            client_kwargs = {"api_key": resolved_key}
            if resolved_base:
                client_kwargs["base_url"] = resolved_base
            client = openai.OpenAI(**client_kwargs)
            model_name = model or "gpt-4o"
            if model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3"):
                response = client.responses.create(
                    model=model_name,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    instructions=SYSTEM_PROMPT,
                    reasoning={"effort": reasoning_effort, "summary": "concise"},
                )
                return _extract_openai_output(response, model_name)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            return AgentModelOutput(text=content, provider="openai", model=model_name)
        return openai_fn

    elif provider == "anthropic":
        resolved_key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")
                        or os.environ.get("OPENAI_API_KEY", ""))
        if not resolved_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY in .env or pass api_key argument.")
        model_name = model or "claude-sonnet-4-20250514"
        gateway_base = (base_url or os.environ.get("OPENAI_API_BASE", "")
                        or os.environ.get("OPENAI_BASE_URL", "")
                        or os.environ.get("ANTHROPIC_BASE_URL", "")).rstrip("/")
        # mr.* models are served by the DashScope compatible-mode gateway via
        # native-protocol passthrough (same as the GUI computer-use path). The
        # Anthropic SDK cannot target that endpoint (it appends its own /v1 to the
        # base, doubling the path), so route those through a direct gateway POST.
        if model_name.startswith("mr.") or "dashscope" in gateway_base:
            if not gateway_base:
                raise ValueError("mr.* Anthropic model requires OPENAI_API_BASE/ANTHROPIC_BASE_URL (gateway).")
            import json as _json
            import urllib.request as _urlreq

            def anthropic_gateway_fn(prompt: str) -> AgentModelOutput:
                body = {
                    "model": model_name,
                    "dashscope_extend_params": {"using_native_protocol": "true"},
                    "max_tokens": int(os.environ.get("ASIL_ANTHROPIC_MAX_TOKENS", "4000") or 4000),
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                }
                last_err = None
                for attempt in range(int(os.environ.get("ASIL_GUI_LLM_RETRIES", "100") or 100)):
                    try:
                        req = _urlreq.Request(
                            gateway_base + "/chat/completions",
                            data=_json.dumps(body).encode("utf-8"),
                            headers={"Authorization": f"Bearer {resolved_key}",
                                     "Content-Type": "application/json"},
                            method="POST")
                        with _urlreq.urlopen(req, timeout=float(os.environ.get("ASIL_GUI_LLM_TIMEOUT_S", "900") or 900)) as r:
                            data = _json.loads(r.read().decode("utf-8"))
                        blocks = data.get("content", []) if isinstance(data, dict) else []
                        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                        return AgentModelOutput(text=text, provider="anthropic-gateway", model=model_name)
                    except Exception as exc:  # noqa: BLE001 - backend owns retry
                        last_err = exc
                        time.sleep(min(10.0, 2.0 * (attempt + 1)))
                raise RuntimeError(f"anthropic gateway call failed after retries: {last_err}")
            return anthropic_gateway_fn

        def anthropic_fn(prompt: str) -> AgentModelOutput:
            import anthropic
            client = anthropic.Anthropic(api_key=resolved_key, base_url=(gateway_base or None))
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return AgentModelOutput(
                text=response.content[0].text,
                provider="anthropic",
                model=model_name,
            )
        return anthropic_fn

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'mock'.")
