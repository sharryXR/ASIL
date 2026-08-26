"""Multimodal LLM wrappers for the screenshot-driven GUI agent."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv

from asil.gui_agent.parser import GUIAction

load_dotenv()


@dataclass(frozen=True)
class GUIModelOutput:
    text: str
    reasoning_summary: str = ""
    provider: str = ""
    model: str = ""
    actions: tuple[GUIAction, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


GUILLMFunction = Callable[[str, bytes], GUIModelOutput]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _gateway_chat_enabled(model_name: str = "") -> bool:
    """Route the openai/anthropic providers through the shared mr.* gateway using
    the chat/completions transport instead of the OpenAI Responses API / Anthropic
    SDK. The `mr.gpt-5.4-*` and `mr.claude-*` models exposed by the DashScope
    compatible-mode gateway only accept chat/completions (Responses `/responses`
    and the hosted computer-use tool return errors), so the screenshot-driven GUI
    agent falls back to the pyautogui text-action protocol the runner already
    parses. Enabled with ASIL_GUI_GATEWAY_CHAT=1 or automatically for `mr.` models
    (whose name reaches the eval container through the --model CLI argument).
    """
    flag = os.environ.get("ASIL_GUI_GATEWAY_CHAT", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return model_name.strip().lower().startswith("mr.")


def _retry_sleep(attempt: int) -> None:
    base = _env_float("ASIL_GUI_LLM_RETRY_BASE_S", 2.0)
    cap = _env_float("ASIL_GUI_LLM_RETRY_CAP_S", 10.0)
    time.sleep(min(cap, base * (attempt + 1)))



def _png_data_url(image_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


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

    parts: list[str] = []
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
        parts: list[str] = []
        for part in summary:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text", "")
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(summary)


def _extract_openai_output(response, model_name: str) -> GUIModelOutput:
    texts: list[str] = []
    summaries: list[str] = []

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

    if not texts:
        output_text = getattr(response, "output_text", None)
        if output_text is None and hasattr(response, "model_dump"):
            output_text = response.model_dump().get("output_text", "")
        if output_text:
            texts.append(output_text)

    return GUIModelOutput(
        text="\n".join(texts).strip(),
        reasoning_summary="\n".join(summaries).strip(),
        provider="openai",
        model=model_name,
    )


def create_gui_llm_fn(
    provider: str = "mock",
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    reasoning_effort: str = "medium",
) -> GUILLMFunction:
    if provider == "mock":
        def mock_fn(prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
            del prompt, screenshot_bytes
            return GUIModelOutput(
                text="Thought: Mock GUI smoke run; stop after validating the screenshot chain.\nAction: DONE",
                provider="mock",
                model="mock",
            )
        return mock_fn

    if provider == "openai":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env or pass api_key.")
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("OPENAI_API_BASE", "")
        request_timeout_s = _env_float("ASIL_GUI_LLM_TIMEOUT_S", 180.0)
        model_name = model or "gpt-5.4"

        # Computer-use path (ScienceBoard-consistent): the `mr.gpt-*-responses`
        # gateway ids and ASIL_GUI_AGENT_BACKEND=osworld_gpt54 use the hosted
        # OpenAI Responses computer tool. Checked before the plain chat fallback.
        from asil.gui_agent.osworld_gpt54 import (
            create_osworld_gpt54_llm_fn,
            should_use_osworld_gpt54_backend,
        )

        _computer_use = os.environ.get("ASIL_GUI_COMPUTER_USE", "").strip().lower() in {"1", "true", "yes", "on"}
        if (
            model_name.strip().lower().endswith("-responses")
            or should_use_osworld_gpt54_backend(provider, model_name)
            or (_computer_use and "gpt" in model_name.lower())
        ):
            backend_effort = os.environ.get("ASIL_GUI_REASONING_EFFORT", "").strip() or (
                "xhigh" if reasoning_effort == "medium" else reasoning_effort
            )
            return create_osworld_gpt54_llm_fn(
                model=model_name,
                api_key=resolved_key,
                base_url=resolved_base,
                timeout_s=request_timeout_s,
                reasoning_effort=backend_effort,
            )

        if _gateway_chat_enabled(model_name):
            effort = os.environ.get("ASIL_GUI_REASONING_EFFORT", "").strip()
            max_tokens = _env_int("ASIL_GUI_MAX_TOKENS", 1500)
            retries = _env_int("ASIL_GUI_LLM_RETRIES", 8)

            def openai_gateway_chat_fn(prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
                import openai

                client_kwargs: dict[str, Any] = {"api_key": resolved_key, "timeout": request_timeout_s}
                if resolved_base:
                    client_kwargs["base_url"] = resolved_base
                client = openai.OpenAI(**client_kwargs)
                request: dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": _png_data_url(screenshot_bytes)}},
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                }
                if effort:
                    request["reasoning_effort"] = effort
                last_error: Exception | None = None
                for attempt in range(max(1, retries)):
                    try:
                        response = client.chat.completions.create(**request)
                        text = (response.choices[0].message.content or "") if response.choices else ""
                        return GUIModelOutput(
                            text=text.strip(),
                            provider="openai-gateway-chat",
                            model=model_name,
                        )
                    except Exception as exc:  # noqa: BLE001 - backend owns retry
                        last_error = exc
                        if attempt + 1 >= max(1, retries):
                            break
                        _retry_sleep(attempt)
                raise RuntimeError(f"gateway chat/completions (openai) failed: {last_error}")

            return openai_gateway_chat_fn

        from asil.gui_agent.osworld_gpt54 import (
            create_osworld_gpt54_llm_fn,
            should_use_osworld_gpt54_backend,
        )

        if should_use_osworld_gpt54_backend(provider, model_name):
            backend_effort = os.environ.get("ASIL_GUI_REASONING_EFFORT", "").strip() or (
                "xhigh" if reasoning_effort == "medium" else reasoning_effort
            )
            return create_osworld_gpt54_llm_fn(
                model=model_name,
                api_key=resolved_key,
                base_url=resolved_base,
                timeout_s=request_timeout_s,
                reasoning_effort=backend_effort,
            )

        def openai_fn(prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
            import openai

            client_kwargs = {"api_key": resolved_key, "timeout": request_timeout_s}
            if resolved_base:
                client_kwargs["base_url"] = resolved_base
            client = openai.OpenAI(**client_kwargs)
            response = client.responses.create(
                model=model_name,
                instructions="You are a GUI agent that responds with Thought/Action only.",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": _png_data_url(screenshot_bytes)},
                        ],
                    }
                ],
                reasoning={"effort": reasoning_effort, "summary": "concise"},
            )
            return _extract_openai_output(response, model_name)

        return openai_fn

    if provider == "anthropic":
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY in .env or pass api_key.")

        # Computer-use path (ScienceBoard-consistent): mr.claude-* through the
        # DashScope native-protocol gateway with the hosted computer_20251124 tool,
        # full message history, and recent-image trimming. Enable with
        # ASIL_GUI_COMPUTER_USE=1 (the campaign default for claude).
        _computer_use = os.environ.get("ASIL_GUI_COMPUTER_USE", "").strip().lower() in {"1", "true", "yes", "on"}
        _disable_cu = os.environ.get("ASIL_GUI_DISABLE_COMPUTER_USE", "").strip().lower() in {"1", "true", "yes", "on"}
        if (_computer_use or "claude" in (model or "").lower()) and not _disable_cu:
            from asil.gui_agent.osworld_claude import create_osworld_claude_llm_fn

            cu_base = (
                base_url
                or os.environ.get("ANTHROPIC_BASE_URL", "")
                or os.environ.get("OPENAI_BASE_URL", "")
                or os.environ.get("OPENAI_API_BASE", "")
            )
            cu_effort = os.environ.get("ASIL_GUI_REASONING_EFFORT", "").strip() or (
                "high" if reasoning_effort == "medium" else reasoning_effort
            )
            return create_osworld_claude_llm_fn(
                model=model or "mr.claude-sonnet-4-6-20260217",
                api_key=resolved_key,
                base_url=cu_base,
                timeout_s=_env_float("ASIL_GUI_LLM_TIMEOUT_S", 180.0),
                reasoning_effort=cu_effort,
            )

        if _gateway_chat_enabled(model or ""):
            resolved_base = (
                base_url
                or os.environ.get("ANTHROPIC_BASE_URL", "")
                or os.environ.get("OPENAI_BASE_URL", "")
                or os.environ.get("OPENAI_API_BASE", "")
            )
            model_name = model or "claude-sonnet-4-6-20260217"
            request_timeout_s = _env_float("ASIL_GUI_LLM_TIMEOUT_S", 180.0)
            max_tokens = _env_int("ASIL_GUI_MAX_TOKENS", 1500)
            retries = _env_int("ASIL_GUI_LLM_RETRIES", 8)
            # The DashScope compatible-mode gateway serves `mr.claude-*` in native
            # Anthropic Messages format at the /chat/completions path; routify (and
            # the real Anthropic API) serve it at /v1/messages with a version header.
            _native = "compatible-mode" not in resolved_base
            url = resolved_base.rstrip("/") + ("/v1/messages" if _native else "/chat/completions")

            def anthropic_gateway_chat_fn(prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
                body: dict[str, Any] = {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "system": "You are a GUI agent that responds with Thought/Action only.",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64.b64encode(screenshot_bytes).decode("ascii"),
                                    },
                                },
                            ],
                        }
                    ],
                }
                headers = {
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                }
                if _native:
                    headers["anthropic-version"] = "2023-06-01"
                last_error: Exception | None = None
                for attempt in range(max(1, retries)):
                    try:
                        req = urllib.request.Request(
                            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
                        )
                        with urllib.request.urlopen(req, timeout=request_timeout_s) as resp:
                            payload = json.loads(resp.read().decode("utf-8"))
                        blocks = payload.get("content", []) if isinstance(payload, dict) else []
                        text = "\n".join(
                            b.get("text", "")
                            for b in blocks
                            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
                        )
                        return GUIModelOutput(
                            text=text.strip(),
                            provider="anthropic-gateway-chat",
                            model=model_name,
                        )
                    except Exception as exc:  # noqa: BLE001 - backend owns retry
                        last_error = exc
                        if attempt + 1 >= max(1, retries):
                            break
                        _retry_sleep(attempt)
                raise RuntimeError(f"gateway chat/completions (anthropic) failed: {last_error}")

            return anthropic_gateway_chat_fn

        def anthropic_fn(prompt: str, screenshot_bytes: bytes) -> GUIModelOutput:
            import anthropic

            model_name = model or "claude-sonnet-4-20250514"
            client = anthropic.Anthropic(api_key=resolved_key)
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system="You are a GUI agent that responds with Thought/Action only.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(screenshot_bytes).decode("ascii"),
                                },
                            },
                        ],
                    }
                ],
            )
            text_parts: list[str] = []
            for block in response.content:
                if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
                    text_parts.append(block.text)
            return GUIModelOutput(
                text="\n".join(text_parts).strip(),
                provider="anthropic",
                model=model_name,
            )

        return anthropic_fn

    raise ValueError(f"Unknown GUI provider: {provider}. Use 'openai', 'anthropic', or 'mock'.")
