"""Regression: the ASIL agent's Anthropic provider must route mr.* models through
the DashScope compatible-mode gateway (native-protocol POST), not the Anthropic SDK.

Bug history: create_llm_fn's anthropic branch used anthropic.Anthropic(api_key=...)
with no base_url, so mr.claude-* runs hit api.anthropic.com with a DashScope key and
failed every step -> empty trajectories / all-zero scores in the #3 comparison.
"""
import json
import urllib.request

from asil import agent


def test_mr_claude_routes_through_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("ASIL_GUI_LLM_RETRIES", "1")
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"content": [
                {"type": "text",
                 "text": 'Thought: done.\nAction: {"action_type": "done", "target": "", "params": {}}'}
            ]}).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    fn = agent.create_llm_fn(provider="anthropic", model="mr.claude-sonnet-4-6-20260217")
    out = fn("Set cell A1 to 'X'")

    # routed to the gateway chat endpoint with native-protocol passthrough
    assert captured["url"].endswith("/compatible-mode/v1/chat/completions")
    assert captured["body"]["dashscope_extend_params"]["using_native_protocol"] == "true"
    assert captured["body"]["model"] == "mr.claude-sonnet-4-6-20260217"
    assert captured["auth"] == "Bearer sk-test-key"
    # native Anthropic content blocks are parsed back to text
    assert "done" in out.text
    assert out.provider == "anthropic-gateway"


def test_non_mr_anthropic_still_uses_sdk(monkeypatch):
    # A non-mr model with no dashscope base must NOT take the gateway branch;
    # it should fall through to the anthropic SDK path (import happens lazily).
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fn = agent.create_llm_fn(provider="anthropic", model="claude-sonnet-4-20250514")
    # The returned closure is the SDK path; we don't call it (no network/SDK here),
    # we only assert routing did not raise and produced a callable.
    assert callable(fn)
