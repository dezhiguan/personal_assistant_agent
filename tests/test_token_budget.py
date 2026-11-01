"""Transport-level budget checks, including provider failures and measured drift."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents.model import build_model
from personal_assistant_agent.agents.token_budget import (
    TokenBudgetViolation,
    TokenCountUnavailable,
)
from personal_assistant_agent.core.config import Settings

pytestmark = pytest.mark.provider_counter


def model():
    return build_model(Settings(api_key="synthetic", context_window_tokens=8192))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("count,allowed", [(6016, True), (6017, False)])
def test_precise_budget_boundary_blocks_generation(monkeypatch, asynchronous, count, allowed):
    value = model()
    packets = []

    def counted(url, **kwargs):
        packets.append(kwargs["json"])
        assert url.endswith("/apps/anthropic/v1/messages/count_tokens")
        return httpx.Response(200, json={"input_tokens": count}, request=httpx.Request("POST", url))

    async def acounted(url, **kwargs):
        return counted(url, **kwargs)

    monkeypatch.setattr(value.root_client._client, "post", counted)
    monkeypatch.setattr(value.root_async_client._client, "post", acounted)
    result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
    with (
        patch("langchain_openai.ChatOpenAI._generate", return_value=result) as sync,
        patch("langchain_openai.ChatOpenAI._agenerate", return_value=result) as async_,
    ):

        def invoke():
            return (
                asyncio.run(value.ainvoke("synthetic question"))
                if asynchronous
                else value.invoke("synthetic question")
            )

        if allowed:
            assert invoke().content == "ok"
        else:
            with pytest.raises(ContextOverflowError):
                invoke()
        assert sync.call_count + async_.call_count == int(allowed)
    assert len(packets) == 1


@pytest.mark.parametrize("status,body", [(503, {}), (200, {}), (200, {"input_tokens": -1})])
def test_count_failure_never_falls_back_to_an_estimate(monkeypatch, status, body):
    value = model()
    monkeypatch.setattr(
        value.root_client._client,
        "post",
        lambda url, **kw: httpx.Response(status, json=body, request=httpx.Request("POST", url)),
    )
    with patch("langchain_openai.ChatOpenAI._generate") as generation:
        with pytest.raises(TokenCountUnavailable):
            value.invoke("synthetic question")
    generation.assert_not_called()


def test_counted_packet_includes_system_tools_and_results_and_caches(monkeypatch):
    value = model()
    captured = []

    def counted(url, **kwargs):
        captured.append(kwargs["json"])
        return httpx.Response(200, json={"input_tokens": 1100}, request=httpx.Request("POST", url))

    monkeypatch.setattr(value.root_client._client, "post", counted)
    messages = [
        SystemMessage(content="synthetic system"),
        HumanMessage(content="read sample"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "id": "r1", "args": {"file_path": "/sample.txt"}}],
        ),
        ToolMessage(content="synthetic result", tool_call_id="r1"),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            },
        }
    ]
    assert value.count_input_tokens(messages, tools=tools) == 1100
    # A cloned summary model shares counts, but never bypasses its own output cap.
    assert (
        value.model_copy(update={"max_tokens": 384}).count_input_tokens(messages, tools=tools)
        == 1100
    )
    assert len(captured) == 1
    packet = captured[0]
    assert packet["system"] == "synthetic system"
    assert packet["messages"][-1]["content"][0]["content"] == "synthetic result"
    assert packet["messages"][-2]["content"][0]["input"] == {"file_path": "/sample.txt"}
    assert packet["tools"][0]["input_schema"] == tools[0]["function"]["parameters"]
    value.count_input_tokens([*messages, HumanMessage(content="changed")], tools=tools)
    assert len(captured) == 2
    assert "synthetic" in json.dumps(packet)


def test_postflight_drift_is_not_retried_and_keeps_billable_usage(monkeypatch):
    value = model()
    monkeypatch.setattr(type(value), "count_input_tokens", lambda *a, **kw: 5000)
    usage = {"input_tokens": 8200, "output_tokens": 10, "total_tokens": 8210}
    result = ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="ok", usage_metadata=usage))]
    )
    with patch("langchain_openai.ChatOpenAI._generate", return_value=result) as generation:
        with pytest.raises(TokenBudgetViolation) as error:
            value.invoke("synthetic")
    assert error.value.usage_metadata == usage
    assert generation.call_count == 1
