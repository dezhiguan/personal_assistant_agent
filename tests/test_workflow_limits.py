"""Long real graphs finish; independent call/time limits still stop unbounded work."""

import asyncio
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.errors import GraphRecursionError

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService


def payload():
    return ChatRequest(
        conversation_id=uuid4(), messages=[{"role": "user", "content": "执行连续计算"}]
    )


async def collect(graph, settings):
    return [e async for e in ChatService(graph, settings).stream(payload(), str(uuid4()))]


def generator(count):
    async def generate(self, messages, **kwargs):
        done = sum(isinstance(m, ToolMessage) for m in messages)
        response = AIMessage(content=f"已完成{count}次计算。")
        if done < count:
            response = AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculate", "args": {"expression": f"{done}+1"}, "id": f"math-{done}"}
                ],
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

    return generate


def test_thirty_sequential_tool_calls_exceed_old_graph_budget_but_finish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(api_key="synthetic")
    with patch("langchain_openai.ChatOpenAI._astream", generator(30)):
        events = asyncio.run(collect(build_agent(settings), settings))
    assert events[-1]["type"] == "done"
    assert sum(e["type"] == "tool" and e["status"] == "finished" for e in events) == 30
    assert not any(e["type"] == "error" for e in events)


def test_model_call_budget_stops_repeating_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(api_key="synthetic", max_model_calls=2)
    with patch("langchain_openai.ChatOpenAI._astream", generator(30)):
        events = asyncio.run(collect(build_agent(settings), settings))
    assert events[-1]["code"] == "WORKFLOW_CALL_LIMIT"
    assert sum(e["type"] == "tool" and e["status"] == "finished" for e in events) == 2
    assert not any(e["type"] == "done" for e in events)


def test_tool_call_budget_stops_agent_separately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(api_key="synthetic", max_tool_calls=2)
    with patch("langchain_openai.ChatOpenAI._astream", generator(30)):
        events = asyncio.run(collect(build_agent(settings), settings))
    assert events[-1]["code"] == "WORKFLOW_CALL_LIMIT"
    assert sum(e["type"] == "tool" and e["status"] == "finished" for e in events) == 2


def test_graph_limit_reports_cause_and_ends_outstanding_progress():
    class Graph:
        async def astream(self, *args, **kwargs):
            yield (
                "updates",
                {
                    "model": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": "validate_time_budget", "args": {}, "id": "pending"}
                                ],
                            )
                        ]
                    }
                },
            )
            raise GraphRecursionError("private-detail-must-not-leak")

    events = asyncio.run(collect(Graph(), Settings()))
    assert events[-1]["code"] == "WORKFLOW_STEP_LIMIT"
    assert [e["status"] for e in events if e["type"] == "tool"] == ["running", "error"]
    assert "private-detail" not in str(events)


def test_workflow_timeout_is_distinct_from_model_timeout():
    class Graph:
        async def astream(self, *args, **kwargs):
            await asyncio.sleep(0.1)
            yield

    events = asyncio.run(collect(Graph(), Settings(workflow_timeout=0.01)))
    assert events[-1]["code"] == "WORKFLOW_TIMEOUT"
