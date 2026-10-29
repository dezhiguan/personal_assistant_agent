"""Cancellation propagation, text filtering and trace metadata."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService


def request():
    return ChatRequest(conversation_id=uuid4(), messages=[{"role": "user", "content": "你好"}])


def test_cancel_reaches_agent():
    cancelled = []

    class SlowGraph:
        async def astream(self, *args, **kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)
            yield

    async def check():
        stream = ChatService(SlowGraph(), Settings()).stream(request(), str(uuid4()))
        assert (await anext(stream))["type"] == "meta"
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation was swallowed")
        assert cancelled == [True]

    asyncio.run(check())


def test_only_main_model_text_is_streamed():
    class Graph:
        async def astream(self, *args, **kwargs):
            yield "messages", (SimpleNamespace(content="internal"), {"langgraph_node": "tools"})
            yield "messages", (
                SimpleNamespace(
                    content=[
                        {"type": "reasoning", "text": "private reasoning"},
                        {"type": "text", "text": "answer"},
                    ]
                ),
                {"langgraph_node": "model"},
            )

    async def check():
        output = [e async for e in ChatService(Graph(), Settings()).stream(request(), str(uuid4()))]
        assert [e.get("text") for e in output if e["type"] == "delta"] == ["answer"]

    asyncio.run(check())


def test_committed_plans_are_streamed_without_tool_arguments_or_duplicate_updates():
    todos = [{"content": "梳理目标", "status": "in_progress"}]

    class Graph:
        async def astream(self, value, *, stream_mode, **kwargs):
            assert stream_mode == ["messages", "updates"]
            assert value["todos"] == []
            yield "messages", (
                SimpleNamespace(content="", tool_call_chunks=[{"args": "internal arguments"}]),
                {"langgraph_node": "model"},
            )
            yield "updates", {"tools": {"todos": todos}}
            yield "updates", {"model": {"todos": todos}}
            yield "updates", {"tools": {"todos": []}}
            yield "messages", (SimpleNamespace(content="结果"), {"langgraph_node": "model"})

    async def check():
        output = [e async for e in ChatService(Graph(), Settings()).stream(request(), str(uuid4()))]
        assert [e["type"] for e in output] == ["meta", "plan", "plan", "delta", "done"]
        assert [e["todos"] for e in output if e["type"] == "plan"] == [todos, []]
        assert "internal arguments" not in str(output)

    asyncio.run(check())


def test_plan_only_run_is_not_reported_as_success():
    class Graph:
        async def astream(self, *args, **kwargs):
            yield "updates", {"tools": {"todos": [{"content": "整理", "status": "pending"}]}}

    async def check():
        output = [e async for e in ChatService(Graph(), Settings()).stream(request(), str(uuid4()))]
        assert [e["type"] for e in output] == ["meta", "plan", "error"]

    asyncio.run(check())


def test_settings_repr_hides_credentials():
    value = repr(Settings(api_key="secret-model-key", langsmith_api_key="secret-trace-key"))
    assert "secret-model-key" not in value
    assert "secret-trace-key" not in value
