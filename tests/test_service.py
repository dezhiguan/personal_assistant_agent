"""Cancellation propagation, text filtering and trace metadata."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService


def request():
    return ChatRequest(conversation_id=uuid4(), messages=[{"role": "user", "content": "你好"}])


def test_explicit_plan_edits_emit_actual_snapshots():
    class Graph:
        async def astream(self, value, **kwargs):
            assert value["todos"] == []
            yield "messages", (SimpleNamespace(content="3"), {"langgraph_node": "model"})

    async def check(text):
        payload = ChatRequest(
            conversation_id=uuid4(),
            messages=[{"role": "user", "content": text}],
            todos=[{"content": "原任务", "status": "completed"}],
        )
        return [e async for e in ChatService(Graph(), Settings()).stream(payload, str(uuid4()))]

    appended = asyncio.run(check("请只在原待办末尾追加一项“等待确认”"))
    assert appended[1]["todos"] == [
        {"content": "原任务", "status": "completed"},
        {"content": "等待确认", "status": "pending"},
    ]
    cleared = asyncio.run(check("清空当前任务计划。只回答数字3。"))
    assert cleared[1] == {"type": "plan", "todos": []}
    assert appended[-1]["type"] == cleared[-1]["type"] == "done"


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
            yield (
                "messages",
                (
                    SimpleNamespace(
                        content=[
                            {"type": "reasoning", "text": "private reasoning"},
                            {"type": "text", "text": "answer"},
                        ]
                    ),
                    {"langgraph_node": "model"},
                ),
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
            yield (
                "messages",
                (
                    SimpleNamespace(content="", tool_call_chunks=[{"args": "internal arguments"}]),
                    {"langgraph_node": "model"},
                ),
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
    value = repr(
        Settings(
            api_key="secret-model-key",
            langsmith_api_key="secret-trace-key",
            baidu_proxy="https://user:secret-proxy-password@proxy.example",
        )
    )
    assert "secret-model-key" not in value
    assert "secret-trace-key" not in value
    assert "secret-proxy-password" not in value


def test_tool_progress_does_not_expose_arguments_outputs_or_change_plan_protocol():
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
                                    {
                                        "name": "execute",
                                        "id": "run-command",
                                        "args": {"command": "private-command"},
                                    },
                                    {"name": "write_todos", "id": "plan", "args": {"todos": []}},
                                ],
                            )
                        ]
                    }
                },
            )
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                name="execute",
                                tool_call_id="run-command",
                                content="private-output",
                                status="error",
                            )
                        ],
                        "todos": [{"content": "执行任务", "status": "in_progress"}],
                    }
                },
            )
            yield "messages", (SimpleNamespace(content="执行失败"), {"langgraph_node": "model"})

    async def check():
        return [e async for e in ChatService(Graph(), Settings()).stream(request(), str(uuid4()))]

    events = asyncio.run(check())
    assert [e["type"] for e in events] == ["meta", "tool", "tool", "plan", "delta", "done"]
    assert [e["status"] for e in events if e["type"] == "tool"] == ["running", "error"]
    assert "private-command" not in str(events)
    assert "private-output" not in str(events)


def test_structured_tool_errors_are_reported_as_failure_in_progress():
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
                                    {
                                        "name": "baidu_search",
                                        "id": "search",
                                        "args": {"query": "test"},
                                    }
                                ],
                            )
                        ]
                    }
                },
            )
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                name="baidu_search",
                                tool_call_id="search",
                                status="success",
                                content='{"source":"baidu","code":"BAIDU_CHALLENGE","error":"验证码"}',
                            )
                        ]
                    }
                },
            )
            yield (
                "messages",
                (SimpleNamespace(content="搜索需要验证码"), {"langgraph_node": "model"}),
            )

    async def check():
        return [e async for e in ChatService(Graph(), Settings()).stream(request(), str(uuid4()))]

    events = asyncio.run(check())
    assert [e["status"] for e in events if e["type"] == "tool"] == ["running", "error"]
    assert events[-1]["type"] == "done"
