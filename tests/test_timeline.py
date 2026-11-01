"""Invocation spans must be measured, request-local and honest about missing billing."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langsmith import RunTree, tracing_context

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.observability.timeline import (
    InvocationCollector,
    measured_stream,
    safe_detail,
)
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService


def response(inputs=1000, outputs=100, cached=200):
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="private model content",
                        usage_metadata={
                            "input_tokens": inputs,
                            "output_tokens": outputs,
                            "total_tokens": inputs + outputs,
                            "input_token_details": {"cache_read": cached},
                        },
                    )
                )
            ]
        ]
    )


def collect(fn):
    async def run():
        queue = asyncio.Queue()
        collector = InvocationCollector(Settings(api_key="private-test-key"), queue)
        await fn(collector, queue)

    asyncio.run(run())


def test_parallel_spans_measure_individual_intervals_and_do_not_sum_wall_time():
    async def check(c, q):
        # Two tools overlap. Each duration must end at its own callback.
        c.on_tool_start({"name": "calculate"}, "", inputs={"expression": "2+2"}, run_id="a")
        c.on_tool_start({"name": "get_city_weather"}, "", inputs={"city": "上海"}, run_id="b")
        await asyncio.sleep(0.02)
        c.on_tool_end({"result": 4}, run_id="a")
        await asyncio.sleep(0.03)
        c.on_tool_end({"temperature": 20, "price": 500}, run_id="b")
        a, b = c.spans.values()
        assert a["duration_ms"] >= 15
        assert b["duration_ms"] >= 45
        assert b["duration_ms"] > a["duration_ms"]
        summary = c.summary()
        assert summary["duration_ms"] < a["duration_ms"] + b["duration_ms"]
        assert a["tool_cost"]["amount"] == 0
        assert b["tool_cost"]["amount"] is None  # a quote is not the API cost
        assert summary["unknown_tool_costs"] == 1
        assert not summary["cost_complete"]

    collect(check)


def test_nested_models_count_once_and_cost_uses_each_calls_own_context_tier():
    async def check(c, q):
        c.on_tool_start({"name": "task"}, "private task instructions", run_id="task")
        c.on_chain_start({}, {}, run_id="nested", parent_run_id="task")
        for key in ("m1", "m2"):
            c.on_chat_model_start({}, [["private prompt"]], run_id=key, parent_run_id="nested")
            c.on_llm_end(response(70000, 100, 0), run_id=key)
        c.on_tool_end("private subagent response", run_id="task")
        summary = c.summary()
        assert summary["model_calls"] == 2
        assert summary["input_tokens"] == 140000
        assert summary["output_tokens"] == 200
        assert summary["total_tokens"] == 140200
        assert summary["model_cost_estimate_cny"] == pytest.approx(0.1124)
        assert summary["usage_complete"] and summary["cost_complete"]
        assert c.spans["m1"]["parent_tool_id"] == "task"
        assert c.spans["task"]["tool_cost"]["status"] == "included"
        public = str([q.get_nowait() for _ in range(q.qsize())])
        for private in (
            "private prompt",
            "private model content",
            "private task instructions",
            "private subagent response",
        ):
            assert private not in public

    collect(check)


def test_native_trace_ancestors_bridge_middleware_without_chain_callbacks():
    async def check(c, q):
        root = RunTree(name="root", run_type="chain", inputs={})
        task_a = root.create_child(name="task", run_type="tool", inputs={})
        task_b = root.create_child(name="task", run_type="tool", inputs={})
        for task in (task_a, task_b):
            c.on_tool_start({"name": "task"}, "", run_id=task.id, parent_run_id=root.id)
        # A native-only traceable middleware span is invisible to on_chain_start.
        middleware = task_a.create_child(name="middleware", run_type="chain", inputs={})
        with tracing_context(parent=middleware, enabled=False):
            c.on_chat_model_start({}, [], run_id="child-model", parent_run_id=middleware.id)
            c.on_tool_start(
                {"name": "calculate"}, "", run_id="child-tool", parent_run_id=middleware.id
            )
        assert c.spans["child-model"]["parent_tool_id"] == str(task_a.id)
        assert c.spans["child-tool"]["parent_tool_id"] == str(task_a.id)
        with tracing_context(parent=root, enabled=False):
            c.on_chat_model_start({}, [], run_id="main-model", parent_run_id=root.id)
        assert c.spans["main-model"]["parent_tool_id"] is None

    collect(check)


def test_missing_usage_unknown_model_and_model_errors_are_not_zero_billing():
    async def check(c, q):
        c.on_chat_model_start({}, [], run_id="missing")
        c.on_llm_end(LLMResult(generations=[]), run_id="missing")
        c.on_chat_model_start({}, [], run_id="unknown", metadata={"ls_model_name": "unknown"})
        c.on_llm_end(response(), run_id="unknown")
        c.on_chat_model_start({}, [], run_id="failure")
        c.on_llm_error(ValueError("private-test-key"), run_id="failure")
        assert c.spans["missing"]["usage"] is None
        assert c.spans["unknown"]["cost_estimate"] is None
        assert c.spans["failure"]["status"] == "error"
        assert c.spans["failure"]["error"] == "ValueError"
        assert not c.summary()["usage_complete"]
        assert not c.summary()["model_cost_complete"]

    collect(check)


def test_safe_tool_details_hide_secrets_and_structured_failure_is_visible():
    async def check(c, q):
        c.on_tool_start(
            {"name": "search_flights"},
            "",
            run_id="flight",
            inputs={
                "api_key": "private-test-key",
                "token": "another-secret",
                "url": "https://example.com/?api_key=other-secret",
                "city": "上海",
            },
        )
        c.on_tool_end(
            ToolMessage(tool_call_id="f", content='{"error":"not configured"}'), run_id="flight"
        )
        span = c.spans["flight"]
        assert span["status"] == "error"
        assert span["tool_cost"]["amount"] is None
        public = str([q.get_nowait() for _ in range(q.qsize())])
        assert "上海" in public
        assert not any(s in public for s in ("private-test-key", "another-secret", "other-secret"))
        assert len(safe_detail("x" * 10000)) < 1700

    collect(check)


def test_callbacks_stream_before_slow_graph_returns_and_cancellation_stops_producer():
    async def check():
        queue = asyncio.Queue()
        c = InvocationCollector(Settings(), queue)
        closed = asyncio.Event()

        async def graph():
            try:
                c.on_chat_model_start({}, [], run_id="slow")
                await asyncio.sleep(60)
                yield "messages", None
            finally:
                closed.set()

        stream = measured_stream(graph(), queue)
        async with asyncio.timeout(1):
            mode, event = await anext(stream)
        assert mode == "trace" and event["span"]["status"] == "running"
        await stream.aclose()
        assert closed.is_set()

    asyncio.run(check())


def test_service_delivers_live_spans_and_final_summary_without_changing_reply():
    class Graph:
        async def astream(self, *args, config, **kwargs):
            c = config["callbacks"][0]
            c.on_chat_model_start({}, [], run_id="model")
            await asyncio.sleep(0.01)
            c.on_llm_end(response(), run_id="model")
            yield "messages", (SimpleNamespace(content="结果"), {"langgraph_node": "model"})

    async def check():
        request = ChatRequest(
            conversation_id=uuid4(), messages=[{"role": "user", "content": "test"}]
        )
        events = [e async for e in ChatService(Graph(), Settings()).stream(request, str(uuid4()))]
        assert [e["type"] for e in events] == ["meta", "trace", "trace", "delta", "usage", "done"]
        assert events[1]["span"]["status"] == "running"
        assert events[2]["span"]["duration_ms"] >= 8
        assert events[-1]["trace_summary"]["total_tokens"] == 1100
        assert events[-1]["trace_summary"]["model_cost_estimate_cny"] == pytest.approx(0.000872)

    asyncio.run(check())


def test_tool_callbacks_from_worker_threads_reach_request_loop():
    async def check(c, q):
        await asyncio.to_thread(c.on_tool_start, {"name": "calculate"}, "2+2", run_id="thread")
        await asyncio.to_thread(c.on_tool_end, {"result": 4}, run_id="thread")
        await asyncio.sleep(0)
        assert [q.get_nowait()[1]["span"]["status"] for _ in range(q.qsize())] == [
            "running",
            "finished",
        ]

    collect(check)
