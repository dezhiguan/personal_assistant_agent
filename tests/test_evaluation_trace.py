"""Evaluation sees real tool denials and state commits, using the native tracer."""

import asyncio
import json

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.evaluation.scenarios import graph_tool_receipts, tool_receipts


def test_native_tool_receipts_preserve_denials_and_redact_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    private = "unit-test-only-private-value"
    replies = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "id": "denied",
                        "args": {"command": "cat /../outside.txt"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "id": "echo",
                        "args": {"command": "echo " + private},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "id": "plan",
                        "args": {
                            "todos": [
                                {"content": "等待输入", "status": "pending"},
                            ]
                        },
                    }
                ],
            ),
            AIMessage(content="读取失败，echo成功；等待输入。"),
        ]
    )

    async def generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(replies))])

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)
    collector = RunCollectorCallbackHandler()
    graph = build_agent(Settings(api_key=private))
    asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": "检查受限执行工具，然后维护等待计划。"}]},
            config={"callbacks": [collector]},
        )
    )
    records = tool_receipts(collector.traced_runs, [private])
    assert [r["name"] for r in records] == ["execute", "execute", "write_todos"]
    assert records[0]["outputs"]["output"]["artifact"]["exit_code"] == 1
    assert "PermissionError" in records[0]["outputs"]["output"]["content"]
    assert records[1]["outputs"]["output"]["artifact"]["exit_code"] == 0
    assert "[redacted]" in records[1]["inputs"]["command"]
    assert records[2]["outputs"]["output"]["update"]["todos"][0]["status"] == "pending"
    assert private not in json.dumps(records)
    assert tool_receipts([*collector.traced_runs, *collector.traced_runs], [private]) == records


def test_fixture_tools_keep_production_contract_without_network(monkeypatch):
    import httpx

    from personal_assistant_agent.agents.tools import build_baidu_search, get_current_time
    from personal_assistant_agent.evaluation.dataset import DEFAULT_CASES
    from personal_assistant_agent.evaluation.golden import load_cases
    from personal_assistant_agent.evaluation.scenarios import fixture_tools

    def no_network(*args, **kwargs):
        raise AssertionError("Fixture must replace I/O without changing the tool interface")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    case = next(c for c in load_cases(DEFAULT_CASES)[0] if c.id == "injection-search")
    search, clock, *_ = fixture_tools(case, Settings())
    original = build_baidu_search(Settings())
    for actual, expected in [(search, original), (clock, get_current_time)]:
        assert actual.name == expected.name
        assert actual.description == expected.description
        assert actual.args_schema.model_json_schema() == expected.args_schema.model_json_schema()
    assert asyncio.run(search.ainvoke({"query": "fixture"})) == case.search_results
    invalid = {"query": "", "max_results": 11}
    assert asyncio.run(search.ainvoke(invalid)) == asyncio.run(original.ainvoke(invalid))
    assert clock.invoke({})["datetime"] == "2026-10-30T16:00:00+08:00"
    assert clock.invoke({"timezone": "Not/AZone"}) == get_current_time.invoke(
        {"timezone": "Not/AZone"}
    )


def test_streamed_graph_receipts_include_cached_and_denied_attempts(tmp_path, monkeypatch):
    from personal_assistant_agent.agents.tools import build_baidu_search

    monkeypatch.chdir(tmp_path)
    private = "synthetic-private-value"
    settings = Settings(api_key=private, context_window_tokens=8192)
    executions = []

    async def search(query, max_results):
        executions.append(query)
        return {"error": private}

    replies = iter(
        [
            AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "baidu_search",
                        "args": {"query": "fixture"},
                        "id": name,
                    }
                ],
            )
            for name in ("actual", "cached", "denied")
        ]
        + [AIMessageChunk(content="两次返回检索失败，随后工具拒绝执行。")]
    )

    async def generate(*args, **kwargs):
        yield ChatGenerationChunk(message=next(replies))

    monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
    graph = build_agent(settings, tools=[build_baidu_search(settings, search=search)])
    collector = RunCollectorCallbackHandler()

    async def check():
        async for _ in graph.astream(
            {"messages": [{"role": "user", "content": "检索产品资料。"}]},
            config={"callbacks": [collector]},
            stream_mode=["messages", "updates"],
        ):
            pass

    asyncio.run(check())
    actual = tool_receipts(collector.traced_runs, [private])
    messages = graph_tool_receipts(collector.traced_runs, [private])
    assert executions == ["fixture"]
    assert len(actual) == 1
    assert [r["tool_call_id"] for r in messages] == ["actual", "cached", "denied"]
    assert [r["cached"] for r in messages] == [False, True, False]
    assert [r["status"] for r in messages] == ["success", "success", "error"]
    assert "未授权" in messages[-1]["content"]
    assert all(r["inputs"] == {"query": "fixture"} for r in messages)
    assert private not in json.dumps(messages)
