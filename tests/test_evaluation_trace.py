"""Evaluation sees real tool denials and state commits, using the native tracer."""

import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.evaluation.scenarios import tool_receipts


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
