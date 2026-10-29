"""Exercise the real Deep Agents graph with a substituted model transport."""

import asyncio
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService


def test_actual_graph_exposes_only_planning_tool():
    calls = []

    async def generate(self, messages, **kwargs):
        calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="你好"))])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        agent = build_agent(Settings(api_key="test-only"))
        result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "你好"}]}))
    assert result["messages"][-1].content == "你好"
    assert calls
    assert all([t["function"]["name"] for t in c["tools"]] == ["write_todos"] for c in calls)


def test_actual_graph_creates_revises_and_completes_plan_through_service():
    plans = [
        [{"content": "梳理目标", "status": "in_progress"}],
        [
            {"content": "梳理目标", "status": "completed"},
            {"content": "根据限制完善方案", "status": "in_progress"},
        ],
        [
            {"content": "梳理目标", "status": "completed"},
            {"content": "根据限制完善方案", "status": "completed"},
        ],
    ]
    calls = []

    async def generate(self, messages, **kwargs):
        index = len(calls)
        calls.append(messages)
        message = (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": plans[index]},
                        "id": f"plan-{index}",
                    }
                ],
            )
            if index < len(plans)
            else AIMessage(content="这是整理后的方案。")
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def check():
        settings = Settings(api_key="test-only")
        service = ChatService(build_agent(settings), settings)
        payload = ChatRequest(
            conversation_id=uuid4(), messages=[{"role": "user", "content": "请制定方案"}]
        )
        return [e async for e in service.stream(payload, str(uuid4()))]

    async def stream_generate(self, messages, **kwargs):
        result = await generate(self, messages, **kwargs)
        message = result.generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_calls=message.tool_calls,
            )
        )

    with patch("langchain_openai.ChatOpenAI._astream", stream_generate):
        output = asyncio.run(check())
    assert [e["todos"] for e in output if e["type"] == "plan"] == plans
    assert "".join(e["text"] for e in output if e["type"] == "delta") == "这是整理后的方案。"
    assert output[-1]["type"] == "done"
    assert len(calls) == 4
    assert "根据限制完善方案" in str(calls[2][0].content)


def test_restored_todos_are_visible_to_model_without_shared_state():
    prompts = []

    async def generate(self, messages, **kwargs):
        prompts.append(str(messages[0].content))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="收到"))])

    async def check():
        agent = build_agent(Settings(api_key="test-only"))
        await agent.ainvoke(
            {
                "messages": [{"role": "user", "content": "继续"}],
                "todos": [{"content": "调整产品发布计划", "status": "in_progress"}],
            }
        )
        await agent.ainvoke({"messages": [{"role": "user", "content": "你好"}], "todos": []})

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        asyncio.run(check())
    assert "调整产品发布计划" in prompts[0]
    assert "调整产品发布计划" not in prompts[1]
