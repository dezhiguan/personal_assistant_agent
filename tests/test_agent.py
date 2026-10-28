"""Exercise the real Deep Agents graph with a substituted model transport."""

import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings


def test_actual_graph_uses_qwen_without_exposing_tools():
    calls = []

    async def generate(self, messages, **kwargs):
        calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="你好"))])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        agent = build_agent(Settings(api_key="test-only"))
        result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "你好"}]}))
    assert result["messages"][-1].content == "你好"
    assert calls
    assert all(not c.get("tools") for c in calls)
