import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings


def test_final_claim_requires_real_plan_commit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    completed = [{"content": "撰写方案", "status": "completed"}]
    replies = iter(
        [
            AIMessage(content="所有待办已完成"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "id": "commit",
                        "args": {"todos": completed},
                    }
                ],
            ),
            AIMessage(content="方案已交付，尚未举办活动。"),
        ]
    )

    async def generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(replies))])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        result = asyncio.run(
            build_agent(Settings(api_key="test-only")).ainvoke(
                {
                    "messages": [{"role": "user", "content": "继续完成方案"}],
                    "todos": [{"content": "撰写方案", "status": "in_progress"}],
                }
            )
        )
    assert result["todos"] == completed
    assert result["plan_checks"] == 1
    assert result["messages"][-1].content == "方案已交付，尚未举办活动。"
