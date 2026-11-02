"""Local synthetic browser acceptance server. Never loads .env or a live provider."""

import asyncio
from pathlib import Path

import uvicorn
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import personal_assistant_agent.app as application
from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.runtime import InteractionRuntime


@tool
async def baidu_search(query: str) -> str:
    """Read synthetic global news. Release the file barrier to complete."""
    Path("tool-started").touch()
    with Path("reads").open("a") as out:
        out.write("read\n")
    while not Path("release").exists():
        await asyncio.sleep(0.02)
    return "合成国内科技新闻：芯片研究。合成海外体育新闻：赛事。"


async def generate(self, messages, **kwargs):
    latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
    chat = any(t in latest for t in ["什么", "适合", "六岁"])
    if chat and "慢" in latest:
        while not Path("chat-release").exists():
            await asyncio.sleep(0.02)
    results = [m for m in messages if isinstance(m, ToolMessage)]
    calls = (
        []
        if chat or results
        else [{"name": "baidu_search", "id": "query", "args": {"query": latest}}]
    )
    text = (
        "科技独角兽指估值达到十亿美元的未上市科技企业。"
        if chat
        else ("合成国内科技新闻：芯片研究。\n\n采用要求：" + latest if results else "")
    )
    yield ChatGenerationChunk(message=AIMessageChunk(content=text, tool_calls=calls))


def main():
    ChatOpenAI._astream = generate
    application.InteractionRuntime = lambda settings: InteractionRuntime(
        settings, agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search], **kw)
    )
    app = application.create_app(
        Settings(api_key="synthetic-only", interaction_dir=".assistant-private/browser-fixture")
    )

    @app.post("/fixture/release")
    async def release():
        Path("release").touch()
        return {"ok": True}

    uvicorn.run(
        app, host="127.0.0.1", port=8769, log_level="warning", timeout_graceful_shutdown=5
    )


if __name__ == "__main__":
    main()
