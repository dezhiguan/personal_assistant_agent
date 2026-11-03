"""Isolated browser fault/barrier fixture; no .env and no real model calls.

Run from a disposable directory with PYTHONPATH pointing at the repository src.
All task/chat/control submissions must still use the page's ordinary input.
"""

import asyncio
import json
import re
from pathlib import Path

import httpx
import uvicorn
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from openai import APITimeoutError

import personal_assistant_agent.app as application
from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.runtime import InteractionRuntime

runtime = None


@tool
async def baidu_search(query: str) -> str:
    """Read synthetic evidence, with an explicit file barrier."""
    with Path("reads").open("a") as output:
        output.write(query + "\n")
    Path("tool-started").touch()
    while not Path("release").exists():
        await asyncio.sleep(0.02)
    return "合成资料：国内科技报道。"


@tool
async def slow_write(content: str) -> str:
    """Write a synthetic receipt once, then wait for settlement."""
    with Path("effect-ledger").open("a") as output:
        output.write(content + "\n")
    Path("effect-started").touch()
    while not Path("effect-release").exists():
        await asyncio.sleep(0.02)
    return "合成写入已结算。"


async def generate(self, messages, run_manager=None, **kwargs):
    latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
    if str(messages[0].content).startswith("你是消息路由器"):
        packet = json.loads(latest)
        text = packet["message"]
        tasks = packet["tasks"]
        found = next((t for t in tasks if t["id"][:8] in text), None)
        control = re.search(r"暂停|继续|恢复|取消", text)
        if control and found:
            intent = {"暂停": "PAUSE", "继续": "RESUME", "恢复": "RESUME", "取消": "CANCEL"}[
                control[0]
            ]
            decision = {"intent": intent, "target_task_id": found["id"]}
        elif "新建" in text or not tasks:
            decision = {
                "intent": "NEW_TASK",
                "goal": "合成写入" if "写入" in text else "搜索新闻",
                "constraints": {"验收": text},
            }
        else:
            decision = {"intent": "CHAT"}
        yield ChatGenerationChunk(message=AIMessageChunk(content=json.dumps(decision)))
        return
    metadata = run_manager.metadata if run_manager else {}
    metadata = {**get_config().get("metadata", {}), **metadata}
    tid = metadata.get("task_id")
    task = runtime.store.task(tid) if tid else None
    results = [m for m in messages if isinstance(m, ToolMessage)]
    if not tid:
        text, calls = "这是独立的合成聊天回复。", []
    elif not results:
        name = "slow_write" if "合成写入" in latest else "baidu_search"
        args = {"content": tid} if name == "slow_write" else {"query": latest}
        text, calls = "", [{"name": name, "id": "operation-" + tid, "args": args}]
    else:
        if task["attempt"] == 1 and "上下文恢复" in latest:
            raise ContextOverflowError("Injected context overflow at a settled model boundary")
        if task["attempt"] == 1 and "超时恢复" in latest:
            raise APITimeoutError(request=httpx.Request("POST", "https://synthetic.invalid"))
        if Path("delivery-hold").exists() and not Path("delivery-release").exists():
            Path("delivery-started-" + tid).touch()
            while not Path("delivery-release").exists():
                await asyncio.sleep(0.02)
        text, calls = "合成最终交付，采用要求：\n" + latest, []
    yield ChatGenerationChunk(message=AIMessageChunk(content=text, tool_calls=calls))


def main():
    global runtime
    ChatOpenAI._astream = generate

    async def nonstream(self, messages, **kwargs):
        chunks = [chunk async for chunk in generate(self, messages, **kwargs)]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="".join(chunk.message.content for chunk in chunks))
                )
            ]
        )

    ChatOpenAI._agenerate = nonstream

    def factory(settings):
        global runtime
        runtime = InteractionRuntime(
            settings,
            agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search, slow_write], **kw),
        )
        return runtime

    application.InteractionRuntime = factory
    uvicorn.run(
        application.create_app(
            Settings(
                api_key="synthetic-only",
                interaction_dir=".assistant-private/fixture",
                tracing_enabled=False,
            )
        ),
        host="127.0.0.1",
        port=8769,
        log_level="warning",
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
