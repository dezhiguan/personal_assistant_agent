"""Synthetic subprocess fixture: real graph and two on-disk databases, no network."""

import asyncio
import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.middleware import ControlMiddleware
from personal_assistant_agent.interaction.runtime import InteractionRuntime


async def main():
    mode = sys.argv[1]
    root = Path.cwd()
    settings = Settings(api_key="synthetic-only", interaction_dir=str(root / "private"))

    @tool
    async def baidu_search(query: str) -> str:
        """Synthetic read tool; never uses a network transport."""
        with (root / "reads").open("a") as out:
            out.write("read\n")
        (root / "tool-started").touch()
        while not (root / "release").exists():
            await asyncio.sleep(0.02)
        return "合成国内新闻资料"

    @tool
    async def fixture_write() -> str:
        """Synthetic supplier effect with a durable ledger outside the agent workspace."""
        with (root / "supplier-ledger").open("a") as out:
            out.write("effect\n")
            out.flush()
            os.fsync(out.fileno())
        (root / "tool-started").touch()
        while not (root / "release").exists():
            await asyncio.sleep(0.02)
        return "已确认的合成回执"

    async def generate(self, messages, **kwargs):
        latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        calls = (
            []
            if results
            else [
                {
                    "id": "stable-call",
                    "name": ("fixture_write" if "effect" in mode else "baidu_search"),
                    "args": {} if "effect" in mode else {"query": "合成新闻"},
                }
            ]
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="合成任务已交付。" + latest if results else "", tool_calls=calls
            )
        )

    ChatOpenAI._astream = generate
    runtime = InteractionRuntime(
        settings,
        agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search, fixture_write], **kw),
    )
    await runtime.start()
    # Inject hard death after an effect result is durable but before ToolNode checkpoint.
    if mode == "effect-gap":
        original = ControlMiddleware.call_tool

        async def gap(self, request, handler):
            result = await original(self, request, handler)
            (root / "effect-settled-gap").touch()
            os._exit(37)
            return result

        ControlMiddleware.call_tool = gap
    if mode == "applied-gap":

        def ack_gap(task_id, revision, checkpoint, effective_request=None):
            if revision >= 1:
                (root / "applied-gap").touch()
                os._exit(38)

        runtime.store.applied = ack_gap
    if mode == "result-gap":

        def result_gap(task_id, revision, result):
            (root / "result-gap").touch()
            os._exit(39)

        runtime.store.finish = result_gap
    if not runtime.store.tasks():
        receipt = runtime.store.receive(
            "synthetic-conversation",
            "synthetic-turn",
            "读取资料",
            {"intent": "NEW_TASK", "goal": "读取资料"},
            {},
        )
        if mode in {"queued-gap", "applied-gap"}:
            runtime.store.control(
                receipt["task_id"],
                "synthetic-conversation",
                "stable-steer",
                "STEER",
                changes={"范围": "中国大陆"},
            )
        if mode == "queued-gap":
            (root / "queued-gap").touch()
            os._exit(36)
        runtime.wakeup.set()
    (root / "ready").touch()
    try:
        while not (root / "stop").exists():
            await asyncio.sleep(0.02)
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
