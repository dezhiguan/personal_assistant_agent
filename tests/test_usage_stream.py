"""Usage events sum all model calls without inventing data for missing usage."""

import asyncio
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService


def test_usage_covers_multiple_planning_calls_and_partial_chunks():
    class Graph:
        async def astream(self, *args, **kwargs):
            for call, inputs, outputs in (
                ("first", 70000, 10),
                ("first", 0, 90),
                ("next", 70000, 100),
            ):
                yield (
                    "messages",
                    (
                        AIMessageChunk(
                            id=call,
                            content="结果" if call == "next" else "",
                            usage_metadata={
                                "input_tokens": inputs,
                                "output_tokens": outputs,
                                "total_tokens": inputs + outputs,
                            },
                        ),
                        {"langgraph_node": "model"},
                    ),
                )

    async def check():
        request = ChatRequest(
            conversation_id=uuid4(), messages=[{"role": "user", "content": "计划"}]
        )
        return [e async for e in ChatService(Graph(), Settings()).stream(request, str(uuid4()))]

    events = asyncio.run(check())
    usage = next(e for e in events if e["type"] == "usage")
    assert usage["total_tokens"] == 140200
    assert usage["cost_estimate"]["model_calls"] == 2
    assert usage["cost_estimate"]["total_cost_cny"] == pytest.approx(0.1124)
    assert events[-1]["type"] == "done"
