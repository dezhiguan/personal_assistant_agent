"""A managed-runtime factory; graph state stays isolated per Agent Server run."""

import asyncio
import atexit
from functools import lru_cache

import truststore
from langsmith import Client

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import get_settings
from personal_assistant_agent.observability.tracer import CostTracer


@lru_cache(maxsize=1)
def _graph():
    truststore.inject_into_ssl()
    settings = get_settings()
    graph = build_agent(settings)
    callbacks = []
    if settings.tracing_configured:
        client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
        atexit.register(client.close)
        callbacks.append(
            CostTracer(settings=settings, client=client, project_name=settings.langsmith_project)
        )
    return graph.with_config(
        {
            "callbacks": callbacks,
            "metadata": {
                "prompt_ref": settings.prompt_ref or "local",
                "runtime": "langsmith-agent-server",
            },
        }
    )


async def make_graph(config):
    # Prompt retrieval is synchronous SDK I/O and must not block Agent Server's loop.
    return await asyncio.to_thread(_graph)
