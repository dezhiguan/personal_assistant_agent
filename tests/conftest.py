"""Deterministic API tests use a graph substitute, never a paid model."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langsmith import tracing_context

from personal_assistant_agent.app import create_app
from personal_assistant_agent.core.config import Settings


class RecordingGraph:
    def __init__(self):
        self.calls = []
        self.error = None

    async def astream(self, value, *, config, stream_mode):
        self.calls.append((value, config))
        if self.error:
            raise self.error
        yield "messages", (SimpleNamespace(content="你好，"), {"langgraph_node": "model"})
        yield "messages", (SimpleNamespace(content="这是测试回复。"), {"langgraph_node": "model"})


@pytest.fixture(autouse=True)
def disable_external_tracing():
    with tracing_context(enabled=False):
        yield


@pytest.fixture
def graph():
    return RecordingGraph()


@pytest.fixture
def client(monkeypatch, graph):
    monkeypatch.setattr("personal_assistant_agent.app.build_agent", lambda _: graph)
    with TestClient(create_app(Settings(api_key="unit-test-only"))) as client:
        yield client
