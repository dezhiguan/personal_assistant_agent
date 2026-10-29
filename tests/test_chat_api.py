"""HTTP behavior, input bounds, isolation and safe streaming errors."""

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from personal_assistant_agent.app import create_app
from personal_assistant_agent.core.config import Settings


def payload(content="你好"):
    return {
        "conversation_id": str(uuid4()),
        "messages": [{"role": "user", "content": content}],
    }


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_home_and_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/static/js/app.js" in response.text
    assert client.get("/static/js/app.js").status_code == 200
    status = client.get("/healthz").json()
    assert status["model_configured"] is True
    assert status["tracing_enabled"] is False
    assert "unit-test-only" not in json.dumps(status)


def test_stream_and_explicit_history(client, graph):
    data = payload("请记住我叫小林")
    data["messages"] += [
        {"role": "assistant", "content": "好的，小林"},
        {"role": "user", "content": "我叫什么？"},
    ]
    response = client.post("/api/chat", json=data)
    assert response.status_code == 200
    streamed = events(response)
    assert [x["type"] for x in streamed] == ["meta", "delta", "delta", "done"]
    assert "".join(x["text"] for x in streamed if x["type"] == "delta") == "你好，这是测试回复。"
    value, config = graph.calls[0]
    assert value["messages"] == data["messages"]
    assert config["metadata"]["conversation_id"] == data["conversation_id"]
    assert str(config["run_id"]) == response.headers["X-Request-ID"]


def test_conversations_do_not_share_history(client, graph):
    first, second = payload("A 的私人上下文"), payload("B 的独立问题")
    client.post("/api/chat", json=first)
    client.post("/api/chat", json=second)
    assert graph.calls[1][0]["messages"] == second["messages"]
    assert first["messages"][0]["content"] not in str(graph.calls[1])


def test_follow_up_plan_is_restored_and_isolated(client, graph):
    first = payload("继续上次的计划")
    first["todos"] = [{"content": "完成初稿", "status": "in_progress"}]
    assert client.post("/api/chat", json=first).status_code == 200
    assert graph.calls[0][0]["todos"] == first["todos"]
    client.post("/api/chat", json=payload("新任务"))
    assert graph.calls[1][0]["todos"] == []


@pytest.mark.parametrize(
    "todos",
    [
        [{"content": "任务", "status": "made_up"}],
        [{"content": "  ", "status": "pending"}],
        [{"content": "x" * 501, "status": "pending"}],
        [{"content": "任务", "status": "pending", "tool": "execute"}],
        [{"content": "任务", "status": "pending"}] * 33,
    ],
)
def test_rejects_invalid_or_oversized_plans(client, todos):
    data = payload()
    data["todos"] = todos
    assert client.post("/api/chat", json=data).status_code == 422


@pytest.mark.parametrize("text", ["", "  ", "a" * 8001])
def test_invalid_message_content(client, text):
    assert client.post("/api/chat", json=payload(text)).status_code == 422


@pytest.mark.parametrize("role", ["system", "tool", "assistant"])
def test_rejects_untrusted_or_final_assistant_role(client, role):
    data = payload()
    data["messages"][0]["role"] = role
    assert client.post("/api/chat", json=data).status_code == 422


def test_history_limits(client):
    data = payload()
    data["messages"] *= 61
    assert client.post("/api/chat", json=data).status_code == 422
    data["messages"] = [{"role": "user", "content": "a" * 8000}] * 11
    assert client.post("/api/chat", json=data).status_code == 422


def test_invalid_conversation_identifier(client):
    data = payload()
    data["conversation_id"] = "not-a-uuid"
    assert client.post("/api/chat", json=data).status_code == 422


def test_missing_model_is_explicit():
    with TestClient(create_app(Settings())) as client:
        assert client.get("/healthz").json()["model_configured"] is False
        assert client.post("/api/chat", json=payload()).status_code == 503


def test_provider_errors_never_expose_secrets(client, graph):
    graph.error = RuntimeError("upstream password=NEVER-EXPOSE-THIS")
    response = client.post("/api/chat", json=payload())
    assert "NEVER-EXPOSE-THIS" not in response.text
    assert events(response)[-1]["type"] == "error"
    assert events(response)[-1]["code"] == "MODEL_ERROR"
