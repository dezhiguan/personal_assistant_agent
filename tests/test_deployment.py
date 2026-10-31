"""The authenticated hosting adapter starts chat without intercepting system APIs."""

from fastapi.testclient import TestClient

from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.app import create_app
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.deployment.web import create_deployment_app


def test_deployment_adapter_lifespan_routes_and_no_catchall():
    app = create_deployment_app(create_app(Settings()))
    assert all(hasattr(route, "app") for route in app.routes)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/js/app.js").status_code == 200
        assert client.get("/healthz").json()["model_configured"] is False
        assert client.post("/api/chat", json={}).status_code == 422
        assert client.post("/runs/wait").status_code == 404


def test_deployed_calendar_download_reaches_chat_app(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.chdir(tmp_path)
    tools = {t.name: t for t in build_tools(Settings())}
    event = asyncio.run(
        tools["create_calendar_event"].ainvoke(
            {"title": "Synthetic event", "start": "2030-01-01T09:00", "end": "2030-01-01T10:00"}
        )
    )
    with TestClient(create_deployment_app(create_app(Settings()))) as client:
        response = client.get(event["download_url"])
        assert response.status_code == 200
        assert "BEGIN:VEVENT" in response.text
