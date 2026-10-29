"""The authenticated hosting adapter starts chat without intercepting system APIs."""

from fastapi.testclient import TestClient

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
