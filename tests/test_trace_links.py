"""Trace redirects use the reply's run metadata and keep failures private."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langsmith.utils import LangSmithNotFoundError


def tracing_client(client):
    run = SimpleNamespace(id=uuid4(), session_id=uuid4(), trace_id=uuid4())
    url = f"https://smith.langchain.com/o/test/projects/p/{run.session_id}/r/{run.id}"
    traced = SimpleNamespace(
        read_run=Mock(return_value=run),
        runs=SimpleNamespace(get_url=AsyncMock(return_value=SimpleNamespace(url=url))),
        flush=Mock(),
        close=Mock(),
    )
    client.app.state.chat_service.trace_client = traced
    return traced, run, url


def test_trace_redirect_resolves_the_actual_run_project(client):
    traced, run, url = tracing_client(client)
    request_id = uuid4()
    response = client.get(f"/api/traces/{request_id}", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == url
    assert response.headers["cache-control"] == "no-store"
    traced.read_run.assert_called_once_with(request_id)
    traced.runs.get_url.assert_awaited_once_with(
        str(run.id), project_id=str(run.session_id), trace_id=str(run.trace_id)
    )


def test_trace_link_when_tracing_is_disabled(client):
    response = client.get(f"/api/traces/{uuid4()}")
    assert response.status_code == 503
    assert "未开启" in response.json()["detail"]


def test_trace_link_rejects_invalid_request_ids(client):
    traced, _, _ = tracing_client(client)
    assert client.get("/api/traces/not-a-uuid").status_code == 422
    traced.read_run.assert_not_called()


@pytest.mark.parametrize("error", [LangSmithNotFoundError("missing"), RuntimeError("SECRET")])
def test_trace_lookup_errors_never_expose_provider_details(client, error):
    traced, _, _ = tracing_client(client)
    traced.read_run.side_effect = error
    response = client.get(f"/api/traces/{uuid4()}")
    assert response.status_code == (404 if isinstance(error, LangSmithNotFoundError) else 503)
    assert "SECRET" not in response.text


@pytest.mark.parametrize("url", [None, "javascript:alert(1)"])
def test_trace_link_rejects_missing_or_invalid_destinations(client, url):
    traced, _, _ = tracing_client(client)
    traced.runs.get_url.return_value = SimpleNamespace(url=url)
    response = client.get(f"/api/traces/{uuid4()}", follow_redirects=False)
    assert response.status_code == 503
    assert "location" not in response.headers
