"""DNS failover keeps public-IP, TLS and HTTP-denial boundaries intact."""

import asyncio
import socket

import httpx
import pytest

from personal_assistant_agent.agents import tool_http
from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.core.config import Settings


def setup_network(
    monkeypatch, *, answers=("93.184.216.35",), primary=("93.184.216.34",), status=None
):
    calls = []

    async def dns(self, host, port, **kwargs):
        addresses = primary if host == "example.org" else ("8.8.8.8",)
        assert host in {"example.org", "dns.google", "cloudflare-dns.com"}
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in addresses]

    def respond(request):
        calls.append(request)
        if request.url.host == "93.184.216.34":
            if status:
                return httpx.Response(status, text="SYNTHETIC-DENIAL")
            raise httpx.ConnectError("SYNTHETIC-UNREACHABLE", request=request)
        if request.url.host == "8.8.8.8":
            assert request.headers["host"] in {"dns.google", "cloudflare-dns.com"}
            assert request.extensions["sni_hostname"] == request.headers["host"]
            assert request.headers["accept"] == "application/dns-json"
            assert dict(request.url.params) == {"name": "example.org", "type": "A"}
            return httpx.Response(
                200, json={"Status": 0, "Answer": [{"type": 1, "data": ip} for ip in answers]}
            )
        assert request.url.host == "93.184.216.35"
        assert request.headers["host"] == "example.org"
        assert request.extensions["sni_hostname"] == "example.org"
        return httpx.Response(
            200, content=b"<rss><channel><item><title>Recovered</title></item></channel></rss>"
        )

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", dns)
    monkeypatch.setattr(
        tool_http,
        "client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    return calls


def fetch(resolver="google"):
    tools = build_tools(Settings(dns_fallback=resolver))
    rss = next(t for t in tools if t.name == "fetch_rss")
    return asyncio.run(rss.coroutine(url="https://example.org/feed.xml", max_items=3))


@pytest.mark.parametrize("resolver", ["google", "cloudflare"])
def test_connection_failure_uses_validated_dns_answers_with_original_tls_name(
    monkeypatch, resolver
):
    calls = setup_network(monkeypatch)
    result = fetch(resolver)
    assert result["items"][0]["title"] == "Recovered"
    assert result["url"] == "https://example.org/feed.xml"
    assert [r.url.host for r in calls] == ["93.184.216.34", "8.8.8.8", "93.184.216.35"]


@pytest.mark.parametrize("private", ["127.0.0.1", "169.254.169.254"])
def test_private_fallback_answers_are_rejected_before_any_target_connection(monkeypatch, private):
    calls = setup_network(monkeypatch, answers=("93.184.216.35", private))
    assert "公网" in fetch()["error"]
    assert [r.url.host for r in calls] == ["93.184.216.34", "8.8.8.8"]


def test_http_denial_does_not_trigger_dns_fallback(monkeypatch):
    calls = setup_network(monkeypatch, status=403)
    assert "error" in fetch()
    assert len(calls) == 1


def test_fallback_is_opt_in(monkeypatch):
    calls = setup_network(monkeypatch)
    assert "error" in fetch("")
    assert len(calls) == 1


def test_private_primary_dns_answer_is_not_overridden_by_fallback(monkeypatch):
    calls = setup_network(monkeypatch, primary=("93.184.216.34", "127.0.0.1"))
    assert "公网" in fetch()["error"]
    assert calls == []


def test_custom_resolver_endpoints_are_not_allowed(monkeypatch):
    calls = setup_network(monkeypatch)
    assert "备用 DNS" in fetch("http://127.0.0.1/resolve")["error"]
    assert calls == []
