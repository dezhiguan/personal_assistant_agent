"""Provider contracts, bounded external data and calendar persistence."""

import asyncio
import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from personal_assistant_agent.agents import tool_http
from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.app import create_app
from personal_assistant_agent.core.config import Settings


def tools():
    return {t.name: t for t in build_tools(Settings())}


def invoke(name, **args):
    return asyncio.run(tools()[name].ainvoke(args))


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        tool_http,
        "client",
        lambda timeout: original(
            transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False
        ),
    )


@pytest.mark.parametrize("host", ["synthetic.qweatherapi.com", "synthetic.xy.qweatherapi.com"])
def test_geocode_weather_and_qweather_use_current_provider_contracts(monkeypatch, host):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["countryCode"] == "CN"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "合成城市", "latitude": 30, "longitude": 120, "country_code": "CN"}
                    ]
                },
            )
        if request.url.host == "api.open-meteo.com":
            assert request.url.params["forecast_days"] == "2"
            return httpx.Response(
                200,
                json={
                    "current": {"temperature_2m": 20},
                    "daily": {},
                    "current_units": {},
                    "daily_units": {},
                    "timezone": "Asia/Shanghai",
                },
            )
        assert request.headers["X-QW-Api-Key"] == "synthetic-key"
        if request.url.path == "/geo/v2/city/lookup":
            assert request.url.params["range"] == "cn"
            return httpx.Response(
                200,
                json={
                    "code": "200",
                    "location": [
                        {
                            "id": "synthetic-location",
                            "name": "合成城市",
                            "lat": "30.12345",
                            "lon": "120.56789",
                        }
                    ],
                },
            )
        assert request.url.path == "/weatheralert/v1/current/30.12/120.57"
        return httpx.Response(200, json={"alerts": [], "metadata": {"zeroResult": True}})

    transport(monkeypatch, respond)
    weather = invoke("get_city_weather", city="合成城市", country_code="CN", days=2)
    assert weather["current"]["temperature_2m"] == 20
    configured = {
        t.name: t
        for t in build_tools(Settings(qweather_host=host, qweather_api_key="synthetic-key"))
    }
    alert = asyncio.run(
        configured["get_weather_alert"].ainvoke({"city": "合成城市", "country_code": "CN"})
    )
    assert alert["alerts"] == [] and alert["metadata"]["zeroResult"]
    assert len(requests) == 4
    assert requests[-2].url.host == host


def test_holidays_distinguish_makeup_weekend_and_unsupported_year():
    assert invoke("check_cn_holiday", date_str="2026-10-01")["is_holiday_break"]
    assert invoke("check_cn_holiday", date_str="2026-10-10")["is_makeup_workday"]
    assert invoke("check_cn_holiday", date_str="2026-11-01")["is_day_off"]
    assert "error" in invoke("check_cn_holiday", date_str="2100-10-01")


def test_currency_uses_provider_date_and_decimal_arithmetic(monkeypatch):
    def respond(request):
        assert request.url.params == httpx.QueryParams(
            {"base": "USD", "quotes": "CNY", "date": "2026-10-01"}
        )
        return httpx.Response(200, json=[{"date": "2026-09-30", "rate": 7.1234}])

    transport(monkeypatch, respond)
    result = invoke(
        "convert_currency",
        amount=100,
        from_currency="usd",
        to_currency="cny",
        date_str="2026-10-01",
    )
    assert result["result"] == "712.34000" and result["date"] == "2026-09-30"


@pytest.mark.parametrize(
    "value,source,target,expected",
    [
        (1, "km", "m", 1000),
        (0, "degC", "degF", 32),
        (36, "km/hour", "m/s", 10),
    ],
)
def test_conversions(value, source, target, expected):
    assert invoke("convert_units", value=value, from_unit=source, to_unit=target)[
        "result"
    ] == pytest.approx(expected)


@pytest.mark.parametrize(
    "source,target", [("kg", "m"), ("m**999**999", "m"), ("__import__('os')", "m")]
)
def test_conversions_reject_incompatible_or_unbounded_input(source, target):
    assert "error" in invoke("convert_units", value=1, from_unit=source, to_unit=target)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://[::1]",
        "http://169.254.169.254/",
        "file:///etc/passwd",
        "https://user:pass@example.com",
        "http://localhost:8000",
        "http://2130706433",
    ],
)
def test_fetch_rejects_local_and_non_http_urls(url):
    assert "error" in invoke("fetch_url", url=url)


def test_redirect_to_private_address_never_connects_and_dns_is_pinned(monkeypatch):
    requests = []
    original = asyncio.BaseEventLoop.getaddrinfo

    async def dns(self, host, port, **kwargs):
        if host == "example.org":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        return await original(self, host, port, **kwargs)

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", dns)

    def respond(request):
        requests.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.org"
        assert request.extensions["sni_hostname"] == "example.org"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    transport(monkeypatch, respond)
    assert "error" in invoke("fetch_url", url="https://example.org")
    assert len(requests) == 1


def test_fetch_text_rss_atom_and_entity_rejection(monkeypatch):
    async def fetch(url, timeout):
        documents = {
            "html": b"<p>Hello</p><script>steal()</script><p>World</p>",
            "rss": b"<rss><channel><item><title>News</title><link>https://example.com/</link>"
            b"<description>&lt;p&gt;Summary&lt;/p&gt;</description></item></channel></rss>",
            "atom": b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom</title>'
            b'<link href="https://example.com/"/><summary>Text</summary></entry></feed>',
            "entity": b'<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b"<rss><channel><item><title>&xxe;</title></item></channel></rss>",
        }
        return documents[url], "text/html; charset=utf-8", "https://example.com/"

    monkeypatch.setattr("personal_assistant_agent.agents.life_tools.fetch_public", fetch)
    assert invoke("fetch_url", url="html", max_chars=5)["text"] == "Hello"
    assert invoke("fetch_url", url="html", max_chars=5)["truncated"]
    assert invoke("fetch_rss", url="rss")["items"][0]["summary"] == "Summary"
    assert invoke("fetch_rss", url="atom")["items"][0]["title"] == "Atom"
    assert "error" in invoke("fetch_rss", url="entity")


def test_html_parser_rejects_pathological_nesting():
    from personal_assistant_agent.agents.life_tools import plain_text
    from personal_assistant_agent.agents.tool_http import ToolError

    with pytest.raises(ToolError, match="嵌套过深"):
        plain_text("<div>" * 10000 + "</invalid>" * 10000)


def test_external_errors_are_redacted_and_responses_bounded(monkeypatch):
    transport(monkeypatch, lambda r: httpx.Response(200, content=b"x" * (2097152 + 1)))
    assert "2 MiB" in invoke("geocode", city="test")["error"]

    def fail(request):
        raise httpx.ConnectError("secret-must-not-leak", request=request)

    transport(monkeypatch, fail)
    result = invoke("translate", text="Hello", source_language="en", target_language="zh")
    assert "error" in result and "secret-must-not-leak" not in str(result)


def test_translation_splits_utf8_and_route_preserves_real_units(monkeypatch):
    chunks = []

    def respond(request):
        if request.url.host == "api.mymemory.translated.net":
            chunk = request.url.params["q"]
            chunks.append(chunk)
            assert len(chunk.encode()) <= 500
            return httpx.Response(
                200, json={"responseStatus": 200, "responseData": {"translatedText": chunk}}
            )
        assert request.url.path.endswith("120.0,30.0;121.0,31.0")
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "routes": [{"distance": 12000, "duration": 600, "legs": [{"steps": []}]}],
            },
        )

    transport(monkeypatch, respond)
    text = "中文" * 200
    assert (
        invoke("translate", text=text, source_language="zh", target_language="en")["translation"]
        == text
    )
    assert len(chunks) == 3
    route = invoke(
        "route_plan",
        origin_latitude=30,
        origin_longitude=120,
        destination_latitude=31,
        destination_longitude=121,
    )
    assert route["distance_km"] == 12 and route["duration_minutes"] == 10


def test_calendar_alarm_download_and_injection_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = invoke(
        "create_calendar_event",
        title="合成测试" * 30 + "\nBEGIN:VALARM",
        start="2030-01-01T09:00:00",
        end="2030-01-01T10:00:00",
        reminder_minutes=15,
    )
    assert result["status"] == "pending_import"
    raw = (tmp_path / "workspace" / result["file_path"].lstrip("/")).read_bytes()
    assert b"DTSTART:20300101T010000Z\r\n" in raw
    assert b"TRIGGER:-PT15M\r\n" in raw
    assert raw.count(b"\r\nBEGIN:VALARM\r\n") == 1
    assert all(len(line) <= 75 for line in raw.split(b"\r\n"))
    assert b"\\nBEGIN:VALARM" in raw.replace(b"\r\n ", b"")
    with TestClient(create_app(Settings())) as client:
        response = client.get(result["download_url"])
        assert response.status_code == 200 and "BEGIN:VCALENDAR" in response.text
        assert client.get("/api/calendar/invalid.ics").status_code == 404
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    assert invoke("set_reminder", title="测试", remind_at=future)["reminder_minutes"] == 0
    assert "error" in invoke("set_reminder", title="过去", remind_at="2000-01-01T00:00:00Z")
    assert "error" in invoke(
        "create_calendar_event",
        title="DST",
        start="2030-03-10T02:30",
        end="2030-03-10T04:00",
        timezone="America/New_York",
    )


def test_missing_provider_config_never_looks_like_no_alerts():
    result = invoke("get_weather_alert", city="北京")
    assert "error" in result and "alerts" not in result
