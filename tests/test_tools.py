"""Deterministic server tools: Baidu HTTP substitute, a fixed clock and safe math."""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from personal_assistant_agent.agents.tools import (
    BAIDU_URL,
    build_tools,
    calculate,
    get_current_time,
    parse_baidu_results,
)
from personal_assistant_agent.core.config import Settings

BAIDU_HTML = """<div class="result c-container"><h3><a href="https://www.baidu.com/link?url=123">
示例<em>标题</em></a></h3><div class="c-abstract">真实摘要 &amp; 资料<script>ignore()</script></div>
</div><div class="c-container"><h3><a href="https://example.org/page">第二个结果</a></h3>
<div class="content-right_abc">另一段摘要</div></div>
<div class="result"><h3><a href="javascript:alert(1)">无效链接</a></h3></div>"""


def test_baidu_parser_extracts_titles_summaries_and_links():
    results = parse_baidu_results(BAIDU_HTML, 10)
    assert len(results) == 2
    assert results[0] == {
        "title": "示例 标题",
        "summary": "真实摘要 & 资料",
        "url": "https://www.baidu.com/link?url=123",
    }
    assert len(parse_baidu_results(BAIDU_HTML, 1)) == 1


def test_baidu_parser_skips_broken_urls_and_accepts_empty_attributes():
    html = (
        "<div class><h3>无链接</h3></div>"
        + ('<div class="result"><h3><a href="https://[invalid">坏链接</a></h3></div>')
        + BAIDU_HTML
    )
    assert len(parse_baidu_results(html, 10)) == 2


def baidu_with_transport(monkeypatch, handler):
    original = httpx.AsyncClient
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("personal_assistant_agent.agents.tools.httpx.AsyncClient", client)
    return build_tools(Settings())[0], options


def test_baidu_only_requests_baidu_with_encoded_keywords(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text=BAIDU_HTML)

    tool, options = baidu_with_transport(monkeypatch, respond)
    result = asyncio.run(tool.ainvoke({"query": "中文 & agent", "max_results": 2}))
    assert len(result["results"]) == 2
    assert result["source"] == "baidu"
    assert len(requests) == 1
    assert str(requests[0].url.copy_with(query=None)) == BAIDU_URL
    assert requests[0].url.params["wd"] == "中文 & agent"
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False


@pytest.mark.parametrize(
    "status,html",
    [(403, "denied"), (200, "百度安全验证"), (302, "redirect"), (200, "<html>无结果</html>")],
)
def test_baidu_failure_reports_no_fabricated_results_or_fallback(monkeypatch, status, html):
    tool, _ = baidu_with_transport(monkeypatch, lambda _: httpx.Response(status, text=html))
    result = asyncio.run(tool.ainvoke({"query": "test"}))
    assert "error" in result
    assert not result.get("results")


def test_baidu_timeout_does_not_leak_proxy_credentials(monkeypatch):
    def fail(request):
        raise httpx.ReadTimeout("proxy password=must-not-leak", request=request)

    tool, _ = baidu_with_transport(monkeypatch, fail)
    result = asyncio.run(tool.ainvoke({"query": "test"}))
    assert "error" in result
    assert "must-not-leak" not in str(result)


def test_baidu_captcha_redirect_is_reported_without_following_it(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            302, headers={"Location": "https://wappass.baidu.com/static/captcha/tuxing_v2.html"}
        )

    tool, _ = baidu_with_transport(monkeypatch, respond)
    result = asyncio.run(tool.ainvoke({"query": "test"}))
    assert result["code"] == "BAIDU_CHALLENGE"
    assert "验证码" in result["error"]
    assert not result.get("results")
    assert len(requests) == 1


def test_baidu_input_and_response_bounds(monkeypatch):
    tool, _ = baidu_with_transport(
        monkeypatch, lambda _: httpx.Response(200, content=b"x" * 2097153)
    )
    assert "error" in asyncio.run(tool.ainvoke({"query": "test"}))
    assert "error" in asyncio.run(tool.ainvoke({"query": " "}))
    assert "error" in asyncio.run(tool.ainvoke({"query": "test", "max_results": 11}))


@pytest.mark.parametrize(
    "settings",
    [
        Settings(baidu_timeout=0),
        Settings(baidu_timeout=61),
        Settings(baidu_proxy="file:///tmp/proxy"),
        Settings(baidu_proxy="not-a-proxy"),
    ],
)
def test_search_config_is_validated(settings):
    with pytest.raises(ValueError):
        build_tools(settings)


def test_clock_uses_requested_timezone_and_week_boundary(monkeypatch):
    fixed = datetime(2026, 10, 25, 17, 30, tzinfo=ZoneInfo("UTC"))

    class Clock:
        @staticmethod
        def now(zone):
            return fixed.astimezone(zone)

    monkeypatch.setattr("personal_assistant_agent.agents.tools.datetime", Clock)
    china = get_current_time.invoke({})
    utc = get_current_time.invoke({"timezone": "UTC"})
    assert china["datetime"] == "2026-10-26T01:30:00+08:00"
    assert china["week_start"] == "2026-10-26"
    assert china["week_end"] == "2026-11-01"
    assert utc["date"] == "2026-10-25"
    assert utc["week_start"] == "2026-10-19"
    assert "error" in get_current_time.invoke({"timezone": "../../etc/passwd"})
    assert "error" in get_current_time.invoke({"timezone": "Invalid/Zone"})


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+3*4", 14),
        ("200*10%", 20),
        ("(50+25)%", 0.75),
        ("sqrt(4)%", 0.02),
        ("200+10%", 200.1),
        ("2^10", 1024),
        ("sin(pi/2)", 1),
        ("round(log10(1000))", 3),
        ("max(1,2,3) + abs(-4)", 7),
        ("1e2 * 5%", 5),
        ("9//2", 4),
    ],
)
def test_calculator_math(expression, expected):
    assert calculate.invoke({"expression": expression})["result"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('pwd')",
        "open('/etc/passwd').read()",
        "(1).__class__",
        "[x for x in range(5)]",
        "lambda:1",
        "True+1",
        "1/0",
        "sqrt(-1)",
        "1e999",
        "9**999999999",
        "10**(10**10)",
        "'x'*100",
        "round(1, ndigits=2)",
        "",
        "1+" * 300,
        "2%3",
        "min(*[1,2])",
        "sin(1,2)",
        "1" * 512,
    ],
)
def test_calculator_rejects_code_invalid_math_and_resource_abuse(expression):
    assert "error" in calculate.invoke({"expression": expression})
