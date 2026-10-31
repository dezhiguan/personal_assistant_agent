"""AMap HTTP contracts, GCJ-02 boundaries and server-only credentials."""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents import tool_http
from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.core.config import Settings, get_settings

KEY = "synthetic-amap-key"


def amap(key=KEY):
    return next(t for t in build_tools(Settings(amap_api_key=key)) if t.name == "amap_maps")


def invoke(**args):
    return asyncio.run(amap().ainvoke(args))


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        tool_http,
        "client",
        lambda timeout: original(
            transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False
        ),
    )


def test_keyword_search_is_encoded_city_limited_and_credential_free_in_results(monkeypatch):
    def respond(request):
        assert request.url.host == "restapi.amap.com"
        assert request.url.path == "/v5/place/text"
        assert request.url.params["key"] == KEY
        assert request.url.params["keywords"] == "合成 公园&测试"
        assert request.url.params["region"] == "北京市"
        assert request.url.params["city_limit"] == "true"
        assert request.url.params["page_size"] == "2"
        return httpx.Response(
            200,
            json={
                "status": "1",
                "infocode": "10000",
                "pois": [
                    {
                        "id": "test",
                        "name": "合成公园",
                        "location": "116.400000,39.900000",
                        "internal_key": KEY,
                    },
                    {"name": "第二项"},
                    {"name": "第三项"},
                ],
            },
        )

    transport(monkeypatch, respond)
    result = invoke(action="search", query="合成 公园&测试", city="北京市", max_results=2)
    assert len(result["results"]) == 2 and result["truncated"]
    assert result["coordinate_system"] == "GCJ-02" and result["untrusted_content"]
    assert KEY not in str(result) and "key" not in amap().args


@pytest.mark.parametrize(
    "action,arguments,path,parameters,response,field",
    [
        (
            "nearby",
            {"location": "116.4,39.9", "query": "咖啡", "radius": 500},
            "/v5/place/around",
            {"location": "116.400000,39.900000", "radius": "500"},
            {"pois": [{"name": "合成咖啡店", "distance": "20"}]},
            "results",
        ),
        (
            "geocode",
            {"query": "合成地址", "city": "北京市"},
            "/v3/geocode/geo",
            {"address": "合成地址", "city": "北京市"},
            {"geocodes": [{"formatted_address": "合成地址", "location": "116.4,39.9"}]},
            "results",
        ),
        (
            "reverse_geocode",
            {"location": "116.4,39.9"},
            "/v3/geocode/regeo",
            {"location": "116.400000,39.900000", "extensions": "base"},
            {
                "regeocode": {
                    "formatted_address": "合成地址",
                    "addressComponent": {"city": "北京市"},
                }
            },
            "address",
        ),
    ],
)
def test_places_and_geocoding_contracts(
    monkeypatch, action, arguments, path, parameters, response, field
):
    def respond(request):
        assert request.url.path == path
        assert all(request.url.params[k] == v for k, v in parameters.items())
        return httpx.Response(200, json={"status": "1", "infocode": "10000", **response})

    transport(monkeypatch, respond)
    result = invoke(action=action, **arguments)
    assert result["source"] == "AMap" and result[field]


@pytest.mark.parametrize("mode", ["driving", "walking", "bicycling"])
def test_routes_request_cost_and_preserve_meters_seconds_and_truncation(monkeypatch, mode):
    def respond(request):
        assert request.url.path == "/v5/direction/" + mode
        assert request.url.params["show_fields"] == "cost"
        assert request.url.params["origin"] == "116.400000,39.900000"
        return httpx.Response(
            200,
            json={
                "status": "1",
                "route": {
                    "paths": [
                        {
                            "distance": "1200",
                            **(
                                {"duration": "600"}
                                if mode == "bicycling"
                                else {"cost": {"duration": "600", "tolls": "0"}}
                            ),
                            "steps": [
                                {
                                    "instruction": "合成路段",
                                    "road_name": "合成路",
                                    "polyline": "omit",
                                }
                            ]
                            * 101,
                        }
                    ]
                },
            },
        )

    transport(monkeypatch, respond)
    result = invoke(action="route", origin="116.4,39.9", destination="116.41,39.91", mode=mode)
    route = result["routes"][0]
    assert route["distance_m"] == 1200 and route["duration_seconds"] == 600
    assert len(route["steps"]) == 100 and route["steps_truncated"]
    assert "polyline" not in route["steps"][0]


@pytest.mark.parametrize(
    "args",
    [
        {"action": "search", "query": " "},
        {"action": "geocode"},
        {"action": "search", "query": "x" * 81},
        {"action": "nearby", "location": "NaN,39.9"},
        {"action": "nearby", "location": "181,39.9"},
        {"action": "nearby", "location": "116,91"},
        {"action": "nearby", "location": "116.1234567,39.9"},
        {"action": "route", "origin": "北京", "destination": "上海"},
        {"action": "reverse_geocode", "location": "116.4,39.9", "radius": 3001},
        {"action": "search", "query": "公园", "max_results": 11},
    ],
)
def test_invalid_inputs_never_make_network_requests(monkeypatch, args):
    def fail(request):
        raise AssertionError("Invalid input made a network request")

    transport(monkeypatch, fail)
    assert "error" in invoke(**args)


def test_missing_key_and_business_errors_do_not_fake_success_or_echo_secrets(monkeypatch):
    def respond(request):
        return httpx.Response(
            200, json={"status": "0", "infocode": "10001", "info": "untrusted key=" + KEY}
        )

    transport(monkeypatch, respond)
    missing = asyncio.run(amap("").ainvoke({"action": "search", "query": "公园"}))
    assert "AMAP_API_KEY" in missing["error"]
    failed = invoke(action="search", query="公园")
    assert failed["code"] == "10001" and "results" not in failed and KEY not in str(failed)


def test_transport_error_is_redacted(monkeypatch):
    def fail(request):
        raise httpx.ReadTimeout("url=" + str(request.url), request=request)

    transport(monkeypatch, fail)
    result = invoke(action="geocode", query="合成地址")
    assert "error" in result and KEY not in str(result) and "restapi" not in str(result)


def test_env_key_is_read_only_server_side_and_hidden_from_repr(monkeypatch):
    monkeypatch.setenv("AMAP_API_KEY", KEY)
    settings = get_settings()
    assert settings.amap_api_key == KEY and KEY not in repr(settings)


def test_real_graph_executes_amap_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    transport(
        monkeypatch,
        lambda _: httpx.Response(
            200, json={"status": "1", "pois": [{"name": "合成公园", "location": "116.4,39.9"}]}
        ),
    )

    async def generate(self, messages, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        message = (
            AIMessage(content="查到了合成公园。")
            if results
            else AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "amap_maps",
                        "id": "maps",
                        "args": {"action": "search", "query": "合成公园"},
                    }
                ],
            )
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        graph = build_agent(Settings(api_key="synthetic-model-key", amap_api_key=KEY))
        result = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "查询公园"}]}))
    messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert messages[0].name == "amap_maps" and "合成公园" in messages[0].content
