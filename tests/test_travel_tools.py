"""Synthetic provider fixtures verify confirmation and duplicate-order boundaries."""

import asyncio
import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.agents.travel_tools import Amadeus, TravelStore
from personal_assistant_agent.core.config import Settings


def runtime(owner="synthetic-conversation", confirmation=""):
    return SimpleNamespace(
        config={
            "configurable": {
                "travel_context": {"conversation_id": owner, "confirmation": confirmation}
            }
        }
    )


@pytest.fixture
def travel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    store = TravelStore(settings)
    profiles = tmp_path / ".assistant-private/profiles"
    profiles.mkdir(parents=True, mode=0o700)
    profile = profiles / "synthetic.json"
    profile.write_text(
        json.dumps(
            {
                "flight": {
                    "travelers": [{"id": "1", "name": {"firstName": "TEST", "lastName": "ONLY"}}],
                    "contacts": [{"emailAddress": "nobody@example.invalid"}],
                },
                "hotel": {
                    "guests": [{"id": 1, "name": {"firstName": "TEST", "lastName": "ONLY"}}],
                    "payments": [{"method": "synthetic-test-only"}],
                },
            }
        )
    )
    profile.chmod(0o600)
    offers = {
        "flight": {
            "id": "F1",
            "price": {"total": "123.00", "currency": "CNY"},
            "itineraries": [{"segments": []}],
            "travelerPricings": [],
        },
        "hotel": {
            "id": "H1",
            "price": {"total": "300.00", "currency": "CNY"},
            "policies": {"paymentType": "guarantee"},
        },
    }
    submitted = []
    fault = {"post": False}

    async def call(self, method, path, **kwargs):
        if path.endswith("/pricing"):
            return {"flightOffers": [copy.deepcopy(offers["flight"])]}
        if path.endswith("/flight-offers"):
            return [copy.deepcopy(offers["flight"])]
        if path.endswith("/by-city"):
            return [{"hotelId": "HTEST"}]
        if path.endswith("/hotel-offers"):
            return [{"hotel": {"name": "SYNTHETIC HOTEL"}, "offers": [offers["hotel"]]}]
        if path.endswith("/H1"):
            return {"offers": [copy.deepcopy(offers["hotel"])]}
        submitted.append((path, kwargs["json"]))
        if fault["post"]:
            raise httpx.ReadTimeout("synthetic-secret-must-not-leak")
        return {"id": "synthetic-order", "travelers": "must-not-expose", "associatedRecords": []}

    monkeypatch.setattr(Amadeus, "call", call)
    return {t.name: t for t in build_tools(settings)}, store, offers, submitted, fault, profile


def run(tools, name, *, context=None, **args):
    return asyncio.run(tools[name].coroutine(**args, runtime=context or runtime()))


def search(tools, kind):
    args = (
        {"origin": "PEK", "destination": "SHA", "departure_date": "2030-01-01"}
        if kind == "flight"
        else {"city_code": "SHA", "check_in": "2030-01-01", "check_out": "2030-01-02"}
    )
    result = run(tools, "search_flights" if kind == "flight" else "search_hotels", **args)
    assert result["environment"] == "test"
    return result["results"][0]["quote_id"]


@pytest.mark.parametrize("kind", ["flight", "hotel"])
def test_prepare_requires_exact_server_confirmation_and_never_duplicates(travel, kind):
    tools, store, _, submitted, _, _ = travel
    key = search(tools, kind)
    tool = "book_" + kind
    args = {"quote_id": key, "profile_name": "synthetic"}
    prepared = run(tools, tool, **args)
    assert prepared["status"] == "requires_confirmation" and not submitted
    assert prepared["travelers"] == [{"firstName": "TEST", "lastName": "ONLY"}]
    confirmation = prepared["confirmation_message"].split()[1]
    assert "confirmation" not in tools[tool].args
    result = run(tools, tool, context=runtime(confirmation=confirmation), **args)
    assert result["status"] == "reserved" and len(submitted) == 1
    assert "must-not-expose" not in str(result)
    assert (
        run(tools, tool, context=runtime(confirmation=confirmation), **args)["status"] == "reserved"
    )
    assert len(submitted) == 1
    assert store.get(key, "synthetic-conversation", kind)["state"] == "reserved"
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert (store.root / "bookings.sqlite3").stat().st_mode & 0o777 == 0o600


def test_no_runtime_context_and_other_conversation_cannot_book(travel):
    tools, _, _, submitted, _, _ = travel
    key = search(tools, "flight")
    args = {"quote_id": key, "profile_name": "synthetic"}
    assert "error" in run(tools, "book_flight", context=runtime(owner="another"), **args)
    assert "error" in run(tools, "book_flight", context=SimpleNamespace(config={}), **args)
    assert not submitted


@pytest.mark.parametrize("change", ["price", "terms", "traveler"])
def test_price_policy_or_passenger_changes_invalidate_confirmation(travel, change):
    tools, _, offers, submitted, _, profile = travel
    key = search(tools, "hotel")
    args = {"quote_id": key, "profile_name": "synthetic"}
    prepared = run(tools, "book_hotel", **args)
    confirmation = prepared["confirmation_message"].split()[1]
    if change == "price":
        offers["hotel"]["price"]["total"] = "400.00"
    elif change == "terms":
        offers["hotel"]["policies"]["paymentType"] = "deposit"
    else:
        data = json.loads(profile.read_text())
        data["hotel"]["guests"][0]["name"]["firstName"] = "CHANGED"
        profile.write_text(json.dumps(data))
    result = run(tools, "book_hotel", context=runtime(confirmation=confirmation), **args)
    assert result["status"] == "requires_confirmation" and not submitted
    assert result["confirmation_message"] != prepared["confirmation_message"]


def test_timeout_is_durable_unknown_and_no_automatic_retry(travel):
    tools, store, _, submitted, fault, _ = travel
    key = search(tools, "flight")
    args = {"quote_id": key, "profile_name": "synthetic"}
    prepared = run(tools, "book_flight", **args)
    fault["post"] = True
    confirmation = prepared["confirmation_message"].split()[1]
    result = run(tools, "book_flight", context=runtime(confirmation=confirmation), **args)
    assert result["status"] == "unknown" and "secret" not in str(result)
    # Build a fresh set of tools to simulate a service restart.
    fresh = {t.name: t for t in build_tools(Settings())}
    assert run(fresh, "book_flight", **args)["status"] == "unknown"
    assert len(submitted) == 1
    assert store.get(key, "synthetic-conversation", "flight")["state"] == "unknown"


def test_atomic_claim_expiry_and_private_profile_permissions(travel):
    tools, store, _, _, _, profile = travel
    key = search(tools, "flight")
    old = store.get(key, "synthetic-conversation", "flight")
    updated = {**old, "expires": time.time() - 1}
    store.update(key, updated, expected=old)
    with pytest.raises(Exception, match="状态已变化"):
        store.update(key, updated, expected=old)
    assert "error" in run(tools, "book_flight", quote_id=key, profile_name="synthetic")
    key = search(tools, "flight")
    profile.chmod(0o644)
    result = run(tools, "book_flight", quote_id=key, profile_name="synthetic")
    assert "0600" in result["error"]


def test_missing_real_credentials_is_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = {t.name: t for t in build_tools(Settings())}
    result = run(
        tools, "search_flights", origin="PEK", destination="SHA", departure_date="2030-01-01"
    )
    assert "AMADEUS_CLIENT_ID" in result["error"]


def test_amadeus_oauth_token_stays_in_server_headers(monkeypatch):
    requests = []

    async def request(timeout, method, url, **kwargs):
        requests.append((url, kwargs))
        if url.endswith("/token"):
            assert kwargs["data"]["client_secret"] == "synthetic-secret"
            return {"access_token": "synthetic-token", "expires_in": 1000}
        assert kwargs["headers"]["Authorization"] == "Bearer synthetic-token"
        return {"data": []}

    monkeypatch.setattr("personal_assistant_agent.agents.travel_tools.request_json", request)
    api = Amadeus(
        Settings(amadeus_client_id="synthetic-id", amadeus_client_secret="synthetic-secret")
    )

    async def twice():
        await api.call("GET", "/v2/shopping/flight-offers")
        await api.call("GET", "/v2/shopping/flight-offers")

    asyncio.run(twice())
    assert len(requests) == 3
    assert all("synthetic-secret" not in url for url, _ in requests)


def test_actual_graph_injects_runtime_and_prepares_without_ordering(travel):
    _, _, _, submitted, _, _ = travel

    async def generate(self, messages, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            name, args = (
                "search_flights",
                {"origin": "PEK", "destination": "SHA", "departure_date": "2030-01-01"},
            )
        elif len(results) == 1:
            key = json.loads(results[-1].content)["results"][0]["quote_id"]
            name, args = "book_flight", {"quote_id": key, "profile_name": "synthetic"}
        else:
            result = json.loads(results[-1].content)
            assert result["status"] == "requires_confirmation"
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="等待用户确认"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="", tool_calls=[{"name": name, "args": args, "id": "call-" + name}]
                    )
                )
            ]
        )

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        graph = build_agent(Settings(api_key="synthetic-model-key"))
        result = asyncio.run(
            graph.ainvoke(
                {"messages": [{"role": "user", "content": "查询并准备航班报价"}]},
                config=runtime().config,
            )
        )
    assert result["messages"][-1].content == "等待用户确认"
    assert not submitted
