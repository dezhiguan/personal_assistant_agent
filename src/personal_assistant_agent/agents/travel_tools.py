"""Amadeus searches and durable, quote-bound, user-confirmed bookings.

Passenger/payment profiles live outside the agent workspace. Only names and
offer terms enter tool results. Claiming a quote is atomic and happens before
the order POST: ambiguous network failures must be reconciled, never retried.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from personal_assistant_agent.agents.life_tools import guarded
from personal_assistant_agent.agents.tool_http import ToolError, request_json
from personal_assistant_agent.core.config import Settings


def digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class TravelStore:
    def __init__(self, settings: Settings):
        configured = Path(settings.travel_private_dir)
        if configured.is_absolute() or ".." in configured.parts or not configured.parts:
            raise ValueError("TRAVEL_PRIVATE_DIR 必须为项目内独立的相对目录。")
        self.root = configured.absolute()
        workspace = Path(settings.workspace_dir).resolve()
        if self.root.resolve().is_relative_to(workspace):
            raise ValueError("TRAVEL_PRIVATE_DIR 必须在智能体工作区之外。")
        if self.root == Path.cwd() or self.root == Path.home():
            raise ValueError("TRAVEL_PRIVATE_DIR 必须是独立私有目录。")

    def secure_root(self):
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ToolError("旅行私有目录不能经过符号链接。")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    @contextmanager
    def connection(self):
        self.secure_root()
        path = self.root / "bookings.sqlite3"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        path.chmod(0o600)
        conn = sqlite3.connect(path, timeout=10)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS quotes (id TEXT PRIMARY KEY, data TEXT)")
            with conn:
                yield conn
        finally:
            conn.close()

    def put(self, data: dict) -> str:
        key = uuid4().hex
        with self.connection() as conn:
            conn.execute("INSERT INTO quotes VALUES (?, ?)", (key, json.dumps(data)))
        return key

    def get(self, key: str, owner: str, kind: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", key):
            raise ToolError("报价编号无效。")
        with self.connection() as conn:
            row = conn.execute("SELECT data FROM quotes WHERE id=?", (key,)).fetchone()
        if not row:
            raise ToolError("报价不存在，请重新搜索。")
        data = json.loads(row[0])
        if data["owner"] != owner or data["kind"] != kind:
            raise ToolError("报价不属于当前会话或预订类型不匹配。")
        if data.get("state") == "open" and data["expires"] < time.time():
            raise ToolError("报价已过期，请重新搜索。")
        return data

    def update(self, key: str, data: dict, *, expected: dict):
        with self.connection() as conn:
            cursor = conn.execute(
                "UPDATE quotes SET data=? WHERE id=? AND data=?",
                (json.dumps(data), key, json.dumps(expected)),
            )
            if cursor.rowcount != 1:
                raise ToolError("此报价状态已变化，请核对预订状态，勿重复提交。")

    def profile(self, name: str, kind: str, provider: str = "amadeus") -> tuple[dict, str, list]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
            raise ToolError("旅客档案名仅允许字母、数字、下划线和连字符。")
        self.secure_root()
        path = self.root / "profiles" / (name + ".json")
        if path.parent.is_symlink() or path.is_symlink():
            raise ToolError("旅客档案不能使用符号链接。")
        if path.parent.exists() and path.parent.stat().st_mode & 0o077:
            raise ToolError("旅客档案目录权限必须为 0700。")
        if not path.is_file():
            raise ToolError("请先由用户在服务端配置私有旅客档案；不要在聊天中提交证件或银行卡。")
        if path.stat().st_mode & 0o077 or path.stat().st_nlink != 1:
            raise ToolError("旅客档案权限必须为 0600，且不能使用硬链接。")
        with path.open("rb") as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise ToolError("旅客档案超过 64 KiB。")
        loaded = json.loads(raw)
        if provider == "tuniu":
            from personal_assistant_agent.agents.tuniu_tools import private_profile

            data, names = private_profile(loaded["tuniu"][kind], kind)
            return data, digest(data), names
        data = loaded[kind]
        fields = ("travelers", "contacts") if kind == "flight" else ("guests", "payments")
        if any(not data.get(field) for field in fields):
            raise ToolError("旅客档案缺少必需的乘客、联系或付款信息。")
        data = {k: data[k] for k in fields + (("rooms",) if "rooms" in data else ())}
        people = data["travelers" if kind == "flight" else "guests"]
        names = [{k: p["name"][k] for k in ("firstName", "lastName")} for p in people]
        return data, digest(data), names


def scope(runtime: ToolRuntime) -> tuple[str, str]:
    # These values are supplied by ChatService from the current HTTP request,
    # never by model tool arguments or tool output/history.
    context = runtime.config.get("configurable", {}).get("travel_context", {})
    owner = context.get("conversation_id", "")
    if not owner:
        raise ToolError("旅行工具需要由聊天服务提供当前会话上下文。")
    return owner, context.get("confirmation", "")


class Amadeus:
    def __init__(self, settings: Settings):
        if settings.amadeus_environment not in {"test", "production"}:
            raise ValueError("AMADEUS_ENVIRONMENT 必须为 test 或 production。")
        self.settings = settings
        self.base = (
            "https://api.amadeus.com"
            if settings.amadeus_environment == "production"
            else "https://test.api.amadeus.com"
        )
        self.token = ""
        self.expiry = 0

    async def call(self, method: str, path: str, **kwargs):
        s = self.settings
        if not s.amadeus_client_id or not s.amadeus_client_secret:
            raise ToolError("请在服务端配置 AMADEUS_CLIENT_ID 和 AMADEUS_CLIENT_SECRET。")
        if self.expiry < time.time():
            auth = await request_json(
                s.tool_timeout,
                "POST",
                self.base + "/v1/security/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": s.amadeus_client_id,
                    "client_secret": s.amadeus_client_secret,
                },
            )
            self.token = auth["access_token"]
            self.expiry = time.time() + max(0, int(auth["expires_in"]) - 30)
        data = await request_json(
            s.tool_timeout,
            method,
            self.base + path,
            headers={"Authorization": "Bearer " + self.token},
            **kwargs,
        )
        if "errors" in data or "data" not in data:
            raise ToolError("Amadeus 未返回可用结果，请检查账号权限或查询条件。")
        return data["data"]


def iata(value: str) -> str:
    value = value.upper()
    if not re.fullmatch(r"[A-Z]{3}", value):
        raise ToolError("机场或城市必须使用三字母 IATA 代码。")
    return value


def future(value: str):
    if date.fromisoformat(value) < date.today():
        raise ToolError("出行日期不能早于今天。")


def terms(offer: dict, kind: str) -> dict:
    fields = (
        (
            "price",
            "itineraries",
            "travelerPricings",
            "validatingAirlineCodes",
            "nonHomogeneous",
            "instantTicketingRequired",
            "lastTicketingDate",
        )
        if kind == "flight"
        else (
            "price",
            "checkInDate",
            "checkOutDate",
            "room",
            "guests",
            "policies",
            "roomQuantity",
            "cancelPolicyHash",
        )
    )
    return {k: offer[k] for k in fields if k in offer}


def build_travel_tools(settings: Settings) -> list:
    if settings.travel_provider == "tuniu":
        from personal_assistant_agent.agents.tuniu_tools import build_tuniu_tools

        return build_tuniu_tools(settings)
    if settings.travel_provider != "amadeus":
        raise ValueError("TRAVEL_PROVIDER 必须为 amadeus 或 tuniu。")
    api = Amadeus(settings)
    store = TravelStore(settings)

    def save_offer(owner, kind, offer, label):
        return store.put(
            {
                "owner": owner,
                "kind": kind,
                "offer": offer,
                "label": label,
                "expires": time.time() + 900,
                "state": "open",
                "environment": settings.amadeus_environment,
            }
        )

    @tool
    @guarded
    async def search_flights(
        origin: str,
        destination: str,
        departure_date: str,
        runtime: ToolRuntime,
        return_date: str = "",
        adults: int = 1,
        currency: str = "CNY",
        max_results: int = 5,
    ) -> dict:
        """通过 Amadeus 检索真实航班候选；机场/城市为三字母 IATA，日期 YYYY-MM-DD。
        需服务端密钥，test 环境为测试数据。候选报价 15 分钟有效，不会自动下单。
        """
        owner, _ = scope(runtime)
        future(departure_date)
        if return_date:
            future(return_date)
            if return_date < departure_date:
                raise ToolError("返程日期不能早于出发日期。")
        if not 1 <= adults <= 9 or not 1 <= max_results <= 10:
            raise ToolError("adults 必须为 1–9，max_results 为 1–10。")
        params = {
            "originLocationCode": iata(origin),
            "destinationLocationCode": iata(destination),
            "departureDate": departure_date,
            "adults": adults,
            "currencyCode": iata(currency),
            "max": max_results,
        }
        if return_date:
            params["returnDate"] = return_date
        offers = await api.call("GET", "/v2/shopping/flight-offers", params=params)
        results = []
        for offer in offers[:max_results]:
            key = save_offer(owner, "flight", offer, {"origin": origin, "destination": destination})
            results.append({"quote_id": key, **terms(offer, "flight")})
        return {
            "source": "Amadeus",
            "environment": settings.amadeus_environment,
            "results": results,
            "expires_in_seconds": 900,
        }

    @tool
    @guarded
    async def search_hotels(
        city_code: str,
        check_in: str,
        check_out: str,
        runtime: ToolRuntime,
        adults: int = 1,
        currency: str = "CNY",
        max_results: int = 5,
    ) -> dict:
        """通过 Amadeus 检索城市酒店、单间房报价及取消政策；城市为三字母 IATA。
        需服务端密钥，test 环境为测试数据。每次最多 10 家候选，不会自动预订。
        """
        owner, _ = scope(runtime)
        future(check_in)
        future(check_out)
        if check_out <= check_in or not 1 <= adults <= 9 or not 1 <= max_results <= 10:
            raise ToolError("退房须晚于入住，adults=1–9，max_results=1–10。")
        hotels = await api.call(
            "GET",
            "/v1/reference-data/locations/hotels/by-city",
            params={"cityCode": iata(city_code)},
        )
        ids = [h["hotelId"] for h in hotels[:20]]
        if not ids:
            return {"source": "Amadeus", "results": [], "environment": settings.amadeus_environment}
        data = await api.call(
            "GET",
            "/v3/shopping/hotel-offers",
            params={
                "hotelIds": ",".join(ids),
                "adults": adults,
                "checkInDate": check_in,
                "checkOutDate": check_out,
                "roomQuantity": 1,
                "currency": iata(currency),
            },
        )
        results = []
        for hotel in data[:max_results]:
            if not hotel.get("available", True):
                continue
            for offer in hotel.get("offers", [])[:1]:
                key = save_offer(owner, "hotel", offer, hotel["hotel"])
                results.append({"quote_id": key, "hotel": hotel["hotel"], **terms(offer, "hotel")})
        return {
            "source": "Amadeus",
            "environment": settings.amadeus_environment,
            "results": results,
            "expires_in_seconds": 900,
        }

    async def refreshed(offer, kind):
        if kind == "flight":
            data = await api.call(
                "POST",
                "/v1/shopping/flight-offers/pricing",
                json={"data": {"type": "flight-offers-pricing", "flightOffers": [offer]}},
            )
            return data["flightOffers"][0]
        data = await api.call("GET", "/v3/shopping/hotel-offers/" + quote(offer["id"], safe=""))
        return data["offers"][0]

    async def book(quote_id, profile_name, runtime, kind):
        owner, confirmation = scope(runtime)
        old = store.get(quote_id, owner, kind)
        if old.get("provider", "amadeus") != "amadeus":
            raise ToolError("供应商已切换，请重新搜索报价。")
        if old["environment"] != settings.amadeus_environment:
            raise ToolError("环境已切换，请重新搜索报价。")
        if old["state"] != "open":
            return {
                "status": old["state"],
                "receipt": old.get("receipt"),
                "message": "此报价已提交过，不会重复下单；状态不明时须向供应商核对。",
            }
        profile, profile_hash, names = store.profile(profile_name, kind)
        offer = await refreshed(old["offer"], kind)
        new = {**old, "offer": offer}
        offer_hash = digest(terms(offer, kind))
        changed = old.get("profile_hash") != profile_hash or old.get("offer_hash") != offer_hash
        if changed or not confirmation or confirmation != old.get("confirmation_id"):
            new.update(
                {
                    "confirmation_id": uuid4().hex,
                    "profile_hash": profile_hash,
                    "offer_hash": offer_hash,
                }
            )
            store.update(quote_id, new, expected=old)
            return {
                "status": "requires_confirmation",
                "quote_id": quote_id,
                "environment": settings.amadeus_environment,
                "travelers": names,
                "selection": old["label"],
                "terms": terms(offer, kind),
                "confirmation_message": "确认预订 " + new["confirmation_id"],
                "message": "请展示旅客、行程/房型、总价、取消与付款政策，"
                "由用户下一轮单独发送 confirmation_message 后再调用预订工具。",
            }
        # Resolve OAuth/refresh before atomically claiming. After this point, any
        # error/cancellation/crash leaves a durable unknown state and no retry.
        new["state"] = "unknown"
        store.update(quote_id, new, expected=old)
        payload = (
            {"type": "flight-order", "flightOffers": [offer], **profile}
            if kind == "flight"
            else {"offerId": offer["id"], **profile}
        )
        path = "/v1/booking/flight-orders" if kind == "flight" else "/v1/booking/hotel-bookings"
        try:
            result = await api.call("POST", path, json={"data": payload})
            records = result if isinstance(result, list) else [result]
            receipt = [
                {
                    k: row[k]
                    for k in ("id", "providerConfirmationId", "associatedRecords")
                    if k in row
                }
                for row in records
            ]
            if not receipt or any(not row.get("id") for row in receipt):
                raise ToolError("供应商未返回可核验的订单编号。")
        except Exception:
            return {
                "error": "订单提交结果不明，已禁止自动重试；请到 Amadeus/供应商核对订单。",
                "status": "unknown",
                "quote_id": quote_id,
            }
        done = {**new, "state": "reserved", "receipt": receipt}
        store.update(quote_id, done, expected=new)
        return {
            "status": "reserved",
            "environment": settings.amadeus_environment,
            "quote_id": quote_id,
            "receipt": receipt,
            "message": "供应商已返回预订记录；test 环境不是真实订单，"
            "航班预订记录不代表已付款或已出票。",
        }

    @tool
    @guarded
    async def book_flight(quote_id: str, profile_name: str, runtime: ToolRuntime) -> dict:
        """预订已搜索航班；旅客信息只从服务端私有档案读取（profile_name 不含 .json）。
        首次仅重新验价并生成确认编号；用户下一轮单独发送“确认预订 编号”后才能下单。
        价格/乘客/条款变化重新确认；同一报价不重复下单，超时需人工核对供应商订单。
        """
        return await book(quote_id, profile_name, runtime, "flight")

    @tool
    @guarded
    async def book_hotel(quote_id: str, profile_name: str, runtime: ToolRuntime) -> dict:
        """预订已搜索酒店；住客及付款信息从服务端私有档案读取，不放进聊天或工具参数。
        首次仅验价并生成确认编号；用户下一轮单独发送“确认预订 编号”才能提交真实订单。
        费用/取消政策/旅客变化需重新确认；结果不明时禁止自动重试，需人工核对订单。
        """
        return await book(quote_id, profile_name, runtime, "hotel")

    return [search_flights, search_hotels, book_flight, book_hotel]
