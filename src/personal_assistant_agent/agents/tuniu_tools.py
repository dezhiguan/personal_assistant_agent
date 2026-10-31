"""Server-side Tuniu MCP adapter; raw credentials and passengers never reach the LLM."""

import asyncio
import json
import re
import time
from decimal import Decimal
from urllib.parse import urlsplit
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from personal_assistant_agent.agents import tool_http
from personal_assistant_agent.agents.life_tools import guarded
from personal_assistant_agent.agents.tool_http import ToolError
from personal_assistant_agent.agents.travel_tools import TravelStore, digest, future, scope
from personal_assistant_agent.core.config import Settings

OPERATIONS = {
    "flight_search": ("flight", ("searchLowestPriceFlight",)),
    "flight_detail": ("flight", ("multiCabinDetails",)),
    "flight_requirements": ("flight", ("getBookingRequiredInfo",)),
    "flight_create": ("flight", ("saveOrder",)),
    "hotel_search": ("hotel", ("tuniuHotelSearch", "tuniu_hotel_search")),
    "hotel_detail": ("hotel", ("tuniuHotelDetail", "tuniu_hotel_detail")),
    "hotel_create": ("hotel", ("tuniuHotelCreateOrder", "tuniu_hotel_create_order")),
}


def select(data: dict, fields) -> dict:
    return {k: data[k] for k in fields if k in data}


def required_text(value, maximum=120) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ToolError("供应商资料缺少必需文本或长度超限。")
    return value.strip()


def private_profile(data: dict, kind: str) -> tuple[dict, list]:
    """Whitelist payload fields so a profile cannot override quote-bound itinerary/price IDs."""
    if kind == "flight":
        people = data["tourists"]
        if not isinstance(people, list) or not 1 <= len(people) <= 9:
            raise ToolError("途牛档案需有 1–9 位成人乘机人。")
        tourists = []
        for p in people:
            if p.get("type") != "成人":
                raise ToolError("当前途牛接入仅支持成人；档案 type 必须明确为成人。")
            clean = {k: required_text(p[k]) for k in ("name", "idType", "idNumber", "mobile")}
            clean.update(select(p, ("type", "psptEndDate", "sex", "birthday")))
            tourists.append(clean)
        contact = data["contactTourist"]
        result = {
            "tourists": tourists,
            "contactTourist": {k: required_text(contact[k]) for k in ("name", "mobile")},
        }
        if contact.get("email"):
            result["contactTourist"]["email"] = required_text(contact["email"])
        return result, [{"name": p["name"]} for p in tourists]
    rooms = data["roomGuests"]
    if not isinstance(rooms, list) or len(rooms) != 1:
        raise ToolError("当前途牛酒店接入每次仅支持一间房。")
    people = rooms[0]["guests"]
    if not isinstance(people, list) or not 1 <= len(people) <= 9:
        raise ToolError("住客档案需有 1–9 位成人。")
    names = [{k: required_text(p[k]) for k in ("firstName", "lastName")} for p in people]
    result = {
        "roomGuests": [{"guests": names}],
        **{k: required_text(data[k]) for k in ("contactName", "contactPhone")},
    }
    if data.get("contactEmail"):
        result["contactEmail"] = required_text(data["contactEmail"])
    return result, names


def payment_link(value) -> str:
    """Only return provider-owned HTTPS links; never navigate or auto-pay."""
    if not isinstance(value, str) or len(value) > 4096 or re.search(r"[\s\\<>]", value):
        raise ToolError("供应商支付链接无效，请凭订单号到途牛核对。")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not parsed.hostname
        or not (parsed.hostname == "tuniu.com" or parsed.hostname.endswith(".tuniu.com"))
    ):
        raise ToolError("供应商支付链接不属于途牛，请凭订单号核对。")
    return value


class Tuniu:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base = settings.tuniu_mcp_base_url.rstrip("/")
        if self.base not in {"https://openapi.tuniu.cn/mcp", "https://openapi.tuniu.cn/hybrid/mcp"}:
            raise ValueError("TUNIU_MCP_BASE_URL 必须使用途牛官方生产 MCP 地址。")
        self.available: dict[str, set[str]] = {}
        self.identity = digest([self.base, settings.tuniu_api_key])

    async def rpc(self, service: str, method: str, params: dict):
        if not self.settings.tuniu_api_key.strip():
            raise ToolError("途牛未启用：请在服务端配置 TUNIU_API_KEY。")
        request_id = uuid4().hex
        async with asyncio.timeout(self.settings.tool_timeout):
            async with tool_http.client(self.settings.tool_timeout) as session:
                async with session.stream(
                    "POST",
                    self.base + "/" + service,
                    headers={
                        "apiKey": self.settings.tuniu_api_key,
                        "Accept": "application/json, text/event-stream",
                    },
                    json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                ) as response:
                    raw = (await tool_http.read_response(response)).decode("utf-8")
        try:
            data = json.loads(raw)
        except ValueError:
            # Select the matching JSON-RPC response; notifications may precede it in SSE.
            data = None
            for event in raw.replace("\r\n", "\n").split("\n\n"):
                body = "\n".join(
                    line[5:].strip() for line in event.splitlines() if line.startswith("data:")
                )
                if body:
                    item = json.loads(body)
                    if isinstance(item, dict) and item.get("id") == request_id:
                        data = item
            if data is None:
                raise ToolError("途牛未返回有效的 MCP 响应。") from None
        if (
            not isinstance(data, dict)
            or data.get("id") != request_id
            or data.get("error")
            or not isinstance(data.get("result"), dict)
        ):
            raise ToolError("途牛接口调用失败，请检查 Key、权限或服务状态。")
        return data["result"]

    async def call(self, operation: str, arguments: dict) -> dict:
        service, aliases = OPERATIONS[operation]
        if service not in self.available:
            listing = await self.rpc(service, "tools/list", {})
            self.available[service] = {t["name"] for t in listing["tools"]}
        name = next((n for n in aliases if n in self.available[service]), None)
        if not name:
            raise ToolError("当前途牛账户未开放所需工具，请检查服务权限或接口版本。")
        result = await self.rpc(service, "tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise ToolError("途牛工具执行失败；未确认业务成功，请核对权限或订单状态。")
        if operation == "flight_requirements":
            if not result.get("content") and not result.get("structuredContent"):
                raise ToolError("途牛未返回预订资料要求。")
            return {}  # Provider instructions are data, never model or authorization instructions.
        payload = result.get("structuredContent")
        if payload is None:
            payload = json.loads(
                "\n".join(c["text"] for c in result.get("content", []) if c.get("type") == "text")
            )
        if (
            not isinstance(payload, dict)
            or any(payload.get(k) is False for k in ("success", "successCode"))
            or payload.get("error")
        ):
            raise ToolError("途牛业务未成功，未获得可用结果。")
        return payload


def flight_offer(detail: dict, args: dict, adults: int, selected: dict | None = None) -> dict:
    cabins = detail["cabinInfo"]
    if selected is not None:
        cabins = [
            c
            for c in cabins
            if c.get("cabinClassCodes") == selected.get("cabinClassCodes")
            and c.get("cabinClass") == selected.get("cabinClass")
        ]
    if not cabins:
        raise ToolError("所选舱位已不可用，请重新搜索。")
    cabin = cabins[0]
    if not cabin.get("cabinPriceId"):
        raise ToolError("途牛未返回可预订的舱位标识。")
    base, tax = Decimal(str(cabin["basePrice"])), Decimal(str(cabin["totalTax"]))
    if not base.is_finite() or not tax.is_finite() or base < 0 or tax < 0:
        raise ToolError("航班报价无效。")
    flights = [
        select(
            f,
            (
                "flightNumber",
                "airlineCompany",
                "departureTime",
                "arrivalTime",
                "departureAirport",
                "arrivalAirport",
                "departureTerminal",
                "arrivalTerminal",
                "type",
                "craftType",
            ),
        )
        for f in detail["flightInfo"]
    ]
    if not flights or flights[0].get("flightNumber") != args["flightNo"]:
        raise ToolError("供应商返回的航班与所选航班不一致，请重新搜索。")
    public = {
        "flights": flights,
        **select(
            cabin,
            (
                "cabinClass",
                "cabinClassCodes",
                "buyCondition",
                "refundChangeRule",
                "baggageInfo",
                "mealCode",
                "reimbursementProof",
                "remainingSeats",
            ),
        ),
        "price": {
            "currency": "CNY",
            "base_per_adult": str(base),
            "tax_per_adult": str(tax),
            "reference_total": str((base + tax) * adults),
        },
        "adults": adults,
    }
    return {"args": args, "cabinPriceId": cabin["cabinPriceId"], "terms": public}


def hotel_offers(detail: dict, args: dict, adults: int) -> list[dict]:
    if str(detail["hotelId"]) != str(args["hotelId"]):
        raise ToolError("供应商返回的酒店与所选酒店不一致，请重新搜索。")
    offers = []
    for room in detail["roomTypes"]:
        if room.get("maxOccupancy") is not None and room["maxOccupancy"] < adults:
            continue
        for rate in room["ratePlans"]:
            if not rate.get("preBookParam") or rate.get("count") == 0:
                continue
            public = {
                "hotel": select(detail, ("hotelId", "hotelName", "address", "cityName")),
                "room": select(room, ("roomTypeId", "roomTypeName", "bedType", "maxOccupancy")),
                "rate": select(
                    rate,
                    (
                        "vendorRatePlanId",
                        "ratePlanName",
                        "rmbPrices",
                        "mealText",
                        "cancelDesc",
                        "count",
                    ),
                ),
                "policies": detail.get("policies", {}),
                "checkIn": args["checkIn"],
                "checkOut": args["checkOut"],
                "roomCount": 1,
                "adults": adults,
                "currency": "CNY",
                "price_note": "报价金额口径以供应商及支付页为准。",
            }
            if not rate.get("rmbPrices"):
                continue
            offers.append(
                {
                    "args": args,
                    "roomId": room["roomTypeId"],
                    "preBookParam": rate["preBookParam"],
                    "terms": public,
                }
            )
    return offers


def build_tuniu_tools(settings: Settings) -> list:
    api, store = Tuniu(settings), TravelStore(settings)

    def save(owner, kind, offer):
        return store.put(
            {
                "owner": owner,
                "kind": kind,
                "offer": offer,
                "provider": "tuniu",
                "identity": api.identity,
                "environment": "production",
                "state": "open",
                "expires": time.time() + 900,
            }
        )

    def limits(adults, currency, max_results):
        if currency.upper() != "CNY" or not 1 <= adults <= 9 or not 1 <= max_results <= 10:
            raise ToolError("途牛接入支持 CNY、1–9 位成人，max_results 为 1–10。")

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
        """途牛国内成人单程航班查询；origin/destination 填中文城市名，如北京、上海。
        返回15分钟有效的会话报价；CNY，含每成人票价与税费，不自动下单。
        往返需分别搜索两次单程。需服务端 TUNIU_API_KEY。
        """
        owner, _ = scope(runtime)
        future(departure_date)
        limits(adults, currency, max_results)
        if return_date:
            raise ToolError("当前途牛接入为单程，请分别搜索去程和返程并分别确认预订。")
        args = {
            "departureCityName": required_text(origin),
            "arrivalCityName": required_text(destination),
            "departureDate": departure_date,
        }
        rows = await api.call("flight_search", args)
        results = []
        for row in rows["data"][:max_results]:
            query = {**args, "flightNo": required_text(row["flightNumber"])}
            detail = await api.call("flight_detail", query)
            if not detail.get("cabinInfo"):
                continue
            offer = flight_offer(detail, query, adults)
            results.append({"quote_id": save(owner, "flight", offer), **offer["terms"]})
        return {
            "source": "Tuniu",
            "environment": "production",
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
        """途牛酒店单间房查询，city_code 填中文城市名（如北京），日期 YYYY-MM-DD。
        返回具体房型、报价和15分钟有效 quote_id。仅成人；不自动下单。
        需服务端 TUNIU_API_KEY，付款在用户确认创单后由本人打开供应商链接完成。
        """
        owner, _ = scope(runtime)
        future(check_in)
        future(check_out)
        limits(adults, currency, max_results)
        if check_out <= check_in:
            raise ToolError("退房日期必须晚于入住日期。")
        args = {
            "cityName": required_text(city_code),
            "checkIn": check_in,
            "checkOut": check_out,
            "adultNum": adults,
            "childNum": 0,
        }
        rows = await api.call("hotel_search", args)
        results = []
        for hotel in rows["hotels"][:max_results]:
            query = {
                "hotelId": hotel["hotelId"],
                "checkIn": check_in,
                "checkOut": check_out,
                "adultNum": adults,
                "roomNum": 1,
                "childNum": 0,
            }
            detail = await api.call("hotel_detail", query)
            offers = hotel_offers(detail, query, adults)
            if offers:
                offer = offers[0]
                results.append({"quote_id": save(owner, "hotel", offer), **offer["terms"]})
        return {
            "source": "Tuniu",
            "environment": "production",
            "results": results,
            "expires_in_seconds": 900,
        }

    async def book(quote_id, profile_name, runtime, kind):
        owner, confirmation = scope(runtime)
        old = store.get(quote_id, owner, kind)
        if old.get("provider") != "tuniu" or old.get("identity") != api.identity:
            raise ToolError("供应商或账户已切换，请重新搜索报价。")
        if old["state"] != "open":
            return {
                "status": old["state"],
                "quote_id": quote_id,
                "receipt": old.get("receipt"),
                "payment_url": old.get("payment_url"),
                "message": "此报价已提交，不会重复下单；支付和履约状态请到途牛核对。",
            }
        profile, profile_hash, names = store.profile(profile_name, kind, "tuniu")
        offer = old["offer"]
        if len(names) != offer["terms"]["adults"]:
            raise ToolError("私有档案人数与报价人数不一致，请使用对应档案或重新搜索。")
        if kind == "flight":
            await api.call("flight_requirements", {})
            detail = await api.call("flight_detail", offer["args"])
            fresh = flight_offer(detail, offer["args"], len(names), offer["terms"])
        else:
            detail = await api.call("hotel_detail", offer["args"])
            candidates = hotel_offers(detail, offer["args"], len(names))
            fresh = next(
                (
                    o
                    for o in candidates
                    if o["roomId"] == offer["roomId"]
                    and o["terms"]["rate"].get("vendorRatePlanId")
                    == offer["terms"]["rate"].get("vendorRatePlanId")
                    and o["terms"]["rate"].get("ratePlanName")
                    == offer["terms"]["rate"].get("ratePlanName")
                ),
                None,
            )
            if fresh is None:
                raise ToolError("所选房型报价已不可用，请重新搜索。")
        new = {**old, "offer": fresh}
        if old["expires"] < time.time():
            raise ToolError("报价已过期，请重新搜索并确认。")
        offer_hash = digest(fresh["terms"])
        if (
            old.get("profile_hash") != profile_hash
            or old.get("offer_hash") != offer_hash
            or not confirmation
            or confirmation != old.get("confirmation_id")
        ):
            new.update(
                confirmation_id=uuid4().hex, profile_hash=profile_hash, offer_hash=offer_hash
            )
            store.update(quote_id, new, expected=old)
            return {
                "source": "Tuniu",
                "status": "requires_confirmation",
                "quote_id": quote_id,
                "environment": "production",
                "travelers": names,
                "terms": fresh["terms"],
                "confirmation_message": "确认预订 " + new["confirmation_id"],
                "message": "展示旅客、行程、金额口径和条款；用户下一轮单独确认后才创建真实"
                "待支付订单。付款由用户在途牛页面完成；最终金额以付款页面为准。",
            }
        if kind == "flight":
            payload = {**fresh["args"], "cabinPriceId": fresh["cabinPriceId"], **profile}
        else:
            payload = {
                "hotelId": str(fresh["args"]["hotelId"]),
                "roomId": str(fresh["roomId"]),
                "preBookParam": fresh["preBookParam"],
                "checkInDate": fresh["args"]["checkIn"],
                "checkOutDate": fresh["args"]["checkOut"],
                "roomCount": 1,
                **profile,
            }
        new["state"] = "unknown"
        store.update(quote_id, new, expected=old)
        try:
            result = await api.call(kind + "_create", payload)
            order_id = result.get("orderId")
            if not isinstance(order_id, (str, int)) or not str(order_id):
                raise ToolError("供应商未返回订单号。")
            receipt = {"order_id": str(order_id)}
            done = {**new, "state": "pending_payment", "receipt": receipt}
            link = result.get("payUrl" if kind == "flight" else "paymentUrl")
            try:
                done["payment_url"] = payment_link(link)
            except (ToolError, ValueError):
                done["payment_url"] = None
            store.update(quote_id, done, expected=new)
        except Exception:
            return {
                "status": "unknown",
                "quote_id": quote_id,
                "error": "提交结果不明，已禁止重试，请到途牛订单中心核对。",
            }
        return {
            "source": "Tuniu",
            "status": "pending_payment",
            "quote_id": quote_id,
            "receipt": receipt,
            "payment_url": done["payment_url"],
            "message": "订单已创建，待本人到途牛支付；尚未确认付款、出票或酒店履约。"
            "没有有效链接时请凭订单号进入途牛订单中心核对，勿重复下单。",
        }

    @tool
    @guarded
    async def book_flight(quote_id: str, profile_name: str, runtime: ToolRuntime) -> dict:
        """预订途牛航班报价；档案从服务端私有 tuniu.flight 读取，不传证件到聊天。
        首次验价展示确认；本人下一轮单独发送确认预订编号后创单，返回本人付款链接。
        pending_payment 不代表已支付或出票；unknown 禁止重试，须到途牛核对。
        """
        return await book(quote_id, profile_name, runtime, "flight")

    @tool
    @guarded
    async def book_hotel(quote_id: str, profile_name: str, runtime: ToolRuntime) -> dict:
        """预订途牛酒店报价；服务端私有 tuniu.hotel 住客档案，无需保存银行卡。
        首次重新验价展示确认；本人下一轮独立确认后创单，返回途牛付款链接。
        pending_payment 不代表已支付或酒店已确认；unknown 禁止重试，须核单。
        """
        return await book(quote_id, profile_name, runtime, "hotel")

    return [search_flights, search_hotels, book_flight, book_hotel]
