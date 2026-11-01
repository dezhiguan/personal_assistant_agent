"""Request-local, measured model/tool spans for the public invocation timeline.

Only provider usage is counted. Tool result prices are never billing evidence.
Model prompts, reasoning, credentials and workspace file bodies are not exported.
"""

import asyncio
import json
import math
import re
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import add_usage
from langsmith.run_helpers import get_current_run_tree

from personal_assistant_agent.observability.costs import estimate_cost

LOCAL_TOOLS = {
    "calculate",
    "get_current_time",
    "validate_time_budget",
    "check_cn_holiday",
    "convert_units",
    "write_todos",
    "ls",
    "glob",
    "grep",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "execute",
    "create_calendar_event",
    "set_reminder",
}
DETAIL_TOOLS = {
    "calculate",
    "get_current_time",
    "validate_time_budget",
    "check_cn_holiday",
    "convert_units",
    "geocode",
    "get_city_weather",
    "get_weather_alert",
    "fetch_url",
    "baidu_search",
    "search_flights",
    "search_hotels",
    "amap_maps",
}
PRIVATE_KEY = re.compile(
    r"key|secret|password|authorization|cookie|ticket|credential|token|passport|"
    r"identity|document|phone|email|contact|traveler|profile|payment|confirmation",
    re.I,
)


def safe_detail(value, secrets=(), depth=0):
    """Bounded preview, recursively redacted before transport or browser storage."""
    if depth > 5:
        return "[已省略]"
    if isinstance(value, dict):
        return {
            str(k)[:100]: "[已隐藏]"
            if PRIVATE_KEY.search(str(k))
            else safe_detail(v, secrets, depth + 1)
            for k, v in list(value.items())[:25]
        }
    if isinstance(value, (list, tuple)):
        return [safe_detail(v, secrets, depth + 1) for v in value[:8]]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    value = str(value)
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[已隐藏]")
    value = re.sub(r"(?i)bearer\s+\S+", "Bearer [已隐藏]", value)
    value = re.sub(
        r"(?i)((?:api[_-]?key|password|secret|token|authorization|cookie)"
        r"[\"'\s]*[:=][\"'\s]*)[^\s\"'&,}]+",
        r"\1[已隐藏]",
        value,
    )
    value = re.sub(r"(https?://)[^/\s]+@", r"\1[已隐藏]@", value)
    return value[:1600] + ("…[已截断]" if len(value) > 1600 else "")


def tool_payload(value):
    if isinstance(value, ToolMessage):
        value = value.content
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def preview(value, secrets):
    safe = safe_detail(value, secrets)
    encoded = json.dumps(safe, ensure_ascii=False)
    return safe if len(encoded) <= 8000 else encoded[:8000] + "…[已截断]"


class InvocationCollector(BaseCallbackHandler):
    """Inherited callbacks include nested agents; each leaf is counted once."""

    run_inline = True

    def __init__(self, settings, queue):
        self.settings = settings
        self.queue = queue
        self.loop = asyncio.get_running_loop()
        self.started = time.perf_counter()
        self.started_at = datetime.now(UTC).isoformat()
        self.spans = {}
        self.parents = {}
        self.calls = {}
        self.lock = threading.RLock()
        self.secrets = tuple(
            getattr(settings, key)
            for key in (
                "api_key",
                "langsmith_api_key",
                "amap_api_key",
                "qweather_api_key",
                "tuniu_api_key",
                "amadeus_client_id",
                "amadeus_client_secret",
                "baidu_proxy",
            )
            if getattr(settings, key)
        )

    def _emit(self, span):
        event = {"type": "trace", "span": {k: v for k, v in span.items() if k != "_start"}}
        try:
            same_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:
            same_loop = False
        if same_loop:
            self.queue.put_nowait(("trace", event))
        elif not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, ("trace", event))

    def _start(self, kind, name, run_id, parent_run_id, **fields):
        with self.lock:
            key = str(run_id)
            self.parents[key] = str(parent_run_id) if parent_run_id else None
            parent = self.parents[key]
            visited = set()
            while parent and parent not in visited:
                visited.add(parent)
                if parent in self.spans and self.spans[parent]["kind"] == "tool":
                    break
                parent = self.parents.get(parent)
            if parent is None and (tree := get_current_run_tree()) is not None:
                # LangSmith @traceable middleware creates real ancestor spans
                # without LangChain on_chain_start callbacks. The native trace
                # order preserves those missing links, including parallel tasks.
                for part in reversed(tree.dotted_order.split(".")):
                    ancestor = part[-36:]
                    if self.spans.get(ancestor, {}).get("kind") == "tool":
                        parent = ancestor
                        break
            now = time.perf_counter()
            span = {
                "id": key,
                "sequence": len(self.spans) + 1,
                "kind": kind,
                "name": name,
                "parent_tool_id": parent,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
                "offset_ms": round((now - self.started) * 1000, 3),
                "duration_ms": None,
                "_start": now,
                **fields,
            }
            self.spans[key] = span
            self._emit(span)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        with self.lock:
            self.parents[str(run_id)] = str(parent_run_id) if parent_run_id else None

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs
    ):
        name = (
            (metadata or {}).get("ls_model_name")
            or kwargs.get("invocation_params", {}).get("model_name")
            or self.settings.model_name
        )
        self._start("model", name, run_id, parent_run_id, usage=None, cost_estimate=None)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.on_chat_model_start(serialized, [], **kwargs)

    def _finish(self, run_id, status, **fields):
        with self.lock:
            span = self.spans.get(str(run_id))
            if span is None or span["status"] != "running":
                return
            span.update(
                status=status,
                duration_ms=round((time.perf_counter() - span["_start"]) * 1000, 3),
                ended_at=datetime.now(UTC).isoformat(),
                **fields,
            )
            self._emit(span)

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = None
        for generations in response.generations:
            for generation in generations:
                measured = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if measured:
                    usage = add_usage(usage, measured)
        with self.lock:
            span = self.spans.get(str(run_id))
            if usage is not None:
                self.calls[str(run_id)] = usage
            estimate = None
            if usage is not None and span:
                estimate = estimate_cost(usage, replace(self.settings, model_name=span["name"]))
            self._finish(run_id, "finished", usage=usage, cost_estimate=estimate)

    def on_llm_error(self, error, *, run_id, **kwargs):
        # Provider exception bodies can contain credentials. Export only the class.
        self._finish(run_id, "error", error=type(error).__name__)

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs
    ):
        name = (serialized or {}).get("name", "unknown")
        if name in LOCAL_TOOLS:
            cost = {
                "status": "no_external_fee",
                "amount": 0,
                "currency": "CNY",
                "source": "本地执行，无外部 API 费用；不含服务器运行成本",
            }
        elif name == "task":
            cost = {
                "status": "included",
                "amount": None,
                "source": "内部模型和工具分别计入汇总，不重复计费",
            }
        else:
            cost = {
                "status": "unknown",
                "amount": None,
                "source": "服务未返回可核验的调用账单；结果报价不属于调用费用",
            }
        detail = preview(inputs if inputs is not None else tool_payload(input_str), self.secrets)
        if name not in DETAIL_TOOLS:
            detail = "为保护文件内容和敏感数据，此工具不展示原始参数。"
        self._start("tool", name, run_id, parent_run_id, arguments=detail, tool_cost=cost)

    def on_tool_end(self, output, *, run_id, **kwargs):
        data = tool_payload(output)
        failed = (isinstance(output, ToolMessage) and output.status == "error") or (
            isinstance(data, dict) and bool(data.get("error"))
        )
        span = self.spans.get(str(run_id), {})
        result = (
            preview(data, self.secrets)
            if span.get("name") in DETAIL_TOOLS
            else ("工具已返回。原始内容已省略，避免暴露文件或敏感数据。")
        )
        self._finish(run_id, "error" if failed else "finished", result=result)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "error", error=type(error).__name__)

    def close_open_spans(self, status):
        for span in list(self.spans.values()):
            if span["status"] == "running":
                self._finish(span["id"], status)

    def summary(self):
        models = [s for s in self.spans.values() if s["kind"] == "model"]
        tools = [s for s in self.spans.values() if s["kind"] == "tool"]
        measured = [s for s in models if s.get("usage") is not None]
        estimates = [s["cost_estimate"] for s in models if s.get("cost_estimate") is not None]
        unknown = sum(s["tool_cost"]["status"] == "unknown" for s in tools)
        return {
            "duration_ms": round((time.perf_counter() - self.started) * 1000, 3),
            "model_calls": len(models),
            "tool_calls": len(tools),
            "measured_model_calls": len(measured),
            "usage_complete": len(measured) == len(models),
            "total_tokens": sum(s["usage"]["total_tokens"] for s in measured),
            "model_cost_estimate_cny": sum(e["total_cost_cny"] for e in estimates),
            "model_cost_complete": len(estimates) == len(models),
            "priced_model_calls": len(estimates),
            "unknown_tool_costs": unknown,
            "cost_complete": not unknown and len(estimates) == len(models),
        }


async def measured_stream(source, queue):
    """Multiplex callbacks while a graph is waiting for a model or tool response."""

    async def pump():
        try:
            async for item in source:
                queue.put_nowait(item)
        except Exception as error:
            queue.put_nowait(("exception", error))
        finally:
            queue.put_nowait(("end", None))

    task = asyncio.create_task(pump())
    try:
        while True:
            mode, data = await queue.get()
            if mode == "end":
                break
            if mode == "exception":
                raise data
            yield mode, data
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
