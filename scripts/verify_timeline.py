"""Live timeline checks: HTTP events vs independent native LangSmith leaf spans.

Uses synthetic questions and real paid model calls. Requires a freshly started
server with the current source and configured LangSmith tracing. Never treats
list-price estimates as invoice charges or tool result prices as API fees.
"""

import argparse
import json
import logging
import math
import time
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx
import truststore
from langsmith import Client

from personal_assistant_agent.core.config import get_settings


def check_events(events, elapsed_ms):
    assert events[0].get("timeline_version") == 1, "Restart server: timeline protocol missing"
    assert events[-1]["type"] == "done", "Chat did not complete"
    spans = {}
    starts = {}
    for event in events:
        if event["type"] != "trace":
            continue
        span = event["span"]
        if span["status"] == "running":
            assert span["id"] not in starts, "Duplicate start"
            starts[span["id"]] = span
        else:
            assert span["id"] in starts, "Missing start callback"
        spans[span["id"]] = span
    ordered = sorted(spans.values(), key=lambda s: s["sequence"])
    assert [s["sequence"] for s in ordered] == list(range(1, len(ordered) + 1))
    assert [s["offset_ms"] for s in ordered] == sorted(s["offset_ms"] for s in ordered)
    for span in ordered:
        assert span["status"] != "running", "Unclosed span"
        assert span["duration_ms"] >= 0
        wall = (
            datetime.fromisoformat(span["ended_at"]) - datetime.fromisoformat(span["started_at"])
        ).total_seconds() * 1000
        assert abs(wall - span["duration_ms"]) < 50, "Span timestamp and timer differ"
    summary = events[-1]["trace_summary"]
    assert 0 <= summary["duration_ms"] <= elapsed_ms + 100, "Impossible request duration"
    assert elapsed_ms - summary["duration_ms"] < 2000, "Request timer misses significant time"
    assert summary["model_calls"] == sum(s["kind"] == "model" for s in ordered)
    assert summary["tool_calls"] == sum(s["kind"] == "tool" for s in ordered)
    models = [s for s in ordered if s["kind"] == "model"]
    assert models and summary["usage_complete"], "Provider usage missing"
    assert summary["total_tokens"] == sum(s["usage"]["total_tokens"] for s in models)
    usage = next(e for e in events if e["type"] == "usage")
    assert usage["total_tokens"] == summary["total_tokens"], "Stream/summary usage differs"
    for span in models:
        estimate = span.get("cost_estimate")
        assert estimate and estimate["cost_is_estimate"], "Missing or mislabeled estimate"
        measured = span["usage"]
        cached = measured.get("input_token_details", {}).get("cache_read", 0)
        # Independently recompute from exported source rates and provider counts.
        inputs = (
            (measured["input_tokens"] - cached) * Decimal(estimate["input_per_million_cny"])
            + cached * Decimal(estimate["cache_read_per_million_cny"])
        ) / 1000000
        outputs = measured["output_tokens"] * Decimal(estimate["output_per_million_cny"]) / 1000000
        assert math.isclose(float(inputs + outputs), estimate["total_cost_cny"], abs_tol=1e-12)
    known_cost = sum(Decimal(str(s["cost_estimate"]["total_cost_cny"])) for s in models)
    assert math.isclose(float(known_cost), summary["model_cost_estimate_cny"], abs_tol=1e-12)
    unknown = sum(s["kind"] == "tool" and s["tool_cost"]["status"] == "unknown" for s in ordered)
    assert unknown == summary["unknown_tool_costs"]
    assert summary["cost_complete"] == (unknown == 0 and summary["model_cost_complete"])
    return ordered, summary


def check_native(client, settings, request_id, spans):
    deadline = time.monotonic() + 30
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            runs = list(
                client.list_runs(
                    project_name=settings.langsmith_project, trace_id=request_id, limit=100
                )
            )
        native = {str(r.id): r for r in runs if r.run_type in {"llm", "tool"}}
        if all(s["id"] in native and native[s["id"]].end_time for s in spans):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Native trace spans not complete within 30 seconds")
        time.sleep(1)
    assert len(native) == len(spans), "Timeline omits a native model/tool span"
    deviations = []
    for span in spans:
        run = native[span["id"]]
        ms = (run.end_time - run.start_time).total_seconds() * 1000
        deviation = abs(ms - span["duration_ms"])
        assert deviation < 250, "Native trace and measured duration differ"
        deviations.append(round(deviation, 3))
        if span["kind"] == "model":
            for public, field in (
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                assert span["usage"][public] == getattr(run, field), "Native/provider tokens differ"
    return max(deviations, default=0)


def verify(args):
    settings = get_settings()
    if not settings.tracing_configured:
        raise RuntimeError("LangSmith must be configured for independent verification")
    cases = [
        ("simple", "这是调用链路验收。只回复：链路验证通过。", set()),
        (
            "tools",
            "这是调用链路验收。请实际调用 calculate 计算 6*7，"
            "并调用 get_current_time 查询上海时间。尽可能在同一轮调用，然后简短报告。",
            {"calculate", "get_current_time"},
        ),
        (
            "nested",
            "这是调用链路验收。请实际调用 task 委派 general-purpose 子代理，"
            "让子代理调用 calculate 计算 7*8，最后简短回答结果。不读写文件，不联网检索。",
            {"task", "calculate"},
        ),
    ]
    if settings.amap_api_key:
        cases.append(
            (
                "external-tool",
                "这是外部工具调用链路验收。请实际调用一次 amap_maps，"
                "action=search、query=外滩、city=上海、max_results=1。简短报告返回结果，"
                "不预订、不付款、不改动文件。",
                {"amap_maps"},
            )
        )
    if not settings.qweather_api_key:
        cases.append(
            (
                "tool-error",
                "这是工具失败链路验收。请实际调用一次 get_weather_alert "
                "查询上海当前预警。未配置时如实报告，不重试，不使用其他工具替代。",
                {"get_weather_alert"},
            )
        )
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
        if {case[0] for case in cases} != set(args.cases):
            raise RuntimeError("Selected cases require unavailable tool configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    try:
        with httpx.Client(timeout=240) as http:
            for name, question, required in cases:
                started = time.perf_counter()
                with http.stream(
                    "POST",
                    args.base_url.rstrip("/") + "/api/chat",
                    json={
                        "conversation_id": str(uuid4()),
                        "messages": [{"role": "user", "content": question}],
                    },
                ) as response:
                    response.raise_for_status()
                    events = [json.loads(line) for line in response.iter_lines() if line]
                elapsed_ms = (time.perf_counter() - started) * 1000
                path = args.output / f"{name}-events.json"
                path.write_text(json.dumps(events, ensure_ascii=False, indent=2))
                path.chmod(0o600)
                spans, summary = check_events(events, elapsed_ms)
                tools = [s for s in spans if s["kind"] == "tool"]
                assert required <= {s["name"] for s in tools}, "Required tool did not run"
                if name == "nested":
                    parent = next(s for s in tools if s["name"] == "task")
                    assert any(s.get("parent_tool_id") == parent["id"] for s in spans)
                    assert parent["tool_cost"]["status"] == "included"
                if name == "tool-error":
                    weather = next(s for s in tools if s["name"] == "get_weather_alert")
                    assert weather["status"] == "error"
                    assert weather["tool_cost"]["amount"] is None
                    assert summary["unknown_tool_costs"] >= 1
                if name == "external-tool":
                    external = next(s for s in tools if s["name"] == "amap_maps")
                    assert external["status"] == "finished", "External API did not succeed"
                    assert external["tool_cost"]["status"] == "unknown"
                    assert external["tool_cost"]["amount"] is None
                    assert not summary["cost_complete"]
                request_id = events[0]["request_id"]
                deviation = check_native(client, settings, request_id, spans)
                rows.append(
                    {
                        "case": name,
                        "request_id": request_id,
                        "summary": summary,
                        "max_native_duration_deviation_ms": deviation,
                        "passed": True,
                    }
                )
                print(
                    f"PASS {name}: {len(spans)} spans; native timing deviation <= {deviation} ms",
                    flush=True,
                )
    finally:
        client.close()
    report = {
        "verified_at": datetime.now(UTC).isoformat(),
        "model": settings.model_name,
        "live": True,
        "actual_invoice_verified": False,
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("Verified real provider usage and measured timings; costs remain list-price estimates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/timeline/live"))
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        choices=["simple", "tools", "nested", "external-tool", "tool-error"],
        help="Run selected cases only; may be repeated",
    )
    args = parser.parse_args()
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    truststore.inject_into_ssl()
    try:
        verify(args)
    except Exception as error:
        # Avoid exporting provider exception bodies or private headers.
        raise SystemExit(f"Timeline verification failed: {type(error).__name__}") from None
