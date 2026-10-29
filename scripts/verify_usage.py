"""Verify real streamed Qwen usage against LangSmith leaf and root aggregates."""

import argparse
import json
import math
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import truststore
from langsmith import Client

from personal_assistant_agent.core.config import get_settings


def verify(args):
    settings = get_settings()
    if not settings.tracing_configured:
        raise RuntimeError("LangSmith tracing must be configured")
    question = (
        "帮我规划一份作品集准备方案，先用write_todos建立至少3项待办，"
        "再给出简短实质建议；不要假装已经完成制作。"
        if args.planning
        else "只回复：验证通过"
    )
    with httpx.Client(timeout=120) as http:
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
    if not events or events[-1]["type"] != "done":
        raise RuntimeError("Chat did not complete")
    usages = [event for event in events if event["type"] == "usage"]
    if len(usages) != 1:
        raise RuntimeError("Expected one measured usage event")
    usage = usages[0]
    request_id = events[0]["request_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("latest-events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False) + "\n"
    )
    if not usage.get("cost_estimate") or min(usage["input_tokens"], usage["output_tokens"]) <= 0:
        raise RuntimeError("Missing measured tokens or supported price estimate")
    if args.planning and not any(event["type"] == "plan" for event in events):
        raise RuntimeError("Planning verification did not invoke the planning tool")
    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    try:
        deadline = time.monotonic() + 30
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                runs = list(
                    client.list_runs(
                        project_name=settings.langsmith_project, trace_id=request_id, limit=100
                    )
                )
            root = next((run for run in runs if str(run.id) == request_id), None)
            leaves = [run for run in runs if run.run_type == "llm"]
            if root and leaves and root.total_cost and all(run.total_cost for run in leaves):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("LangSmith did not expose complete native token/cost fields")
            time.sleep(1)
        if root.error or any(run.error for run in leaves):
            raise RuntimeError("Trace contains a model error")
        for field, native in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            if usage[field] != getattr(root, native) or usage[field] != sum(
                getattr(run, native) or 0 for run in leaves
            ):
                raise RuntimeError("Provider, leaf and root token totals differ")
        estimated = usage["cost_estimate"]
        if len(leaves) != estimated["model_calls"]:
            raise RuntimeError("Model call counts differ")
        for field, native in (
            ("input_cost", "prompt_cost"),
            ("output_cost", "completion_cost"),
            ("total_cost", "total_cost"),
        ):
            for actual in (
                float(getattr(root, native) or 0),
                sum(float(getattr(run, native) or 0) for run in leaves),
            ):
                if not math.isclose(estimated[field], actual, rel_tol=1e-5, abs_tol=1e-9):
                    raise RuntimeError("Streamed and LangSmith cost estimates differ")
        report = {
            "verified_at": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "trace_url": root.url or client.get_run_url(run=root),
            "model": settings.model_name,
            "prompt_ref": settings.prompt_ref or "local",
            "planning": args.planning,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cache_read_tokens": usage.get("input_token_details", {}).get("cache_read", 0),
            "cost_estimate": estimated,
            "langsmith_total_cost_usd": str(root.total_cost),
            "verified": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("Measured tokens:", usage["total_tokens"], "Model calls:", len(leaves))
        print("Verified LangSmith cost estimate (USD):", root.total_cost)
        print("Trace:", report["trace_url"])
        print("Report:", args.output)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/usage/latest.json"))
    args = parser.parse_args()
    truststore.inject_into_ssl()
    try:
        verify(args)
    except Exception as error:
        raise SystemExit(f"Usage verification failed: {type(error).__name__}: {error}") from None
