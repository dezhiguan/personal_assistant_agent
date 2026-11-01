"""Run the authored 32-tool scenario through real HTTP; audit real traces and disk.

No substitute model/providers, no real booking confirmation, no private profile
contents. Live provider failures remain explicit blocked results in the report.
"""

import argparse
import csv
import io
import json
import re
import shlex
import sqlite3
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from langsmith import Client

from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.core.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
BUILTINS = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "delete",
    "execute",
    "task",
}


def save(path, data):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def decode(value):
    if isinstance(value, dict) and set(value) == {"output"}:
        return decode(value["output"])
    if (
        isinstance(value, dict)
        and isinstance(value.get("kwargs"), dict)
        and "content" in value["kwargs"]
    ):
        return decode(value["kwargs"]["content"])
    if isinstance(value, dict) and "content" in value and "type" in value:
        return decode(value["content"])
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def verify(args):
    import os

    os.chdir(ROOT)
    settings = get_settings()
    if not settings.tracing_configured:
        raise RuntimeError("真实调用审计需要启用服务端 LangSmith 追踪。")
    stamp = args.audit or (datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    folder = "e2e-travel-" + stamp
    output = ROOT / "artifacts/comprehensive" / stamp
    output.mkdir(parents=True, mode=0o700, exist_ok=bool(args.audit))
    prompt = (
        (ROOT / "evaluations/scenarios/all-tools.txt").read_text().replace("{{RUN_DIR}}", folder)
    )
    conversation = str(uuid4())
    events = []
    if args.audit:
        metadata = output / "metadata.json"
        previous = json.loads(
            (metadata if metadata.exists() else output / "report.json").read_text()
        )
        conversation = previous["conversation_id"]
        events = json.loads((output / "events.json").read_text())
    else:
        save(output / "metadata.json", {"conversation_id": conversation, "folder": folder})
    with httpx.Client(
        timeout=httpx.Timeout(settings.workflow_timeout + 90), trust_env=False
    ) as client:
        if not args.audit:
            client.get(args.url + "/healthz").raise_for_status()
            response_context = client.stream(
                "POST",
                args.url + "/api/chat",
                json={
                    "conversation_id": conversation,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            with response_context as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    events.append(event)
                    if event["type"] == "tool":
                        print(f"{event['status']}: {event['name']}", flush=True)
                    elif event["type"] in {"error", "done"}:
                        print(event["type"] + ": " + event.get("code", ""), flush=True)
    save(output / "events.json", events)
    request_id = events[0]["request_id"]
    trace = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    tool_rows = []
    previous_count = None
    try:
        for attempt in range(12):
            try:
                root_run = trace.read_run(request_id)
                tool_rows = list(
                    trace.list_runs(
                        project_name=settings.langsmith_project,
                        trace_id=root_run.trace_id,
                        run_type="tool",
                        select=[
                            "id",
                            "name",
                            "start_time",
                            "end_time",
                            "inputs",
                            "outputs",
                            "error",
                        ],
                    )
                )
                if (
                    root_run.end_time
                    and tool_rows
                    and all(r.end_time for r in tool_rows)
                    and previous_count == len(tool_rows)
                ):
                    break
                previous_count = len(tool_rows)
            except Exception:
                if attempt == 11:
                    raise RuntimeError("无法读取本次请求的完整工具追踪。") from None
            time.sleep(2)
        else:
            raise RuntimeError("工具追踪尚未完整上传，不能判定通过；稍后使用 --audit 复核。")
    finally:
        trace.close()
    secrets = [
        v
        for k, v in dotenv_values(".env").items()
        if v and len(v) >= 10 and re.search(r"KEY|SECRET|TOKEN|PASSWORD", k, re.I)
    ]

    def clean(value):
        text = json.dumps(value, ensure_ascii=False, default=str)
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        return json.loads(text)

    calls = []
    for row in sorted(tool_rows, key=lambda r: r.start_time):
        calls.append(
            clean(
                {"name": row.name, "inputs": row.inputs, "outputs": row.outputs, "error": row.error}
            )
        )
    save(output / "tools.json", calls)
    names = Counter(c["name"] for c in calls)
    expected = {t.name for t in build_tools(settings)} | BUILTINS
    failures = []

    def require(ok, name):
        if not ok:
            failures.append(name)

    require(bool(events) and events[-1]["type"] == "done", "request_not_completed")
    require(not any(e["type"] == "error" for e in events), "request_error")
    require(not root_run.error, "trace_execution_error")
    require(all(r.end_time for r in tool_rows), "incomplete_tool_trace")
    require(expected <= names.keys(), "missing_tools:" + ",".join(sorted(expected - names.keys())))
    directory = ROOT / settings.workspace_dir / folder
    for filename in ("params.json", "report.md", "review.md", "coverage.csv"):
        require((directory / filename).is_file(), "missing_file:" + filename)
    require(not (directory / "scratch.txt").exists(), "scratch_not_deleted")
    if (directory / "params.json").is_file():
        try:
            params = json.loads((directory / "params.json").read_text())
            require(params.get("budget_cny") == 5000, "budget_not_5000")
        except ValueError:
            failures.append("invalid_params_json")
    coverage_discrepancies = []
    if (directory / "coverage.csv").is_file():
        content = (directory / "coverage.csv").read_text()
        reported_rows = [
            row for row in csv.reader(io.StringIO(content)) if row and row[0] in expected
        ]
        require(expected <= {row[0] for row in reported_rows}, "incomplete_coverage_csv")
        for row in reported_rows:
            require(
                len(row) >= 4 and row[2] in {"success", "expected_rejection", "blocked", "failed"},
                "invalid_coverage_row:" + row[0],
            )
            try:
                claimed = int(row[1])
                require(claimed >= 0, "invalid_coverage_count:" + row[0])
                if claimed != names[row[0]]:
                    coverage_discrepancies.append(
                        {"name": row[0], "self_reported": claimed, "measured": names[row[0]]}
                    )
            except (ValueError, IndexError):
                failures.append("invalid_coverage_count:" + row[0])
    outputs = [(c, decode(c.get("outputs"))) for c in calls]
    require(
        any(
            c["name"] == "baidu_search"
            and not c.get("error")
            and isinstance(v, dict)
            and (not v.get("error") or v.get("code") == "BAIDU_CHALLENGE")
            for c, v in outputs
        ),
        "unexpected_baidu_failure",
    )
    expected_blocks = {"baidu_search"}
    if not settings.qweather_host or not settings.qweather_api_key:
        expected_blocks.add("get_weather_alert")
    custom_names = {t.name for t in build_tools(settings)}
    for name in custom_names - expected_blocks - {"book_flight", "book_hotel"}:
        require(
            any(
                c["name"] == name
                and not c.get("error")
                and isinstance(v, dict)
                and not v.get("error")
                for c, v in outputs
            ),
            "no_successful_result:" + name,
        )
    amap_expected = {
        ("search", ""),
        ("geocode", ""),
        ("reverse_geocode", ""),
        ("nearby", ""),
        ("route", "driving"),
        ("route", "walking"),
        ("route", "bicycling"),
    }
    amap_observed = {
        (
            c["inputs"].get("action"),
            c["inputs"].get("mode", "driving") if c["inputs"].get("action") == "route" else "",
        )
        for c, v in outputs
        if c["name"] == "amap_maps"
        and not c.get("error")
        and isinstance(v, dict)
        and not v.get("error")
    }
    require(amap_expected <= amap_observed, "missing_amap_operations")
    clocks = [v for c, v in outputs if c["name"] == "get_current_time" and isinstance(v, dict)]
    today = date.fromisoformat(clocks[0]["date"]) if clocks and clocks[0].get("date") else None
    if today:
        flight_dates = {
            c["inputs"].get("departure_date")
            for c, v in outputs
            if c["name"] == "search_flights" and isinstance(v, dict) and not v.get("error")
        }
        require(
            {(today + timedelta(days=n)).isoformat() for n in (7, 8)} <= flight_dates,
            "missing_outbound_or_return_flight",
        )
    require(
        any(
            c["name"] == "calculate" and isinstance(v, dict) and v.get("result") == 600
            for c, v in outputs
        ),
        "missing_verified_600",
    )
    require(
        any(
            c["name"] == "convert_units" and isinstance(v, dict) and v.get("result") == 5000
            for c, v in outputs
        ),
        "missing_verified_5000m",
    )
    require(
        any(
            c["name"] == "validate_time_budget"
            and isinstance(v, dict)
            and v.get("total") == 480
            and v.get("fits")
            and v.get("matches_expected")
            for c, v in outputs
        ),
        "missing_verified_480",
    )
    for name in ("book_flight", "book_hotel"):
        found = [c for c in calls if c["name"] == name]
        require(bool(found), "missing_rejection:" + name)
        for call in found:
            require(call["inputs"].get("quote_id") == "0" * 32, "unsafe_booking_quote:" + name)
            result = decode(call.get("outputs"))
            require(
                isinstance(result, dict) and bool(result.get("error")),
                "booking_not_rejected:" + name,
            )
    calendars = []
    for call, result in outputs:
        if call["name"] not in {"create_calendar_event", "set_reminder"}:
            continue
        require(
            isinstance(result, dict) and result.get("status") == "pending_import",
            "calendar_not_pending_import",
        )
        if isinstance(result, dict) and result.get("file_path"):
            path = ROOT / settings.workspace_dir / result["file_path"].lstrip("/")
            require(path.is_file() and "BEGIN:VALARM" in path.read_text(), "calendar_file_missing")
            if today and result.get("start_utc"):
                start = datetime.fromisoformat(result["start_utc"])
                expected_hour = 12 if call["name"] == "create_calendar_event" else 10
                require(
                    start.date() == today + timedelta(days=6) and start.hour == expected_hour,
                    "incorrect_calendar_date_or_timezone",
                )
            calendars.append(result)
    require(len(calendars) >= 2, "missing_two_calendar_files")
    file_mutations = {"write_file", "edit_file", "delete"}
    for call in calls:
        inputs = call.get("inputs") or {}
        if call["name"] in file_mutations and not call.get("error"):
            path = inputs.get("file_path", "")
            require(path.startswith("/" + folder + "/"), "write_outside_test_directory")
        if call["name"] == "execute":
            argv = shlex.split(inputs.get("command", ""))
            require(
                bool(argv) and argv[0] in {"mkdir", "wc", "ls", "pwd", "cat", "head", "tail"},
                "unexpected_execute_command",
            )
    database = ROOT / settings.travel_private_dir / "bookings.sqlite3"
    if database.exists():
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
            orders = [
                json.loads(row[0])
                for row in conn.execute(
                    "SELECT data FROM quotes WHERE json_extract(data, '$.owner') = ?",
                    (conversation,),
                )
            ]
        require(
            not any(r.get("owner") == conversation and r.get("state") != "open" for r in orders),
            "unexpected_order_or_booking_intent",
        )
    blocked = [
        {"name": c["name"], "error": v.get("error")}
        for c, v in outputs
        if isinstance(v, dict) and v.get("error") and c["name"] not in {"book_flight", "book_hotel"}
    ]
    plans = [e["todos"] for e in events if e["type"] == "plan"]
    # This is measured evidence, independent of the assistant's self-counted CSV.
    measured = io.StringIO()
    writer = csv.writer(measured)
    writer.writerow(["tool_name", "actual_calls", "status"])
    for name in sorted(expected):
        status = "success"
        if name in {"book_flight", "book_hotel"}:
            status = "expected_rejection"
        elif any(b["name"] == name for b in blocked):
            status = "blocked"
        writer.writerow([name, names[name], status])
    measured_path = output / "observed_coverage.csv"
    measured_path.touch(mode=0o600, exist_ok=True)
    measured_path.write_text(measured.getvalue())
    report = {
        "request_id": request_id,
        "conversation_id": conversation,
        "directory": str(directory),
        "passed": not failures,
        "failures": failures,
        "tool_coverage": f"{len(expected & names.keys())}/{len(expected)}",
        "calls": dict(names),
        "blocked": blocked,
        "todos": plans[-1] if plans else [],
        "calendar_files": calendars,
        "provider_success": not blocked,
        "measured_coverage_file": str(measured_path),
        "self_reported_coverage_discrepancies": coverage_discrepancies,
    }
    save(output / "report.json", report)
    save(ROOT / "artifacts/comprehensive/latest.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print(f"Report: {output / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--audit", help="Re-audit an existing artifact stamp without new model calls"
    )
    raise SystemExit(verify(parser.parse_args()))
