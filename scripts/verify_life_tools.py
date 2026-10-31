"""Verify keyless tools against real public services, without booking or paid model calls."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.core.config import get_settings


async def main():
    root = Path(__file__).resolve().parents[1]
    import os

    os.chdir(root)
    output = root / "artifacts/life-tools"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings = replace(
        get_settings(), workspace_dir=f"artifacts/life-tools/workspace-{uuid4().hex}"
    )
    tools = {t.name: t for t in build_tools(settings)}
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    cases = {
        "geocode": {"city": "北京", "country_code": "CN"},
        "get_city_weather": {"city": "上海", "country_code": "CN", "days": 1},
        "convert_currency": {"amount": 100, "from_currency": "USD", "to_currency": "CNY"},
        "convert_units": {"value": 0, "from_unit": "degC", "to_unit": "degF"},
        "check_cn_holiday": {"date_str": "2026-10-01"},
        "fetch_url": {"url": "https://example.com", "max_chars": 500},
        "fetch_rss": {"url": "https://feeds.bbci.co.uk/news/rss.xml", "max_items": 2},
        "translate": {"text": "Hello world", "source_language": "en", "target_language": "zh-CN"},
        "ip_locate": {"ip": "8.8.8.8"},
        "route_plan": {
            "origin_latitude": 52.517037,
            "origin_longitude": 13.38886,
            "destination_latitude": 52.529407,
            "destination_longitude": 13.397634,
        },
        "create_calendar_event": {
            "title": "Synthetic integration check",
            "start": tomorrow,
            "end": (datetime.now(UTC) + timedelta(days=1, hours=1)).isoformat(),
            "reminder_minutes": 15,
        },
        "set_reminder": {"title": "Synthetic reminder check", "remind_at": tomorrow},
    }
    results = {}

    async def check(name, args):
        try:
            result = await tools[name].ainvoke(args)
            ok = "error" not in result
            if ok and name == "fetch_rss":
                ok = bool(result.get("items"))
            if ok and name in {"create_calendar_event", "set_reminder"}:
                path = root / settings.workspace_dir / result["file_path"].lstrip("/")
                ok = path.is_file() and "BEGIN:VEVENT" in path.read_text()
            results[name] = {"passed": ok, "result": result}
        except Exception as error:
            results[name] = {"passed": False, "error_type": type(error).__name__}
        print(("PASS " if results[name]["passed"] else "FAIL ") + name, flush=True)

    await asyncio.gather(*(check(name, args) for name, args in cases.items()))
    pending = [
        "amap_maps",
        "get_weather_alert",
        "search_flights",
        "search_hotels",
        "book_flight",
        "book_hotel",
    ]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "checks": results,
        "passed": all(row["passed"] for row in results.values()),
        "not_live_verified": pending,
        "note": "本脚本只验收无需密钥工具。高德、预警与旅行集成需另行配置并验收。",
    }
    path = output / "latest.json"
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {path}")
    print("Not live verified: " + ", ".join(pending))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
