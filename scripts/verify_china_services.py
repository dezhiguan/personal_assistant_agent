"""Report configuration separately from real, read-only China-provider verification."""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langsmith import tracing_context

from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.core.config import get_settings


async def verify(live: bool):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    settings = get_settings()
    tools = {t.name: t for t in build_tools(settings)}
    configured = {
        "get_weather_alert": bool(settings.qweather_host and settings.qweather_api_key.strip()),
        "search_flights": settings.travel_provider == "tuniu"
        and bool(settings.tuniu_api_key.strip()),
        "search_hotels": settings.travel_provider == "tuniu"
        and bool(settings.tuniu_api_key.strip()),
    }
    checks = {
        name: {"configured": ready, "status": "not_verified" if ready else "missing_config"}
        for name, ready in configured.items()
    }
    runtime = SimpleNamespace(
        config={
            "configurable": {
                "travel_context": {
                    "conversation_id": "read-only-check-" + uuid4().hex,
                    "confirmation": "",
                }
            }
        }
    )
    day = (datetime.now(UTC) + timedelta(days=30)).date()
    cases = {
        "get_weather_alert": {"city": "北京", "country_code": "CN"},
        "search_flights": {
            "origin": "北京",
            "destination": "上海",
            "departure_date": day.isoformat(),
        },
        "search_hotels": {
            "city_code": "北京",
            "check_in": day.isoformat(),
            "check_out": (day + timedelta(days=1)).isoformat(),
            "max_results": 1,
        },
    }
    if live:
        for name, args in cases.items():
            if not configured[name]:
                continue
            try:
                if name == "get_weather_alert":
                    result = await tools[name].ainvoke(args)
                else:
                    result = await tools[name].coroutine(runtime=runtime, **args)
                checks[name]["status"] = "failed" if "error" in result else "verified"
                if "error" in result:
                    checks[name]["error"] = result["error"]
                else:
                    rows = result.get("alerts", result.get("results", []))
                    checks[name]["result_count"] = len(rows)
                    if name != "get_weather_alert" and not rows:
                        checks[name]["status"] = "empty_results"
            except Exception as error:
                checks[name].update(status="failed", error_type=type(error).__name__)
    profiles = Path(settings.travel_private_dir) / "profiles"
    # Presence is not validation; no names or profile content enter reports.
    profile_present = profiles.is_dir() and any(profiles.glob("*.json"))
    for name in ("book_flight", "book_hotel"):
        checks[name] = {
            "configured": settings.travel_provider == "tuniu"
            and bool(settings.tuniu_api_key.strip())
            and profile_present,
            "status": "not_verified" if profile_present else "missing_private_profile",
            "note": "本脚本绝不调用创单；真实预订需本人逐单确认，付款在途牛完成。",
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "live_requested": live,
        "provider": settings.travel_provider,
        "checks": checks,
        "read_only_verified": all(checks[n]["status"] == "verified" for n in cases),
        "booking_verified": False,
    }
    output = root / "artifacts/china-services"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = output / "latest.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    path.chmod(0o600)
    for name, item in checks.items():
        print(f"{name}: {item['status']}")
    print(f"Report: {path}")
    return 0 if report["read_only_verified"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="执行真实只读查询，不创建订单")
    args = parser.parse_args()
    with tracing_context(enabled=False):
        raise SystemExit(asyncio.run(verify(args.live)))
