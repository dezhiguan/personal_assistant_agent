"""Read-only live AMap search check; credentials come only from server settings."""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from personal_assistant_agent.agents.amap_tools import build_amap_tools
from personal_assistant_agent.core.config import get_settings


async def main(args):
    os.chdir(Path(__file__).resolve().parents[1])
    result = await build_amap_tools(get_settings())[0].ainvoke(
        {
            "action": "search",
            "query": args.query,
            "city": args.city,
            "max_results": 3,
        }
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "result": result,
        "passed": "error" not in result and bool(result.get("results")),
    }
    path = Path("artifacts/amap/latest.json")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PASS amap_maps" if report["passed"] else "FAIL amap_maps")
    if result.get("error"):
        print(result["error"])
    elif not result.get("results"):
        print("未获得地点候选，请核对搜索条件。")
    print(f"Report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="故宫")
    parser.add_argument("--city", default="北京市")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
