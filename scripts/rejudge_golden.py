"""Explicit scorer calibration using unchanged, recorded target observations.

Only scoring definitions may change. Never use replay to validate new agent code.
"""

import argparse
import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import truststore
from langsmith import tracing_context
from run_golden import JUDGE_PROMPT, ROOT, needs_judge

from personal_assistant_agent.agents.model import build_model
from personal_assistant_agent.core.config import get_settings
from personal_assistant_agent.evaluation.evaluators import Judgement
from personal_assistant_agent.evaluation.golden import (
    check_turn,
    finalize_row,
    gate,
    load_cases,
    source_hash,
)


def inputs(case):
    value = case.model_dump()
    value.pop("rubric")
    for turn in value["turns"]:
        turn.pop("checks")
    return value


async def run(args):
    args.source = args.source.resolve()
    old = json.loads((args.source / "report.json").read_text())
    original, original_hash = load_cases(args.source / "cases.snapshot.json")
    current, current_hash = load_cases(ROOT / "evaluations/golden/cases.json")
    if original_hash != old["suite_hash"] or [inputs(c) for c in original] != [
        inputs(c) for c in current
    ]:
        raise ValueError("Only scoring definitions may change during calibration")
    target_hash = source_hash(ROOT, agent_only=True)
    if old.get("agent_source_hash"):
        if target_hash != old["agent_source_hash"]:
            raise ValueError("Source changed: run fresh target observations instead")
    elif source_hash(ROOT) != old["source_hash"]:
        raise ValueError("Source changed: missing verified agent-only fingerprint")
    settings = get_settings()
    if settings.model_name != old["model"]:
        raise ValueError("Target model changed: fresh evaluation required")
    if (settings.prompt_ref or "local") != old["prompt_ref"]:
        raise ValueError("Target prompt changed: fresh evaluation required")
    judge = build_model(replace(settings, model_name=old["judge_model"]), max_tokens=400).bind(
        response_format={"type": "json_object"}
    )
    destination = (
        ROOT / "artifacts/golden" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-calibrated")
    )
    destination.mkdir(parents=True, mode=0o700)
    rows = []
    for case, previous_case in zip(current, original, strict=True):
        for repeat in range(1, old["repeats"] + 1):
            row = json.loads((args.source / f"{case.id}-{repeat}.json").read_text())
            before = list(row["failures"])
            row.pop("judge", None)
            if case.engine == "live" and not row["error"]:
                failures, previous = [], case.todos
                for index, (turn, observed) in enumerate(
                    zip(case.turns, row["turns"], strict=True)
                ):
                    checked = check_turn(
                        turn.checks,
                        observed["answer"],
                        observed["todos"],
                        previous,
                        observed["tools"],
                        args.source / "workspaces" / f"{case.id}-{repeat}",
                    )
                    checked += [
                        f for f in observed["failures"] if f in ("missing_done", "service_error")
                    ]
                    observed["failures"] = checked
                    failures += [f"turn{index + 1}:{f}" for f in checked]
                    previous = observed["todos"]
                row["failures"] = failures
                if needs_judge(case, row):
                    with tracing_context(enabled=False):
                        result = await judge.ainvoke(
                            [
                                {"role": "system", "content": JUDGE_PROMPT},
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "rubric": case.rubric,
                                            "requests": [t.prompt for t in case.turns],
                                            "outputs": [
                                                {k: t[k] for k in ("answer", "todos", "tools")}
                                                for t in row["turns"]
                                            ],
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            ]
                        )
                    score = Judgement.model_validate_json(result.content)
                    row["judge"] = score.model_dump()
                    if score.score < 0.8:
                        row["failures"].append("semantic_quality_below_0.8")
            finalize_row(row)
            row["calibration_previous_failures"] = before
            print(f"REVIEW {case.id} #{repeat}: {before} -> {row['failures']}", flush=True)
            target = destination / f"{case.id}-{repeat}.json"
            target.write_text(json.dumps(row, ensure_ascii=False, indent=2))
            target.chmod(0o600)
            rows.append({k: v for k, v in row.items() if k != "turns"})
    report = {
        **old,
        "suite_version": "2.0",
        "suite_hash": current_hash,
        "rows": rows,
        "calibrated_from": str(args.source.relative_to(ROOT)),
        "agent_source_hash": target_hash,
        "evaluation_source_hash": source_hash(ROOT),
        "judge_prompt_hash": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "calibration_runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_hash": hashlib.sha256(
            (ROOT / "src/personal_assistant_agent/evaluation/golden.py").read_bytes()
        ).hexdigest(),
        "calibrated_at": datetime.now(UTC).isoformat(),
        "gate": gate(rows, None, current_hash, old["repeats"]),
    }
    (destination / "cases.snapshot.json").write_bytes(
        (ROOT / "evaluations/golden/cases.json").read_bytes()
    )
    for target in [
        destination / "report.json",
        ROOT / "artifacts/golden/latest.json",
        args.write_baseline,
    ]:
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Calibrated {len(rows)} observations; target answers were not regenerated")
    print("Gate:", report["gate"])
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--write-baseline", type=Path, required=True)
    args = parser.parse_args()
    truststore.inject_into_ssl()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except Exception as error:
        print("Calibration failed:", type(error).__name__)
        raise SystemExit(2) from None
