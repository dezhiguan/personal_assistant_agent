"""LangSmith owns experiment execution; this adapter enforces project release checks."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import truststore
from langsmith import Client, aevaluate, tracing_context
from openevals.prompts import CORRECTNESS_PROMPT

from personal_assistant_agent.core.config import get_settings
from personal_assistant_agent.evaluation.dataset import (
    DEFAULT_CASES,
    DEFAULT_DATASET,
    examples,
    sync_dataset,
)
from personal_assistant_agent.evaluation.evaluators import (
    assess,
    make_quality_evaluator,
    needs_quality,
    project_assertions,
)
from personal_assistant_agent.evaluation.golden import Case, finalize_row, source_hash
from personal_assistant_agent.evaluation.scenarios import ROOT, ScenarioTarget
from personal_assistant_agent.services.prompts import load_prompt

QUALITY_FLOOR = 0.8


def result_row(item, *, min_quality=QUALITY_FLOOR):
    case = Case.model_validate(item["example"].inputs["case"])
    observation = item["run"].outputs
    if not observation:
        observation = {"repeat": 0, "error": "target_failed", "turns": [], "workspace": ""}
    row = assess(case, observation)
    feedback = {s.key: s for s in item.get("evaluation_results", {}).get("results", [])}
    if not all(
        key in feedback for key in ("task_success", "semantic", "contract", "execution", "safety")
    ):
        row["error"] = "missing_evaluator_feedback"
        row["failures"].append("execution_error")
    required = needs_quality(case, observation)
    row["quality_required"] = required
    if required:
        quality = feedback.get("judge_quality")
        score = quality.score if quality else None
        valid = type(score) in {int, float} and math.isfinite(score) and 0 <= score <= 1
        if not valid:
            row["error"] = "missing_quality_feedback"
            row["failures"].append("execution_error")
        else:
            row["judge"] = {"score": score, "reason": quality.comment}
            if score < min_quality:
                row["failures"].append(f"semantic_quality_below_{min_quality:g}")
        if "judge_error" in feedback:
            row["error"] = feedback["judge_error"].value
            row["failures"].append("execution_error")
    row["run_id"] = str(item["run"].id)
    return finalize_row(row)


async def run(args):
    os.chdir(ROOT)
    records, digest = examples(args.cases)
    if not records:
        raise ValueError("No selected scenarios")
    settings = get_settings()
    if not settings.model_configured or not settings.langsmith_api_key:
        raise ValueError("Live evaluation requires DASHSCOPE_API_KEY and LANGSMITH_API_KEY")
    if settings.model_name != "qwen3.8-flash":
        raise ValueError("This evaluation profile requires MODEL_NAME=qwen3.8-flash")
    settings = replace(settings, tracing_enabled=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    output = ROOT / "artifacts/golden" / stamp
    output.mkdir(parents=True, mode=0o700)
    (output / "cases.snapshot.json").write_bytes(args.cases.read_bytes())
    client = None
    try:
        client = Client(
            api_key=settings.langsmith_api_key,
            api_url=settings.langsmith_endpoint,
            timeout_ms=15000,
        )
        dataset, _, _ = await asyncio.to_thread(sync_dataset, client, args.cases, args.dataset)
        data, dataset_id = dataset.name, str(dataset.id)
        prompt = await asyncio.to_thread(load_prompt, settings)
        scenario = ScenarioTarget(settings, prompt, output, timeout=args.timeout)

        async def target(inputs):
            return await scenario(inputs)

        evaluators = [project_assertions, make_quality_evaluator(settings)]
        metadata = {
            "model": settings.model_name,
            "judge_model": settings.model_name,
            "judge_package": "openevals:" + version("openevals"),
            "judge_prompt_hash": hashlib.sha256(CORRECTNESS_PROMPT.encode()).hexdigest(),
            "suite_hash": digest,
            "agent_source_hash": source_hash(ROOT, agent_only=True),
            "source_hash": source_hash(ROOT),
            "prompt_ref": prompt.reference,
            "prompt_hash": hashlib.sha256(prompt.text.encode()).hexdigest(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        }
        with tracing_context(enabled=True, client=client):
            results = await aevaluate(
                target,
                data=data,
                client=client,
                evaluators=evaluators,
                experiment_prefix="personal-assistant-golden",
                metadata=metadata,
                max_concurrency=args.concurrency,
                num_repetitions=args.repeats,
                upload_results=True,
                blocking=False,
            )
            rows = []
            async for item in results:
                row = result_row(item, min_quality=args.min_quality)
                rows.append(row)
                path = output / f"{row['id']}-{row['repeat']}.json"
                path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
                path.chmod(0o600)
                print(
                    f"{'PASS' if row['passed'] else 'FAIL'} {row['id']} #{row['repeat']}: "
                    + ",".join(row["failures"]),
                    flush=True,
                )
        expected = {
            (r["metadata"]["case_id"], n) for r in records for n in range(1, args.repeats + 1)
        }
        complete = len(rows) == len(expected) and {(r["id"], r["repeat"]) for r in rows} == expected
        critical = [
            {"id": r["id"], "repeat": r["repeat"], "failures": r["failures"]}
            for r in rows
            if r["critical"] and not r["passed"]
        ]
        passed = sum(r["passed"] for r in rows)
        report = {
            **metadata,
            "created_at": datetime.now(UTC).isoformat(),
            "offline": False,
            "dataset_id": dataset_id,
            "dataset": args.dataset,
            "experiment_name": results.experiment_name,
            "suite_version": "2.0",
            "repeats": args.repeats,
            "expected_cases": len(expected),
            "passed": passed,
            "total": len(rows),
            "min_quality": args.min_quality,
            "gate": {
                "passed": complete and passed == len(expected),
                "reasons": ([] if complete else ["incomplete_coverage"])
                + ([] if not critical else ["critical_failure"])
                + ([] if passed == len(expected) else ["case_failure"]),
            },
            "critical_failures": critical,
            "rows": rows,
            "legacy_baseline": "Retained unchanged; old judge scores are not interchangeable "
            "with the OpenEvals profile. Compare compatible experiments in LangSmith.",
        }
        for path in (output / "report.json", ROOT / "artifacts/golden/latest.json"):
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            path.chmod(0o600)
        print(f"Passed {passed}/{len(expected)}; experiment: {results.experiment_name}")
        print("Report:", output / "report.json")
        print("Critical failures:", len(critical))
        return 0 if report["gate"]["passed"] else 1
    finally:
        if client:
            await asyncio.to_thread(client.flush)
            await asyncio.to_thread(client.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Compatibility flag; live evaluation always includes quality scoring",
    )
    parser.add_argument("--min-quality", type=float, default=QUALITY_FLOOR)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if (
        not 1 <= args.repeats <= 10
        or not 1 <= args.concurrency <= 4
        or args.timeout <= 0
        or not QUALITY_FLOOR <= args.min_quality <= 1
    ):
        parser.error("repeats=1..10, concurrency=1..4, timeout>0, min-quality=0.8..1 required")
    if args.offline:
        if args.cases != DEFAULT_CASES or args.repeats != 2:
            parser.error("Offline compatibility command uses the canonical suite, twice")
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(ROOT / "tests/test_golden_offline.py"),
                ]
            )
        )
    truststore.inject_into_ssl()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except Exception as error:
        print("Evaluation failed:", type(error).__name__)
        if isinstance(error, ValueError) and str(error).startswith(
            ("Live evaluation requires", "This evaluation profile requires")
        ):
            print(str(error))
        raise SystemExit(2) from None
