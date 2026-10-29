"""Run a real LangSmith dataset experiment; model calls are billable."""

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import truststore
from langsmith import Client, get_current_run_tree, tracing_context

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import get_settings
from personal_assistant_agent.evaluation.dataset import DEFAULT_CASES, DEFAULT_DATASET, sync_dataset
from personal_assistant_agent.evaluation.evaluators import make_judge, task_success
from personal_assistant_agent.observability.tracer import CostTracer
from personal_assistant_agent.schemas.chat import ChatRequest


async def run(args) -> None:
    settings = get_settings()
    if not settings.model_configured or not settings.langsmith_api_key:
        raise RuntimeError("Qwen and LangSmith credentials are required")
    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    try:
        dataset, version, count = await asyncio.to_thread(
            sync_dataset, client, args.cases, args.dataset
        )
        graph = await asyncio.to_thread(build_agent, settings)

        async def target(inputs: dict) -> dict:
            from uuid import uuid4

            payload = ChatRequest(conversation_id=uuid4(), **inputs)
            tree = get_current_run_tree()
            tracer = CostTracer(
                settings=settings,
                client=client,
                project_name=tree.session_name if tree else settings.langsmith_project,
            )
            result = await graph.ainvoke(
                {
                    "messages": [m.model_dump() for m in payload.messages],
                    "todos": [t.model_dump() for t in payload.todos],
                },
                config={
                    "callbacks": [tracer],
                    "recursion_limit": 100,
                    "metadata": {"prompt_ref": settings.prompt_ref or "local", "evaluation": True},
                },
            )
            return {"answer": result["messages"][-1].text, "todos": result.get("todos", [])}

        evaluators = [task_success]
        if args.judge:
            evaluators.append(make_judge(settings, client))
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        with tracing_context(enabled=True, client=client):
            results = await client.aevaluate(
                target,
                data=dataset.name,
                evaluators=evaluators,
                experiment_prefix="personal-assistant-regression",
                metadata={
                    "model": settings.model_name,
                    "prompt_ref": settings.prompt_ref or "local",
                    "source_hash": version,
                    "revision": revision,
                },
                description="Synthetic regression with real Qwen calls",
                max_concurrency=2,
                blocking=True,
            )
            rows = []
            async for row in results:
                scores = row.get("evaluation_results", {}).get("results", [])
                rows.append(
                    {
                        "example_id": str(row["example"].id),
                        "run_id": str(row["run"].id),
                        "scores": [
                            {"key": s.key, "score": s.score, "comment": s.comment} for s in scores
                        ],
                        "error": bool(row["run"].error),
                    }
                )
        client.flush()
        passed = sum(
            any(s["key"] == "task_success" and s["score"] == 1 for s in row["scores"])
            and not row["error"]
            for row in rows
        )
        quality_scores = [
            s["score"]
            for row in rows
            for s in row["scores"]
            if s["key"] == "judge_quality" and isinstance(s["score"], (int, float))
        ]
        quality_passed = not args.judge or (
            len(quality_scores) == count and all(s >= args.min_quality for s in quality_scores)
        )
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": str(dataset.id),
            "dataset": dataset.name,
            "source_hash": version,
            "expected_cases": count,
            "experiment_name": results.experiment_name,
            "passed": passed,
            "total": len(rows),
            "judge_enabled": args.judge,
            "min_quality": args.min_quality,
            "quality_mean": sum(quality_scores) / len(quality_scores) if quality_scores else None,
            "quality_gate_passed": quality_passed,
            "prompt_ref": settings.prompt_ref or "local",
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Experiment: {results.experiment_name}")
        print(f"Task checks passed: {passed}/{len(rows)}; expected cases: {count}")
        print("Report:", args.output)
        if args.judge:
            print(f"Quality gate (each score >= {args.min_quality}): {quality_passed}")
        if passed != len(rows) or len(rows) != count or not quality_passed:
            raise RuntimeError("Evaluation gate failed; inspect the experiment report")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--judge", action="store_true", help="Also score with a real Qwen judge")
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/latest.json"))
    args = parser.parse_args()
    if not 0 <= args.min_quality <= 1:
        parser.error("--min-quality must be between 0 and 1")
    truststore.inject_into_ssl()
    try:
        asyncio.run(run(args))
    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
