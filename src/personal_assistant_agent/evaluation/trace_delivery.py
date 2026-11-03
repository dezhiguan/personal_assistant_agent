"""Check native LangSmith ingestion before spending tokens and before releasing."""

from datetime import UTC, datetime
from uuid import uuid4

from langsmith import Client
from langsmith.utils import LangSmithRetry


def check_trace_ingestion(settings):
    # A synchronous SDK write surfaces quota errors instead of logging and
    # dropping them in a background batch. Never retry a monthly quota here.
    client = Client(
        api_key=settings.langsmith_api_key,
        api_url=settings.langsmith_endpoint,
        auto_batch_tracing=False,
        tracing_sampling_rate=1,
        timeout_ms=5000,
        retry_config=LangSmithRetry(total=0),
    )
    try:
        now = datetime.now(UTC)
        client.create_run(
            id=uuid4(),
            name="agent-evaluation-ingestion-check",
            run_type="chain",
            project_name=settings.langsmith_project,
            inputs={"synthetic": True},
            outputs={"purpose": "verify trace ingestion before model execution"},
            start_time=now,
            end_time=now,
        )
    finally:
        client.close(timeout=0)


def missing_trace_runs(client, run_ids):
    expected = set(run_ids)
    if not expected:
        return []
    persisted = {
        str(run.id)
        for run in client.list_runs(
            run_ids=list(expected),
            select=["id", "name", "start_time", "run_type", "trace_id"],
        )
    }
    return sorted(expected - persisted)
