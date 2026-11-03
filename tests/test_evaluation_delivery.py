"""Quota failures must stop paid scenarios and remain release failures."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import requests
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langsmith import Client
from langsmith.utils import LangSmithRateLimitError

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.evaluation import runner, scenarios, trace_delivery
from personal_assistant_agent.evaluation.dataset import DEFAULT_CASES, examples
from personal_assistant_agent.evaluation.evaluators import project_assertions
from personal_assistant_agent.services.chat import ChatService


@pytest.mark.parametrize("status", [200, 429])
def test_ingestion_preflight_uses_synchronous_native_sdk_write(monkeypatch, status):
    session = requests.Session()
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = (
            b'{"error":"Monthly unique traces usage limit exceeded"}' if status == 429 else b"{}"
        )
        return response

    monkeypatch.setattr(session, "request", request)
    clients = []

    def client(**kwargs):
        assert kwargs["auto_batch_tracing"] is False
        assert kwargs["retry_config"].total == 0
        assert kwargs["tracing_sampling_rate"] == 1
        clients.append(Client(session=session, **kwargs))
        return clients[-1]

    monkeypatch.setattr(trace_delivery, "Client", client)
    settings = Settings(langsmith_api_key="synthetic", langsmith_endpoint="http://langsmith.test")
    if status == 429:
        with pytest.raises(LangSmithRateLimitError):
            trace_delivery.check_trace_ingestion(settings)
    else:
        trace_delivery.check_trace_ingestion(settings)
    assert len(calls) == 1
    assert calls[0][:2] == ("POST", "http://langsmith.test/runs")
    payload = json.loads(calls[0][2]["data"])
    assert payload["inputs"] == {"synthetic": True}
    assert payload["end_time"] == payload["start_time"]


def test_failed_preflight_does_not_invoke_models_or_start_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    settings = Settings(
        model_name="qwen3.8-flash", api_key="synthetic", langsmith_api_key="synthetic"
    )
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    probe = Mock(side_effect=LangSmithRateLimitError("Monthly unique traces usage limit exceeded"))
    monkeypatch.setattr(runner, "check_trace_ingestion", probe)
    evaluation = Mock(side_effect=AssertionError("No paid experiment after failed ingestion"))
    monkeypatch.setattr(runner, "aevaluate", evaluation)
    model = Mock(side_effect=AssertionError("No model construction after failed ingestion"))
    monkeypatch.setattr(runner, "make_quality_evaluator", model)
    args = SimpleNamespace(cases=DEFAULT_CASES, repeats=2)
    assert asyncio.run(runner.run(args)) == 2
    evaluation.assert_not_called()
    model.assert_not_called()
    report = json.loads((tmp_path / "artifacts/golden/latest.json").read_text())
    assert report["execution_status"] == "not_started"
    assert report["expected_cases"] == 122 and report["total"] == 0
    assert report["critical_checks_completed"] == 0
    assert not report["gate"]["passed"]
    assert "langsmith_monthly_trace_limit" in report["gate"]["reasons"]


def test_missing_experiment_targets_are_not_accepted_as_delivered():
    ids = [str(uuid4()) for _ in range(122)]
    client = SimpleNamespace(list_runs=Mock(return_value=[SimpleNamespace(id=i) for i in ids]))
    assert trace_delivery.missing_trace_runs(client, ids) == []
    client.list_runs.return_value = [SimpleNamespace(id=i) for i in ids[:-1]]
    assert trace_delivery.missing_trace_runs(client, ids) == ids[-1:]
    assert set(client.list_runs.call_args.kwargs["run_ids"]) == set(ids)


def test_shared_client_lifetime_belongs_to_experiment(monkeypatch):
    shared = Mock(spec=Client)
    factory = Mock(side_effect=AssertionError("Do not create a second tracing client"))
    monkeypatch.setattr("personal_assistant_agent.services.chat.Client", factory)
    service = ChatService(
        object(), Settings(langsmith_api_key="synthetic", tracing_enabled=True), trace_client=shared
    )
    assert service.trace_client is shared
    asyncio.run(service.close())
    shared.flush.assert_not_called()
    shared.close.assert_not_called()


def test_owned_client_closes_once_with_bounded_native_drain(monkeypatch):
    owned = Mock(spec=Client)
    monkeypatch.setattr("personal_assistant_agent.services.chat.Client", Mock(return_value=owned))
    service = ChatService(object(), Settings(langsmith_api_key="synthetic", tracing_enabled=True))
    asyncio.run(service.close())
    owned.close.assert_called_once_with(timeout=10)
    owned.flush.assert_not_called()


@pytest.mark.parametrize("failure", [None, "upload", "missing", "early"])
def test_case_success_cannot_override_native_trace_delivery_failure(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    record = next(r for r in examples()[0] if r["metadata"]["case_id"] == "schema-empty")
    monkeypatch.setattr(runner, "examples", lambda path: ([record], "synthetic-suite"))
    monkeypatch.setattr(runner, "check_trace_ingestion", lambda settings: None)
    settings = Settings(
        model_name="qwen3.8-flash", api_key="synthetic", langsmith_api_key="synthetic"
    )
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kwargs: "base" if "HEAD" in command else "",
    )
    monkeypatch.setattr(
        runner,
        "sync_dataset",
        lambda *args: (SimpleNamespace(name="synthetic", id=uuid4()), None, 1),
    )
    monkeypatch.setattr(
        runner, "load_prompt", lambda *args: SimpleNamespace(text="test", reference="local")
    )
    monkeypatch.setattr(runner, "make_quality_evaluator", lambda *args, **kwargs: object())
    ids = []
    shared = SimpleNamespace(close=Mock(), list_runs=Mock())

    def client(**kwargs):
        shared.error_callback = kwargs["tracing_error_callback"]
        return shared

    def flush(**kwargs):
        if failure == "upload":
            shared.error_callback(LangSmithRateLimitError("synthetic quota failure"))
        shared.list_runs.return_value = (
            [] if failure == "missing" else [SimpleNamespace(id=i) for i in ids]
        )

    shared.flush = Mock(side_effect=flush)
    monkeypatch.setattr(runner, "Client", client)
    calls = []

    class Scenario:
        def __init__(self, *args, **kwargs):
            assert kwargs["trace_client"] is shared

        async def __call__(self, inputs):
            calls.append(inputs)
            return {
                "id": "schema-empty",
                "repeat": len(calls),
                "workspace": ".",
                "turns": [],
                "engine_failures": [],
                "error": None,
            }

    monkeypatch.setattr(runner, "ScenarioTarget", Scenario)

    class Results:
        experiment_name = "synthetic-experiment"

        async def __aiter__(self):
            if failure == "early":
                shared.error_callback(LangSmithRateLimitError("synthetic quota failure"))
                with pytest.raises(RuntimeError, match="trace delivery unavailable"):
                    await target(record["inputs"])
                return
            for _ in range(2):
                observed = await target(record["inputs"])
                run_id = uuid4()
                ids.append(str(run_id))
                yield {
                    "example": SimpleNamespace(**record),
                    "run": SimpleNamespace(id=run_id, outputs=observed),
                    "evaluation_results": {
                        "results": [
                            SimpleNamespace(**f)
                            for f in project_assertions(
                                record["inputs"], observed, record["outputs"]
                            )
                        ]
                    },
                }

    target = None

    async def evaluate(fn, **kwargs):
        nonlocal target
        target = fn
        return Results()

    monkeypatch.setattr(runner, "aevaluate", evaluate)
    args = SimpleNamespace(
        cases=DEFAULT_CASES,
        dataset="synthetic",
        repeats=2,
        timeout=180,
        concurrency=2,
        min_quality=0.8,
    )
    assert asyncio.run(runner.run(args)) == (0 if failure is None else 1)
    report = json.loads((tmp_path / "artifacts/golden/latest.json").read_text())
    assert report["passed"] == (0 if failure == "early" else 2)
    assert report["gate"]["passed"] is (failure is None)
    if failure in {"upload", "early"}:
        assert "trace_ingestion_failed" in report["gate"]["reasons"]
    if failure == "missing":
        assert set(report["trace_delivery"]["missing_target_runs"]) == set(ids)
    assert len(calls) == (0 if failure == "early" else 2)
    shared.close.assert_called_once_with(timeout=0)


def test_native_sdk_local_recording_runs_real_graph_without_langsmith_requests(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(scenarios, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kwargs: "base" if "HEAD" in command else "",
    )
    records = [
        r
        for r in examples()[0]
        if r["metadata"]["case_id"] in {"arithmetic-product", "format-translation"}
    ]
    monkeypatch.setattr(runner, "examples", lambda path: (records, "synthetic-suite"))
    monkeypatch.setattr(
        runner, "get_settings", lambda: Settings(model_name="qwen3.8-flash", api_key="synthetic")
    )
    cloud_calls = []

    def no_cloud(*args, **kwargs):
        cloud_calls.append(args)
        raise AssertionError("Local SDK recording must not access LangSmith")

    monkeypatch.setattr(requests.Session, "request", no_cloud)
    monkeypatch.setattr(runner, "check_trace_ingestion", no_cloud)
    monkeypatch.setattr(runner, "sync_dataset", no_cloud)
    monkeypatch.setattr(runner, "missing_trace_runs", no_cloud)
    model_calls = []
    grader_calls = []

    async def stream(self, messages, **kwargs):
        model_calls.append(messages)
        user = next(m for m in reversed(messages) if m.type == "human")
        if "翻译" in str(user.content):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Hello, world!"))
            return
        done = any(isinstance(m, ToolMessage) and m.name == "calculate" for m in messages)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="391" if done else "",
                tool_calls=[]
                if done
                else [{"name": "calculate", "id": "product", "args": {"expression": "17*23"}}],
            )
        )

    async def score(self, messages, **kwargs):
        grader_calls.append(messages)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "CorrectnessResult",
                                "id": "score",
                                "args": {
                                    "score": 1,
                                    "reasoning": "The answer matches the synthetic reference.",
                                },
                            }
                        ],
                    )
                )
            ]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", stream)
    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", score)
    previous = tmp_path / "artifacts/golden/latest.json"
    previous.parent.mkdir(parents=True)
    previous.write_text('{"cloud_gate_passed":false}')
    args = SimpleNamespace(
        cases=DEFAULT_CASES,
        dataset="synthetic",
        repeats=2,
        concurrency=2,
        timeout=180,
        min_quality=0.8,
        local=True,
    )
    assert asyncio.run(runner.run(args)) == 0
    report = json.loads((tmp_path / "artifacts/golden/local-latest.json").read_text())
    assert report["gate"] == {"scope": "local_behavior", "passed": True, "reasons": []}
    assert report["recording"] == "local" and report["passed"] == report["total"] == 4
    assert report["cloud_validation_passed"] is False
    assert report["trace_delivery"]["requested"] is False
    assert json.loads(previous.read_text()) == {"cloud_gate_passed": False}
    assert len(model_calls) == 6 and len(grader_calls) == 2
    assert cloud_calls == []
    assert all(
        row["turns"][0]["tools"] == ["calculate"]
        for row in report["rows"]
        if row["id"] == "arithmetic-product"
    )
