"""Release checks fail closed when native experiment feedback is absent or invalid."""

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from langchain_openai import ChatOpenAI

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.evaluation import evaluators
from personal_assistant_agent.evaluation.dataset import examples, sync_dataset
from personal_assistant_agent.evaluation.runner import result_row


def native_result():
    record = next(r for r in examples()[0] if r["metadata"]["case_id"] == "format-translation")
    observed = {
        "repeat": 1,
        "workspace": ".",
        "turns": [
            {
                "answer": "**Hello**, world!",
                "todos": [],
                "tools": [],
                "done": True,
                "errors": [],
            }
        ],
    }
    scores = evaluators.project_assertions(record["inputs"], observed, record["outputs"])
    assert scores[0]["score"]
    return {
        "example": SimpleNamespace(**record),
        "run": SimpleNamespace(id=uuid4(), outputs=observed),
        "evaluation_results": {"results": [SimpleNamespace(**s) for s in scores]},
    }


@pytest.mark.parametrize(
    "score,passed",
    [(0.8, True), (1, True), (0.79, False), (float("nan"), False), (True, False), (None, False)],
)
def test_quality_threshold_and_missing_feedback(score, passed):
    item = native_result()
    item["evaluation_results"]["results"].append(
        SimpleNamespace(key="judge_quality", score=score, comment="Synthetic feedback")
    )
    assert result_row(item)["passed"] is passed


def test_missing_assertion_feedback_is_blocking():
    item = native_result()
    item["evaluation_results"]["results"] = [
        SimpleNamespace(key="judge_quality", score=1, comment="Good")
    ]
    assert not result_row(item)["passed"]
    assert result_row(item)["error"] == "missing_evaluator_feedback"


def test_trace_failure_stops_paid_judging(monkeypatch):
    scorer = MagicMock(side_effect=AssertionError("Do not spend tokens after ingestion failure"))
    monkeypatch.setattr(evaluators, "create_async_llm_as_judge", lambda **kwargs: scorer)
    monkeypatch.setattr(evaluators, "build_model", lambda *args, **kwargs: MagicMock())
    item = native_result()
    quality = evaluators.make_quality_evaluator(
        Settings(), trace_errors=["LangSmithRateLimitError"]
    )
    result = asyncio.run(
        quality(item["example"].inputs, item["run"].outputs, item["example"].outputs)
    )
    assert result[1] == {"key": "judge_error", "value": "TraceDeliveryUnavailable"}
    scorer.assert_not_called()


@pytest.mark.parametrize(
    "response,passed",
    [
        ('{"score":0.9,"reasoning":"Correct"}', True),
        ("not json", False),
        ('{"score":2,"reasoning":"Invalid"}', False),
        ('{"score":-Infinity,"reasoning":"Invalid"}', False),
        ('{"score":true,"reasoning":"Invalid"}', False),
        ('{"score":0,"reasoning":""}', False),
        ('{"score":0,"reasoning":"   "}', False),
    ],
)
def test_official_openevals_parsing_with_fake_model(monkeypatch, response, passed):
    def respond(request):
        payload = json.loads(request.content)
        actual_input = json.loads(
            re.search(
                r"<input>\s*(.*?)\s*</input>",
                payload["messages"][-1]["content"],
                re.S,
            )[1]
        )
        assert actual_input["initial_todos"] == [{"content": "原有待办", "status": "pending"}]
        assert actual_input["user_turns"] == [
            {
                "turn": 1,
                "role": "user",
                "content": "将“你好，世界！”翻译为英文，只输出译文。",
            }
        ]
        actual_output = json.loads(
            re.search(r"<output>\s*(.*?)\s*</output>", payload["messages"][-1]["content"], re.S)[1]
        )
        assert actual_output[0]["role"] == "assistant"
        assert actual_output[0]["tool_receipts"][0]["outputs"]["exit_code"] == 1
        assert actual_output[0]["graph_tool_receipts"] == [
            {"tool_call_id": "cached", "cached": True, "content": "Synthetic failed search"}
        ]
        reference = json.loads(
            re.search(
                r"<reference_outputs>\s*(.*?)\s*</reference_outputs>",
                payload["messages"][-1]["content"],
                re.S,
            )[1]
        )
        assert reference["length_measurements"] == [
            {"turn": 1, "count": 12, "limit": 25, "mode": "visible"}
        ]
        assert "as a whole before model or tool execution" in reference["credential_guard"]
        assert "unnecessary refusal" in reference["credential_guard"]
        assert "response_format" not in payload
        assert payload["temperature"] == 0
        assert payload["tool_choice"]["function"]["name"] == "CorrectnessResult"
        schema = payload["tools"][0]["function"]["parameters"]
        assert schema["properties"]["score"]["minimum"] == 0
        assert schema["properties"]["score"]["maximum"] == 1
        assert schema["properties"]["reasoning"]["minLength"] == 1
        assert "maxLength" not in schema["properties"]["reasoning"]
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "model": "fake",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "scoring",
                                    "type": "function",
                                    "function": {
                                        "name": "CorrectnessResult",
                                        "arguments": response,
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model="fake", api_key="synthetic", http_async_client=client)
            monkeypatch.setattr(evaluators, "build_model", lambda *a, **k: model)
            item = native_result()
            item["example"].inputs["case"]["todos"] = [{"content": "原有待办", "status": "pending"}]
            item["run"].outputs["turns"][0]["tools"] = ["execute"]
            item["run"].outputs["turns"][0]["tool_receipts"] = [
                {
                    "name": "execute",
                    "inputs": {"command": "cat /../blocked"},
                    "outputs": {"exit_code": 1, "output": "PermissionError"},
                }
            ]
            item["run"].outputs["turns"][0]["graph_tool_receipts"] = [
                {"tool_call_id": "cached", "cached": True, "content": "Synthetic failed search"}
            ]
            score = await evaluators.make_quality_evaluator(Settings())(
                item["example"].inputs, item["run"].outputs, item["example"].outputs
            )
            assert isinstance(score, dict) is passed
            if passed:
                assert score["score"] == 0.9
            else:
                assert score[1]["key"] == "judge_error"

    asyncio.run(check())


@pytest.mark.parametrize(
    "responses,attempts,expected,retries",
    [
        ([{"score": 0.75, "reasoning": "Incomplete"}], 1, 0.75, 0),
        ([{"score": 1, "reasoning": "Evidence. " * 80}], 1, 1, 0),
        (
            [{"score": 1, "reason": "Wrong field"}, {"score": 0.75, "reasoning": "Incomplete"}],
            2,
            0.75,
            1,
        ),
        ([{"score": 0, "reasoning": ""}] * 2, 2, None, 0),
        ([RuntimeError("Synthetic provider failure")], 1, None, 0),
        ([TimeoutError("Synthetic timeout")], 1, None, 0),
    ],
)
def test_only_invalid_scoring_format_gets_one_native_retry(
    monkeypatch, responses, attempts, expected, retries
):
    calls = []

    async def scorer(**kwargs):
        response = responses[len(calls)]
        calls.append(kwargs)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(evaluators, "create_async_llm_as_judge", lambda **kwargs: scorer)
    monkeypatch.setattr(evaluators, "build_model", lambda *args, **kwargs: MagicMock())

    async def check():
        item = native_result()
        feedback = await evaluators.make_quality_evaluator(Settings())(
            item["example"].inputs, item["run"].outputs, item["example"].outputs
        )
        values = feedback if isinstance(feedback, list) else [feedback]
        by_key = {value["key"]: value for value in values}
        assert len(calls) == attempts
        if expected is None:
            assert "judge_error" in by_key
        else:
            assert by_key["judge_quality"]["score"] == expected
            assert "judge_error" not in by_key
        assert by_key.get("judge_format_retries", {}).get("score", 0) == retries

    asyncio.run(check())


def test_sync_reuses_ids_and_preserves_unrelated_cloud_examples():
    client = MagicMock()
    dataset = SimpleNamespace(id=uuid4(), name="synthetic")
    client.list_datasets.return_value = iter([dataset])
    client.list_examples.return_value = []
    sync_dataset(client, name=dataset.name)
    created = client.create_examples.call_args.kwargs["examples"]
    client.reset_mock()
    client.list_datasets.return_value = iter([dataset])
    client.list_examples.return_value = [
        SimpleNamespace(**{**r, "metadata": {**r["metadata"], "dataset_split": ["base"]}})
        for r in created
    ]
    sync_dataset(client, name=dataset.name)
    client.create_examples.assert_not_called()
    client.update_example.assert_not_called()
    client.list_datasets.return_value = iter([dataset])
    client.list_examples.return_value.append(SimpleNamespace(id=uuid4()))
    with pytest.raises(ValueError, match="extra examples"):
        sync_dataset(client, name=dataset.name)
    client.delete_example.assert_not_called()
    client.update_example.assert_not_called()
