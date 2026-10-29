"""Provider usage and tier/cache-aware estimated costs must remain distinguishable."""

from unittest.mock import patch

import pytest
from langchain_core.tracers.schemas import Run
from langsmith import Client

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.observability.costs import estimate_calls, estimate_cost
from personal_assistant_agent.observability.tracer import CostTracer


def test_cached_tokens_are_not_charged_twice():
    result = estimate_cost(
        {"input_tokens": 1000, "output_tokens": 500, "input_token_details": {"cache_read": 200}},
        Settings(),
    )
    assert result["input_cost_cny"] == pytest.approx(0.000672)
    assert result["output_cost_cny"] == pytest.approx(0.001)
    assert result["total_cost_cny"] == pytest.approx(0.001672)
    assert result["cost_is_estimate"] is True
    assert result["total_cost"] * float(result["cny_per_usd"]) == pytest.approx(
        result["total_cost_cny"]
    )


def test_tier_is_selected_by_input_context_size():
    value = estimate_cost({"input_tokens": 129000, "output_tokens": 1000}, Settings())
    assert value["input_cost_cny"] == pytest.approx(0.3096)
    assert value["output_cost_cny"] == pytest.approx(0.02)


def test_planning_calls_are_priced_individually_before_summing():
    result = estimate_calls([{"input_tokens": 70000, "output_tokens": 100}] * 2, Settings())
    assert result["total_cost_cny"] == pytest.approx(0.1124)
    assert result["model_calls"] == 2
    assert estimate_calls([], Settings()) is None
    assert estimate_calls([{"input_tokens": 10}], Settings(model_name="unknown")) is None


def test_unknown_model_or_region_does_not_invent_zero_cost():
    assert estimate_cost({"input_tokens": 1}, Settings(model_name="another-model")) is None
    assert (
        estimate_cost(
            {"input_tokens": 1},
            Settings(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        )
        is None
    )
    assert estimate_cost({"input_tokens": 1000001}, Settings()) is None


def test_invalid_cache_counts_rejected():
    with pytest.raises(ValueError):
        estimate_cost({"input_tokens": 1, "input_token_details": {"cache_read": 2}}, Settings())


def test_tracer_preserves_settings_when_framework_copies_it():
    client = Client(api_key="unit-test-only")
    try:
        settings = Settings(model_name="custom-model")
        tracer = CostTracer(settings=settings, client=client)
        copied = tracer.copy_with_metadata_defaults(metadata={"evaluation": "yes"}, tags=["test"])
        assert copied.settings is settings
    finally:
        client.close()


def test_costs_are_patched_on_llm_span_not_duplicated_on_parent():
    client = Client(api_key="unit-test-only")
    try:
        tracer = CostTracer(settings=Settings(), client=client)
        run = Run(
            name="test",
            run_type="llm",
            inputs={},
            client=client,
            extra={
                "metadata": {
                    "usage_metadata": {
                        "input_tokens": 1000,
                        "output_tokens": 10,
                        "total_tokens": 1010,
                    }
                }
            },
        )
        with patch.object(Run, "patch"):
            tracer._update_run_single(run)
        usage = run.extra["metadata"]["usage_metadata"]
        assert usage["input_tokens"] == 1000
        assert usage["total_cost"] > 0
        assert run.extra["metadata"]["cost_estimate"]["cost_is_estimate"]
        parent = Run(name="parent", run_type="chain", inputs={}, client=client)
        with patch.object(Run, "patch"):
            tracer._update_run_single(parent)
        assert "cost_estimate" not in parent.extra.get("metadata", {})
    finally:
        client.close()
