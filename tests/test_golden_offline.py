"""The canonical schema and workspace cases run twice, without any network client."""

import asyncio

import httpx
import pytest

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.evaluation import scenarios
from personal_assistant_agent.evaluation.dataset import DEFAULT_CASES
from personal_assistant_agent.evaluation.evaluators import assess
from personal_assistant_agent.evaluation.golden import load_cases

CASES = [case for case in load_cases(DEFAULT_CASES)[0] if case.engine != "live"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("repeat", [1, 2])
def test_offline_scenario(case, repeat, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline checks must not contact a model or cloud service")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scenarios, "ROOT", tmp_path)
    target = scenarios.ScenarioTarget(Settings(), None, tmp_path / str(repeat))
    observed = asyncio.run(target({"case": case.model_dump()}))
    row = assess(case, observed)
    assert row["passed"], row
