import pytest

from personal_assistant_agent.agents.tools import validate_time_budget


@pytest.mark.parametrize(
    "durations,available,expected,over,matches",
    [
        ([5, 12, 10, 5, 5, 3], 45, 45, 0, False),
        ([2, 8, 5, 8, 7], 30, 30, 0, True),
        ([1] * 5, 10, 10, 0, False),
        ([1] * 10, 10, 10, 0, True),
        ([4, 6, 3, 1], 8, None, 6, None),
        ([4, 6, 3], 8, None, 5, None),
    ],
)
def test_schedule_constraints(durations, available, expected, over, matches):
    result = validate_time_budget.invoke(
        dict(durations=durations, available=available, expected_total=expected)
    )
    assert result["over_budget"] == over
    assert result["matches_expected"] is matches


@pytest.mark.parametrize("durations", [[], [-1], [float("nan")], [float("inf")]])
def test_invalid_durations(durations):
    assert "error" in validate_time_budget.invoke(dict(durations=durations, available=8))
