"""Regression checks must reject plausible-looking incorrect answers."""

from personal_assistant_agent.evaluation.evaluators import task_success


def test_rules_reject_wrong_context_and_false_capability_claims():
    assert task_success({"answer": "林舟"}, {"contains_all": ["林舟"]})["score"] is True
    assert task_success({"answer": "小林"}, {"contains_all": ["林舟"]})["score"] is False
    assert task_success({"answer": "已设置提醒"}, {"forbidden": ["已设置"]})["score"] is False


def test_json_and_planning_are_evaluated_structurally():
    reference = {"json_equals": {"count": 2}}
    assert task_success({"answer": '{"count": 2}'}, reference)["score"] is True
    assert task_success({"answer": '```json\n{"count": 2}\n```'}, reference)["score"] is False
    assert task_success({"answer": "已规划", "todos": []}, {"min_todos": 3})["score"] is False


def test_length_limit_is_enforced():
    assert task_success({"answer": "a" * 41}, {"max_length": 40})["score"] is False
