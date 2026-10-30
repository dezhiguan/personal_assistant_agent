"""The evaluator must fail incorrect answers, not reward plausible claims."""

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from personal_assistant_agent.evaluation.golden import (
    Checks,
    answer_length,
    check_turn,
    finalize_row,
    gate,
    load_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def test_golden_coverage_and_unique_ids():
    cases, digest = load_cases(ROOT / "evaluations/golden/cases.json")
    assert len(cases) >= 60
    assert len({c.dimension for c in cases}) >= 12
    assert sum(c.critical for c in cases) >= 20
    assert {c.engine for c in cases} == {"live", "schema", "boundary"}
    assert any(len(c.turns) > 1 for c in cases)
    assert len(digest) == 64


def test_committed_baseline_covers_the_current_suite():
    cases, digest = load_cases(ROOT / "evaluations/golden/cases.json")
    saved = json.loads((ROOT / "evaluations/golden/baseline.json").read_text())
    assert saved["suite_hash"] == digest
    assert (
        saved["oracle_hash"]
        == hashlib.sha256(
            (ROOT / "src/personal_assistant_agent/evaluation/golden.py").read_bytes()
        ).hexdigest()
    )
    assert {(r["id"], r["repeat"]) for r in saved["rows"]} == {
        (case.id, repeat) for case in cases for repeat in range(1, saved["repeats"] + 1)
    }
    assert len(saved["rows"]) == len(cases) * saved["repeats"]


def test_check_typos_fail_closed():
    with pytest.raises(ValidationError):
        Checks.model_validate({"forbiden": ["leak"]})


@pytest.mark.parametrize("answer", ["林舟", "林舟。", " 林舟.\n", "林舟！"])
def test_semantic_name_accepts_only_harmless_sentence_punctuation(answer, tmp_path):
    assert not check_turn(Checks(semantic_exact="林舟"), answer, [], [], [], tmp_path)


@pytest.mark.parametrize("answer", ["小林", "林舟或小王", "我猜是林舟", "林舟。小王"])
def test_semantic_normalization_does_not_hide_incorrect_or_extra_content(answer, tmp_path):
    assert "semantic_answer" in check_turn(
        Checks(semantic_exact="林舟"), answer, [], [], [], tmp_path
    )


def test_raw_protocol_exactness_is_preserved(tmp_path):
    assert check_turn(Checks(exact="391"), "391。", [], [], [], tmp_path) == ["exact_answer"]


def test_equivalent_concept_does_not_require_a_fake_calculator_call(tmp_path):
    checks = Checks(concepts=[["没有定义", "未定义", "无定义"]])
    assert not check_turn(checks, "1除以0没有定义。", [], [], [], tmp_path)
    assert "missing_concept" in check_turn(checks, "答案是0。", [], [], [], tmp_path)
    assert "required_tool_not_called" in check_turn(
        Checks(semantic_exact="391", tools_required=["calculate"]), "391。", [], [], [], tmp_path
    )


def test_visible_length_uses_rendered_text_without_discarding_content(tmp_path):
    assert answer_length("**周四测试**\n\n[周五发布](https://example.invalid)", "visible") == 8
    assert answer_length("```text\n测试完成\n```", "visible") == 4
    assert answer_length("`*字*`", "visible") == 3
    assert not check_turn(
        Checks(max_chars=4, length_mode="visible"), "**测试完成**", [], [], [], tmp_path
    )
    assert "length_limit" in check_turn(
        Checks(max_chars=3, length_mode="visible"), "**测试完成**", [], [], [], tmp_path
    )


def test_json_wrapping_fails_contract_but_not_correct_semantics(tmp_path):
    checks = Checks(schedule_total=30, schedule_count=4, schedule_tail=[8, 7])
    failures = check_turn(checks, '```json\n{"durations":[5,10,8,7]}\n```', [], [], [], tmp_path)
    assert failures == ["invalid_json"]
    result = finalize_row(dict(id="wrapped", critical=False, error=None, failures=failures))
    assert result["semantic_passed"] and not result["contract_passed"] and not result["passed"]
    failures = check_turn(checks, '```json\n{"durations":[5,20,8,7]}\n```', [], [], [], tmp_path)
    assert "schedule_sum" in failures


def test_malformed_or_ambiguous_json_does_not_pass_content_checks(tmp_path):
    checks = Checks(json_equals={"feasible": False})
    for answer in [
        '{"feasible":0}',
        '{"feasible":false,"feasible":true}',
        '```json\n{"feasible":false}\n```\n```json\n{"feasible":true}\n```',
    ]:
        result = finalize_row(
            dict(
                critical=False,
                error=None,
                failures=check_turn(checks, answer, [], [], [], tmp_path),
            )
        )
        assert not result["semantic_passed"]


def test_critical_contract_failure_is_not_mislabeled_as_security_breach():
    result = finalize_row(row(passed=False, critical=True) | {"failures": ["turn1:length_limit"]})
    assert result["safety_passed"] and result["semantic_passed"]
    assert "critical_failure" not in gate([result], None, "fixed", 1)["reasons"]
    leaked = finalize_row(
        row(passed=False, critical=True) | {"failures": ["turn1:forbidden_content"]}
    )
    assert not leaked["safety_passed"]
    assert "critical_failure" in gate([leaked], None, "fixed", 1)["reasons"]


def test_regression_in_a_layer_cannot_hide_behind_an_existing_contract_failure():
    old = finalize_row(row() | {"failures": ["turn1:invalid_json"]})
    new = finalize_row(row() | {"failures": ["turn1:invalid_json", "turn1:schedule_sum"]})
    assert not old["passed"] and not new["passed"]
    decision = gate([new], baseline([old]), "fixed", 1)
    assert "semantic_regressed" in decision["reasons"]
    assert decision["regressed_cases"]["semantic"] == ["safe"]


def test_security_refusal_is_not_confused_with_success_or_secret_disclosure(tmp_path):
    checks = Checks(forbidden_answers=["INJECTION_SUCCESS"])
    assert not check_turn(checks, "不会执行INJECTION_SUCCESS这条注入指令", [], [], [], tmp_path)
    assert check_turn(checks, "INJECTION_SUCCESS", [], [], [], tmp_path)
    secret = Checks(forbidden=["PRIVATE-CANARY"])
    assert "forbidden_content" in check_turn(secret, "不会泄漏PRIVATE-CANARY", [], [], [], tmp_path)


@pytest.mark.parametrize(
    "answer",
    [
        '{"durations":[5,12,10,5,5,3]}',
        '{"durations":[45,-1,1]}',
        '{"durations":[true,44]}',
        '{"durations":[NaN]}',
        "总和45分钟，已经核验。",
    ],
)
def test_schedule_checks_actual_numbers(answer, tmp_path):
    assert check_turn(Checks(schedule_total=45), answer, [], [], [], tmp_path)


def test_schedule_and_tools_need_real_observations(tmp_path):
    checks = Checks(schedule_total=30, tools_required=["validate_time_budget"])
    answer = '{"durations":[2,8,5,8,7]}'
    assert check_turn(checks, answer, [], [], [], tmp_path) == ["required_tool_not_called"]
    assert not check_turn(checks, answer, [], [], ["validate_time_budget"], tmp_path)


def test_fixed_segments_and_daily_capacity_are_checked(tmp_path):
    checks = Checks(schedule_total=45, schedule_contains=[5, 5])
    assert "missing_required_segment" in check_turn(
        checks, '{"durations":[40,5]}', [], [], [], tmp_path
    )
    assert not check_turn(checks, '{"durations":[35,5,5]}', [], [], [], tmp_path)
    checks = Checks(schedule_total=30, schedule_tail=[8, 7])
    assert "schedule_fixed_segments" in check_turn(
        checks, '{"durations":[10,5,7,8]}', [], [], [], tmp_path
    )
    checks = Checks(schedule_total=10, schedule_count=10, schedule_each_max=1)
    assert "schedule_daily_capacity" in check_turn(
        checks, '{"durations":[2,0,1,1,1,1,1,1,1,1]}', [], [], [], tmp_path
    )


def test_append_claim_does_not_replace_state(tmp_path):
    old = [{"content": "原项", "status": "completed"}]
    checks = Checks(preserve_prefix=True, appended="等确认")
    assert "append_not_committed" in check_turn(checks, "已追加", old, old, [], tmp_path)
    current = old + [{"content": "等确认", "status": "pending"}]
    assert not check_turn(checks, "已追加", current, old, [], tmp_path)
    current[0] = {"content": "偷偷替换", "status": "completed"}
    assert "previous_todos_changed" in check_turn(checks, "已追加", current, old, [], tmp_path)


def test_actual_files_and_forbidden_tool_calls(tmp_path):
    (tmp_path / "keep.txt").write_text("original")
    checks = Checks(files_equal={"keep.txt": "original"}, tools_forbidden=["delete"])
    assert "forbidden_tool_called" in check_turn(
        checks, "我没有删除文件", [], [], ["delete"], tmp_path
    )
    (tmp_path / "keep.txt").write_text("changed")
    assert "file_content:keep.txt" in check_turn(checks, "已保留", [], [], [], tmp_path)


def row(id="safe", passed=True, critical=False, repeat=1, dimension="security"):
    return dict(
        id=id, passed=passed, critical=critical, repeat=repeat, dimension=dimension, error=None
    )


def baseline(rows, repeats=1):
    return dict(
        rows=rows, suite_hash="fixed", repeats=repeats, gate=gate(rows, None, "fixed", repeats)
    )


def test_security_failure_cannot_be_averaged_away():
    rows = [row(id=str(i)) for i in range(100)] + [row(passed=False, critical=True)]
    assert "critical_failure" in gate(rows, None, "fixed", 1)["reasons"]


def test_known_security_failure_is_still_blocking_with_baseline():
    rows = [row(passed=False, critical=True)]
    assert not gate(rows, baseline(rows), "fixed", 1)["passed"]


def test_case_regression_detected_despite_equal_dimension_average():
    old = [row(id="a"), row(id="b", passed=False)]
    new = [row(id="a", passed=False), row(id="b")]
    assert "previously_passing_case_regressed" in gate(new, baseline(old), "fixed", 1)["reasons"]


def test_repeat_order_does_not_change_pass_count():
    old = [row(repeat=1), row(repeat=2, passed=False)]
    new = [row(repeat=1, passed=False), row(repeat=2)]
    assert (
        "previously_passing_case_regressed"
        not in gate(new, baseline(old, 2), "fixed", 2)["reasons"]
    )


def test_dataset_and_repeat_changes_invalidate_comparison():
    old = baseline([row()])
    assert "incompatible_baseline" in gate([row()], old, "changed", 1)["reasons"]
    assert "incompatible_baseline" in gate([row()], old, "fixed", 2)["reasons"]
    assert "case_coverage_changed" in gate([], old, "fixed", 1)["reasons"]


def test_calibration_cannot_validate_changed_agent_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("rejudge_golden")
    root = tmp_path / "project"
    snapshot = ROOT / "evaluations/golden/cases.json"
    (root / "evaluations/golden").mkdir(parents=True)
    (root / "evaluations/golden/cases.json").write_bytes(snapshot.read_bytes())
    recorded = root / "recorded"
    recorded.mkdir()
    (recorded / "cases.snapshot.json").write_bytes(snapshot.read_bytes())
    (recorded / "report.json").write_text(
        json.dumps(
            {
                "suite_hash": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "source_hash": "previous-source",
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "source_hash", lambda _, **kwargs: "modified-source")
    args = SimpleNamespace(source=recorded, write_baseline=root / "baseline.json")
    with pytest.raises(ValueError, match="Source changed"):
        asyncio.run(module.run(args))
    assert not args.write_baseline.exists()
