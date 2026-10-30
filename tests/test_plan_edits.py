from personal_assistant_agent.services.plan_edits import explicit_edit


def test_append_keeps_original_states():
    old = [{"content": "原任务", "status": "completed"}]
    updated, answer = explicit_edit(
        "请只在原待办末尾追加一项“等待确认录制权限”，保持pending。", old
    )
    assert updated == old + [{"content": "等待确认录制权限", "status": "pending"}]
    assert answer
    assert len(old) == 1


def test_clear_and_negative_instruction():
    assert explicit_edit(
        "结束分享会规划并清空当前任务计划。只回答数字。", [dict(content="旧", status="pending")]
    ) == ([], None)
    assert explicit_edit("不要清空当前任务计划", []) is None
    assert explicit_edit("解释清空当前任务计划的含义", []) is None
    assert explicit_edit("请重新设计议程", []) is None
