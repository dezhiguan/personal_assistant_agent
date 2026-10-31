import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.services.delivery import initial_steps, length_limit, visible_length
from personal_assistant_agent.services.scheduling import validate_schedule, verified_serial_answer

PROMPT = (
    "请为6名新同事设计30分钟入职介绍：先明确三个学习目标，再分配议程和每段时长，"
    "接着设计三个提问，最后起草邀请文案。全部材料现在交付，全文450字以内。"
)


def invoke(replies, text, todos=None):
    replies = iter(replies)

    async def generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(replies))])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        return asyncio.run(
            build_agent(Settings(api_key="synthetic")).ainvoke(
                {"messages": [{"role": "user", "content": text}], "todos": todos or []}
            )
        )


def commit(todos, call_id="commit"):
    return AIMessage(
        content="", tool_calls=[{"name": "write_todos", "id": call_id, "args": {"todos": todos}}]
    )


def test_initial_plan_precedes_work_and_does_not_claim_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(initial_steps(PROMPT)) == 4
    steps = [{"content": s, "status": "pending"} for s in initial_steps(PROMPT)]
    result = invoke([commit(steps), AIMessage(content="等待确认材料")], PROMPT)
    assert result["todos"] == steps
    assert all(t["status"] == "pending" for t in result["todos"])
    assert initial_steps("请解释为什么先学习再练习最后复习") == []
    assert initial_steps("你好") == []


def test_completed_draft_cannot_be_replaced_by_future_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = [
        {"content": "起草会前筹备清单", "status": "completed"},
        {"content": "等待会议链接", "status": "pending"},
    ]
    future = {"content": "执行会前筹备清单", "status": "pending"}
    result = invoke(
        [
            commit([future], "bad"),
            commit([*original, future], "good"),
            AIMessage(content="保留草稿；实际执行待后续确认。"),
        ],
        "只修改议程时长，保留其他步骤",
        original,
    )
    assert result["todos"] == [*original, future]
    tool_results = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
    assert len(tool_results) == 2
    assert "起草会前筹备清单" in tool_results[0].content
    assert "completed" in tool_results[0].content


def test_length_rewrites_complete_response_without_truncation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = invoke(
        [AIMessage(content="内容" * 20), AIMessage(content="**简洁完整。**")], "请用10字以内回答"
    )
    assert result["messages"][-1].content == "**简洁完整。**"
    assert result["delivery_checks"] == 1
    assert visible_length("**甲**， `x+1`\n") == 5
    assert length_limit("全文300字以内；最多400字符") == 300


TASKS = [("A", 4), ("B", 6), ("C", 3), ("D", 1)]


@pytest.mark.parametrize(
    "days,kwargs",
    [
        ([[("A", 4), ("B", 6), ("C", 3), ("D", 1)]], {}),
        ([[("A", 4), ("C", 3)], [("B", 6), ("D", 1)]], {}),
        ([[("A", 4), ("B", 4)], [("B", 2), ("C", 3), ("D", 1)]], {}),
        ([[("A", 4)], [("B", 6)], [("C", 3), ("D", 1)]], {"resources": 2}),
        ([[("A", 4)], [("B", 5)], [("C", 3), ("D", 1)]], {}),
    ],
)
def test_all_invalid_candidates_rejected(days, kwargs):
    assert validate_schedule(TASKS, days, 8, **kwargs)


def test_valid_splits_and_indivisible_alternative():
    assert not validate_schedule(
        TASKS, [[("A", 4), ("B", 4)], [("B", 2), ("C", 3), ("D", 1)]], 8, allow_split=True
    )
    text = "单人每天最多8小时，严格依赖：A需求确认4小时→B开发6小时→C测试3小时→D发布1小时。"
    answer = verified_serial_answer(text)
    assert "缺口6小时" in answer and "需用户确认" in answer and "需2天" in answer
    assert "需3天" in verified_serial_answer(text + "不允许跨天暂停")
    assert verified_serial_answer("普通议程设计") is None


def test_engineer_wording_and_json_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = (
        "只有一名工程师，每天8小时；A需求确认4小时，B开发6小时且必须在A完成后开始，"
        "C测试3小时且必须在B完成后开始，D发布1小时且必须在C完成后开始。"
    )
    answer = verified_serial_answer(text)
    assert "缺口6小时" in answer
    assert "第1天：A4小时、B4小时" in answer
    assert "第2天：B2小时、C3小时、D1小时" in answer
    assert "并行做" not in answer
    assert verified_serial_answer(text + "最终只输出JSON") is None
    result = invoke(
        [AIMessage(content='```json\n{"ok":true}\n```'), AIMessage(content='{"ok":true}')],
        "最终只输出JSON：ok为true",
    )
    assert result["messages"][-1].content == '{"ok":true}'
    assert result["delivery_checks"] == 1


def test_failed_repair_does_not_invent_plan_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Plan commit validation failed"):
        invoke(
            [AIMessage(content="全部待办已完成") for _ in range(3)],
            "继续完成起草方案",
            [{"content": "起草方案", "status": "pending"}],
        )


def test_length_repair_exhaustion_fails_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Delivery length validation failed"):
        invoke([AIMessage(content="长文" * 20) for _ in range(4)], "请用10字以内回答")


def test_requested_duration_array_rejects_objects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = invoke(
        [
            AIMessage(content='{"durations":[{"name":"讲解","minutes":30}]}'),
            AIMessage(content='{"durations":[30]}'),
        ],
        "最终只输出JSON：durations是各段分钟数。",
    )
    assert result["messages"][-1].content == '{"durations":[30]}'
    assert result["delivery_checks"] == 1
