"""Current-turn scope and read deduplication regressions."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from personal_assistant_agent.agents.tool_policy import (
    cached_read,
    selected_tools,
    standalone_question,
)


def tool(name):
    return {"function": {"name": name}}


def read_pair(name="read_file", call_id="r1"):
    return [
        AIMessage(
            content="", tool_calls=[{"name": name, "id": call_id, "args": {"file_path": "/a.txt"}}]
        ),
        ToolMessage(content="synthetic", name=name, tool_call_id=call_id),
    ]


def test_standalone_scope_does_not_swallow_dependent_questions():
    for text in ["你是谁？", "你好", "请问你叫什么名字", "Who are you?"]:
        assert standalone_question(text)
        assert selected_tools([HumanMessage(content=text)], [tool("read_file")]) == []
    for text in [
        "你是谁，读一下我的配置",
        "文件里的你是谁是什么意思",
        "回忆刚才的预算",
        "介绍一下你自己在刚才任务中的角色",
    ]:
        assert not standalone_question(text)


def test_advice_does_not_enable_execution_or_mutation():
    names = ["read_file", "ls", "execute", "delete", "write_file", "edit_file", "task"]
    tools = [tool(n) for n in names]
    user = HumanMessage(content="先告诉我清理建议，未经我确认不要删除或修改文件。")
    assert selected_tools([user], tools) == tools[:2]


def test_cached_read_stays_in_current_turn_and_is_invalidated_by_mutation():
    messages = [HumanMessage(content="读取文件"), *read_pair()]
    call = {"name": "read_file", "id": "r2", "args": {"file_path": "/a.txt"}}
    assert cached_read(messages, call).content == "synthetic"
    assert cached_read([*messages, HumanMessage(content="再次读取")], call) is None
    assert cached_read([*messages, *read_pair("edit_file", "edit")], call) is None
    assert selected_tools([*messages, *read_pair(call_id="r2")], [tool("read_file")]) == []
