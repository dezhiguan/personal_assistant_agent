"""Current-turn scope and read deduplication regressions."""

import asyncio

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.tool_policy import (
    cached_read,
    repeated_search_result,
    selected_tools,
    standalone_question,
)
from personal_assistant_agent.core.config import Settings


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


@pytest.mark.parametrize(
    "name,args",
    [
        ("write_file", {"file_path": "/new.txt", "content": "unauthorized"}),
        ("edit_file", {"file_path": "/important.txt", "old_string": "原稿", "new_string": "修改"}),
        ("delete", {"file_path": "/important.txt"}),
        ("execute", {"command": "touch /new.txt"}),
        ("task", {"subagent_type": "general-purpose", "description": "删除important.txt"}),
    ],
)
def test_default_window_blocks_unconfirmed_effects_even_if_model_invents_call(
    tmp_path, monkeypatch, name, args
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    original = tmp_path / "workspace/important.txt"
    original.write_text("原稿")
    actual = []

    class Tools(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):
            actual.append(serialized["name"])

    async def generate(self, messages, **kwargs):
        offered = {t["function"]["name"] for t in kwargs.get("tools", [])}
        assert not offered.intersection({"write_file", "edit_file", "delete", "execute", "task"})
        reply = (
            AIMessage(content="仅提供整理建议，等待确认。")
            if any(isinstance(m, ToolMessage) for m in messages)
            else AIMessage(content="", tool_calls=[{"name": name, "id": "effect", "args": args}])
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)
    graph = build_agent(Settings(api_key="synthetic", context_window_tokens=0))
    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="先告诉我整理建议，未经我确认不要修改或删除。")],
            },
            config={"callbacks": [Tools()]},
        )
    )
    assert original.read_text() == "原稿"
    assert not (tmp_path / "workspace/new.txt").exists()
    assert not actual
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in result["messages"])


def test_new_explicit_confirmation_restores_mutation_permission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    original = tmp_path / "workspace/important.txt"
    original.write_text("原稿")

    async def generate(self, messages, **kwargs):
        assert "edit_file" in {t["function"]["name"] for t in kwargs.get("tools", [])}
        calls = (
            []
            if any(isinstance(m, ToolMessage) for m in messages)
            else [
                {
                    "name": "edit_file",
                    "id": "allowed",
                    "args": {
                        "file_path": "/important.txt",
                        "old_string": "原稿",
                        "new_string": "确认版",
                    },
                }
            ]
        )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="完成" if not calls else "", tool_calls=calls)
                )
            ]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)
    graph = build_agent(Settings(api_key="synthetic"))
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="未经我确认不要修改或删除。"),
                    AIMessage(content="等待确认。"),
                    HumanMessage(content="现在确认，将important.txt中的原稿改成确认版。"),
                ]
            }
        )
    )
    assert original.read_text() == "确认版"


def test_repeated_search_detects_no_new_evidence_but_respects_turns_and_changes():
    import json

    def response(query, results, error=None):
        return ToolMessage(
            name="baidu_search",
            tool_call_id=query,
            content=json.dumps(
                {
                    "query": query,
                    "results": results,
                    "error": error,
                }
            ),
        )

    a = response("first", [{"summary": "已有事实"}])
    b = response("other query", [{"summary": "已有事实"}])
    messages = [HumanMessage(content="查资料"), a, b]
    assert repeated_search_result(messages)
    assert not repeated_search_result([*messages, HumanMessage(content="重新查资料")])
    assert not repeated_search_result([a, response("new", [{"summary": "新增事实"}])])
    assert not repeated_search_result([a, response("failed", [], "error")])


@pytest.mark.parametrize("mutable_directive", [False, True])
def test_real_graph_no_progress_hint_keeps_tools_and_skips_live_directives(
    tmp_path,
    monkeypatch,
    mutable_directive,
):
    from langchain.agents.middleware import AgentMiddleware

    from personal_assistant_agent.agents.tools import build_baidu_search
    from personal_assistant_agent.interaction.middleware import ControlState

    monkeypatch.chdir(tmp_path)

    class DirectiveContext(AgentMiddleware):
        state_schema = ControlState

        def branch(self):
            return DirectiveContext()

        def before_model(self, state, runtime):
            return {"directive_revision": 0}

    async def search(query, max_results):
        return {"source": "baidu", "results": [{"title": "资料", "summary": "当前事实"}]}

    calls = 0

    async def generate(self, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "baidu_search",
                                    "id": str(calls),
                                    "args": {"query": str(calls)},
                                }
                            ],
                        )
                    )
                ]
            )
        instructions = "\n".join(str(m.content) for m in messages if m.type == "system")
        assert ("检索进展：连续查询" in instructions) is (not mutable_directive)
        assert "baidu_search" in {t["function"]["name"] for t in kwargs.get("tools", [])}
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="已根据资料概括。"))]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)
    settings = Settings(api_key="synthetic")
    graph = build_agent(
        settings,
        tools=[build_baidu_search(settings, search=search)],
        control=DirectiveContext() if mutable_directive else None,
    )
    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="检索并说明资料。")]}))
    assert calls == 3
    assert result["messages"][-1].content == "已根据资料概括。"
