"""Exercise the real Deep Agents graph with a substituted model transport."""

import asyncio
import json
from unittest.mock import patch
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService
from personal_assistant_agent.services.prompts import PromptVersion


def test_actual_graph_exposes_planning_execution_and_custom_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    async def generate(self, messages, **kwargs):
        calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="你好"))])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        agent = build_agent(Settings(api_key="test-only"))
        result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "你好"}]}))
    assert result["messages"][-1].content == "你好"
    assert calls
    expected = {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "delete",
        "execute",
        "task",
        "baidu_search",
        "get_current_time",
        "calculate",
        "validate_time_budget",
    }
    assert all({t["function"]["name"] for t in c["tools"]} == expected for c in calls)
    assert list((tmp_path / "workspace").iterdir()) == []


def test_actual_graph_writes_edits_reads_and_executes_in_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    steps = [
        ("write_file", {"file_path": "/result.txt", "content": "before\n"}),
        ("edit_file", {"file_path": "/result.txt", "old_string": "before", "new_string": "after"}),
        ("read_file", {"file_path": "/result.txt"}),
        ("execute", {"command": "cat /result.txt"}),
        ("calculate", {"expression": "200 * 10%"}),
        ("get_current_time", {"timezone": "Asia/Shanghai"}),
    ]
    calls = []

    async def generate(self, messages, **kwargs):
        index = len(calls)
        calls.append(messages)
        message = AIMessage(content="已写入并核对结果。")
        if index < len(steps):
            name, args = steps[index]
            message = AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": f"call-{index}"}]
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        graph = build_agent(Settings(api_key="test-only"))
        result = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "执行任务"}]}))
    assert (tmp_path / "workspace/result.txt").read_text() == "after\n"
    messages = {m.name: m for m in result["messages"] if isinstance(m, ToolMessage)}
    assert all(m.status == "success" for m in messages.values())
    assert "after" in messages["read_file"].content
    assert "after" in messages["execute"].content
    assert json.loads(messages["calculate"].content)["result"] == 20
    assert json.loads(messages["get_current_time"].content)["timezone"] == "Asia/Shanghai"


def test_task_subagent_inherits_workspace_and_custom_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    child_calls = []

    async def generate(self, messages, **kwargs):
        is_child = "子任务写入文件" in str(messages[0:2])
        has_result = any(isinstance(m, ToolMessage) for m in messages)
        if is_child:
            child_calls.append(kwargs)
            message = (
                AIMessage(content="子任务已完成")
                if has_result
                else AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "id": "child-write",
                            "args": {"file_path": "/child.txt", "content": "child"},
                        }
                    ],
                )
            )
        else:
            message = (
                AIMessage(content="委派结果已核对")
                if has_result
                else AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "id": "delegate",
                            "args": {
                                "subagent_type": "general-purpose",
                                "description": "子任务写入文件",
                            },
                        }
                    ],
                )
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        result = asyncio.run(
            build_agent(Settings(api_key="test-only")).ainvoke(
                {"messages": [{"role": "user", "content": "委派任务"}]}
            )
        )
    assert result["messages"][-1].content == "委派结果已核对"
    assert (tmp_path / "workspace/child.txt").read_text() == "child"
    assert child_calls
    assert {"baidu_search", "get_current_time", "calculate", "execute"} <= {
        t["function"]["name"] for t in child_calls[0]["tools"]
    }


def test_actual_graph_baidu_search_uses_http_tool_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = []

    async def send(self, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            text=(
                '<div class="result c-container"><h3><a href="https://example.com">'
                '百度结果</a></h3><div class="c-abstract">摘要内容</div></div>'
            ),
        )

    async def generate(self, messages, **kwargs):
        if any(isinstance(m, ToolMessage) for m in messages):
            message = AIMessage(content="检索结果已返回")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "baidu_search", "id": "search", "args": {"query": "测试关键词"}}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    with (
        patch("langchain_openai.ChatOpenAI._agenerate", generate),
        patch("httpx.AsyncClient.send", send),
    ):
        result = asyncio.run(
            build_agent(Settings(api_key="test-only")).ainvoke(
                {"messages": [{"role": "user", "content": "查百度"}]}
            )
        )
    message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert json.loads(message.content)["results"][0]["summary"] == "摘要内容"
    assert [r.url.host for r in requests] == ["www.baidu.com"]


def test_streamed_task_reports_progress_without_subagent_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def generate(self, messages, **kwargs):
        child = "内部子任务计算" in str(messages[:2])
        has_result = any(isinstance(m, ToolMessage) for m in messages)
        calls = []
        if not has_result:
            calls = (
                [{"name": "calculate", "id": "child-math", "args": {"expression": "1+1"}}]
                if child
                else [
                    {
                        "name": "task",
                        "id": "delegate",
                        "args": {
                            "subagent_type": "general-purpose",
                            "description": "内部子任务计算",
                        },
                    }
                ]
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=("子代理内部结果" if child else "最终交付") if has_result else "",
                tool_calls=calls,
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            )
        )

    async def check():
        settings = Settings(api_key="test-only")
        payload = ChatRequest(
            conversation_id=uuid4(), messages=[{"role": "user", "content": "委派计算"}]
        )
        return [
            e
            async for e in ChatService(build_agent(settings), settings).stream(
                payload, str(uuid4())
            )
        ]

    with patch("langchain_openai.ChatOpenAI._astream", generate):
        output = asyncio.run(check())
    assert [(e["name"], e["status"]) for e in output if e["type"] == "tool"] == [
        ("task", "running"),
        ("task", "finished"),
    ]
    assert "".join(e["text"] for e in output if e["type"] == "delta") == "最终交付"
    assert output[-1]["type"] == "done"
    usage = next(e for e in output if e["type"] == "usage")
    assert usage["total_tokens"] == 44  # Two main calls plus two child calls.
    assert usage["cost_estimate"]["model_calls"] == 4


def test_actual_graph_creates_revises_and_completes_plan_through_service():
    plans = [
        [{"content": "梳理目标", "status": "in_progress"}],
        [
            {"content": "梳理目标", "status": "completed"},
            {"content": "根据限制完善方案", "status": "in_progress"},
        ],
        [
            {"content": "梳理目标", "status": "completed"},
            {"content": "根据限制完善方案", "status": "completed"},
        ],
    ]
    calls = []

    async def generate(self, messages, **kwargs):
        index = len(calls)
        calls.append(messages)
        message = (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": plans[index]},
                        "id": f"plan-{index}",
                    }
                ],
            )
            if index < len(plans)
            else AIMessage(content="这是整理后的方案。")
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def check():
        settings = Settings(api_key="test-only")
        service = ChatService(build_agent(settings), settings)
        payload = ChatRequest(
            conversation_id=uuid4(), messages=[{"role": "user", "content": "请制定方案"}]
        )
        return [e async for e in service.stream(payload, str(uuid4()))]

    async def stream_generate(self, messages, **kwargs):
        result = await generate(self, messages, **kwargs)
        message = result.generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_calls=message.tool_calls,
            )
        )

    with patch("langchain_openai.ChatOpenAI._astream", stream_generate):
        output = asyncio.run(check())
    assert [e["todos"] for e in output if e["type"] == "plan"] == plans
    assert "".join(e["text"] for e in output if e["type"] == "delta") == "这是整理后的方案。"
    assert output[-1]["type"] == "done"
    assert len(calls) == 4
    assert "根据限制完善方案" in str(calls[2][0].content)


def test_restored_todos_are_visible_to_model_without_shared_state():
    prompts = []

    async def generate(self, messages, **kwargs):
        prompts.append(str(messages[0].content))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="收到"))])

    async def check():
        agent = build_agent(Settings(api_key="test-only"))
        await agent.ainvoke(
            {
                "messages": [{"role": "user", "content": "继续"}],
                "todos": [{"content": "调整产品发布计划", "status": "in_progress"}],
            }
        )
        await agent.ainvoke({"messages": [{"role": "user", "content": "你好"}], "todos": []})

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        asyncio.run(check())
    assert "调整产品发布计划" in prompts[0]
    # A restored plan with no commit receives two bounded correction attempts.
    assert len(prompts) == 4
    assert "调整产品发布计划" not in prompts[-1]


def test_hosted_prompt_is_followed_by_current_execution_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = []

    async def generate(self, messages, **kwargs):
        prompts.append(str(messages[0].content))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="收到"))])

    with (
        patch(
            "personal_assistant_agent.agents.assistant.load_prompt",
            return_value=PromptVersion("旧提示词：只支持文字对话", "hosted"),
        ),
        patch("langchain_openai.ChatOpenAI._agenerate", generate),
    ):
        agent = build_agent(Settings(api_key="test-only", prompt_ref="hosted"))
        asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "你好"}]}))
    assert "旧提示词中冲突的能力描述已失效" in prompts[0]
    assert "baidu_search" in prompts[0]
    assert "get_current_time" in prompts[0]
    assert "workspace/" in prompts[0]
