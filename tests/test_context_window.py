"""Exercise simulated context limits through the real graph and compaction hooks."""

import asyncio
import json
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.model import build_model
from personal_assistant_agent.core.config import Settings, get_settings
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService, public_error


def settings(**kwargs):
    return Settings(api_key="synthetic", context_window_tokens=8192, **kwargs)


def response(message):
    return ChatResult(generations=[ChatGeneration(message=message)])


def test_small_window_keeps_core_tools_and_caps_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    async def generate(self, messages, **kwargs):
        names = {tool["function"]["name"] for tool in kwargs["tools"]}
        calls.append(names)
        assert {"write_file", "read_file", "task", "calculate", "baidu_search"} <= names
        assert "get_city_weather" not in names
        assert len(names) == 14
        assert count_tokens_approximately(messages, tools=kwargs["tools"]) < 6016
        assert self.max_tokens == 2048
        assert self.profile["max_input_tokens"] == 8192
        if any(isinstance(m, ToolMessage) for m in messages):
            return response(AIMessage(content="完成"))
        return response(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "id": "calc",
                        "args": {"expression": "2+2"},
                    }
                ],
            )
        )

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        result = asyncio.run(
            build_agent(settings()).ainvoke(
                {
                    "messages": [HumanMessage(content="计算 2+2")],
                }
            )
        )
    assert len(calls) == 2
    assert result["messages"][-1].content == "完成"
    tool_result = next(m.content for m in result["messages"] if isinstance(m, ToolMessage))
    assert '"result": 4' in tool_result


def test_long_history_is_summarized_archived_and_fits_before_transport(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = []
    for index in range(12):
        history.extend(
            [
                HumanMessage(content=f"历史{index}: " + "synthetic context " * 100),
                AIMessage(content="synthetic reply " * 40),
            ]
        )
    history.append(HumanMessage(content="继续简短回答"))
    summary_calls, model_calls = [], []

    async def generate(self, messages, **kwargs):
        if not kwargs.get("tools"):
            summary_calls.append(messages)
            assert self.max_tokens == 384
            return response(AIMessage(content="SYNTHETIC_SUMMARY：此前在核对合成资料。"))
        model_calls.append(messages)
        assert count_tokens_approximately(messages, tools=kwargs["tools"]) <= 6016
        assert any("SYNTHETIC_SUMMARY" in str(m.content) for m in messages)
        assert "继续简短回答" in str(messages[-1].content)
        return response(AIMessage(content="已继续"))

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        result = asyncio.run(build_agent(settings()).ainvoke({"messages": history}))
    assert len(summary_calls) == len(model_calls) == 1
    assert len(result["messages"]) == len(history) + 1
    archives = list((tmp_path / "workspace/conversation_history").glob("*.md"))
    assert len(archives) == 1
    assert "历史0" in archives[0].read_text()


def test_subagent_inherits_small_window_and_tool_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    child_calls = []

    async def generate(self, messages, **kwargs):
        names = {t["function"]["name"] for t in kwargs["tools"]}
        assert "get_city_weather" not in names
        assert self.max_tokens == 2048
        assert count_tokens_approximately(messages, tools=kwargs["tools"]) <= 6016
        if "task" not in names:
            child_calls.append(messages)
            return response(AIMessage(content="子任务完成"))
        if any(isinstance(m, ToolMessage) for m in messages):
            return response(AIMessage(content="完成"))
        return response(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": "child",
                        "args": {
                            "subagent_type": "general-purpose",
                            "description": "回复子任务完成",
                        },
                    }
                ],
            )
        )

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        result = asyncio.run(
            build_agent(settings()).ainvoke(
                {
                    "messages": [HumanMessage(content="请委派子代理回复你好")],
                }
            )
        )
    assert len(child_calls) == 1
    assert result["messages"][-1].content == "完成"


@pytest.mark.parametrize("asynchronous", [True, False])
@pytest.mark.parametrize("system_padding", [0, 4000])
def test_archive_read_after_compaction_fits_remaining_budget(
    tmp_path, monkeypatch, asynchronous, system_padding
):
    from personal_assistant_agent.services.prompts import LOCAL_PROMPT, PromptVersion

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "workspace"
    directory.mkdir()
    original = "\n".join(f"{i}: " + "synthetic archive background " * 30 for i in range(60))
    (directory / "archive.txt").write_text(original)
    main_calls, summary_calls = [], []

    def generate(self, messages, **kwargs):
        if not kwargs.get("tools"):
            summary_calls.append(messages)
            return response(AIMessage(content="旧任务是整理合成项目资料。"))
        main_calls.append(messages)
        assert count_tokens_approximately(messages, tools=kwargs["tools"]) <= 6016
        assert any(
            isinstance(m, HumanMessage) and m.content == "读取 /archive.txt 后回答项目代号"
            for m in messages
        )
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_results:
            return response(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "id": "read-archive",
                            "args": {"file_path": "/archive.txt"},
                        }
                    ],
                    response_metadata={"model_provider": "openai"},
                    usage_metadata={
                        "input_tokens": 6515,
                        "output_tokens": 28,
                        "total_tokens": 6543,
                    },
                )
            )
        assert len(tool_results[0].content) < len(original)
        assert "Preview truncated" in tool_results[0].content
        assert "/archive.txt" in tool_results[0].content
        call_ids = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
        assert all(m.tool_call_id in call_ids for m in tool_results)
        return response(AIMessage(content="Bluepine"))

    async def agenerate(self, messages, **kwargs):
        return generate(self, messages, **kwargs)

    with (
        patch("langchain_openai.ChatOpenAI._generate", generate),
        patch("langchain_openai.ChatOpenAI._agenerate", agenerate),
    ):
        prompt = PromptVersion(LOCAL_PROMPT.read_text() + "x" * system_padding, "synthetic")
        graph = build_agent(settings(), prompt=prompt)
        value = {"messages": [HumanMessage(content="读取 /archive.txt 后回答项目代号")]}
        result = asyncio.run(graph.ainvoke(value)) if asynchronous else graph.invoke(value)
    assert len(main_calls) == 2
    assert len(summary_calls) == 1
    assert result["messages"][-1].content == "Bluepine"
    assert (directory / "archive.txt").read_text() == original


def test_old_usage_does_not_force_another_summary_for_a_small_packet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls, summaries = [], []

    async def generate(self, messages, **kwargs):
        if not kwargs.get("tools"):
            summaries.append(messages)
            return response(AIMessage(content="不应调用摘要"))
        calls.append(messages)
        if not any(isinstance(m, ToolMessage) for m in messages):
            return response(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate",
                            "id": "calc-small",
                            "args": {"expression": "2+2"},
                        }
                    ],
                    response_metadata={"model_provider": "openai"},
                    usage_metadata={
                        "input_tokens": 7000,
                        "output_tokens": 10,
                        "total_tokens": 7010,
                    },
                )
            )
        return response(AIMessage(content="4"))

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        asyncio.run(
            build_agent(settings()).ainvoke({"messages": [HumanMessage(content="计算 2+2")]})
        )
    assert len(calls) == 2
    assert not summaries


def test_real_streaming_graph_compacts_archive_read_without_leaking_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "workspace"
    directory.mkdir()
    (directory / "archive.txt").write_text(("synthetic archive background " * 30 + "\n") * 80)

    async def stream(self, messages, **kwargs):
        if not kwargs.get("tools"):
            chunk = AIMessageChunk(content="INTERNAL_SUMMARY: old project background")
        elif not any(isinstance(m, ToolMessage) for m in messages):
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "read_file",
                        "id": "archive-read",
                        "index": 0,
                        "args": json.dumps({"file_path": "/archive.txt"}),
                    }
                ],
            )
        else:
            assert count_tokens_approximately(messages, tools=kwargs["tools"]) <= 6016
            assert any(
                isinstance(m, HumanMessage) and m.content == "读取 /archive.txt" for m in messages
            )
            chunk = AIMessageChunk(content="Bluepine")
        yield ChatGenerationChunk(message=chunk)

    async def run():
        value = settings()
        service = ChatService(build_agent(value), value)
        payload = ChatRequest(
            conversation_id=uuid4(),
            messages=[
                {
                    "role": "user",
                    "content": "读取 /archive.txt",
                }
            ],
        )
        return [e async for e in service.stream(payload, str(uuid4()))]

    with patch("langchain_openai.ChatOpenAI._astream", stream):
        events = asyncio.run(run())
    assert [e["text"] for e in events if e["type"] == "delta"] == ["Bluepine"]
    assert events[-1]["type"] == "done"
    assert events[-1]["trace_summary"]["model_calls"] == 3


def test_oversized_system_prompt_fails_without_sending_to_model(tmp_path, monkeypatch):
    from personal_assistant_agent.services.prompts import PromptVersion

    monkeypatch.chdir(tmp_path)
    with patch("langchain_openai.ChatOpenAI._agenerate") as transport:
        with pytest.raises(ContextOverflowError):
            asyncio.run(
                build_agent(
                    settings(),
                    prompt=PromptVersion(text="system " * 10000, reference="synthetic"),
                ).ainvoke({"messages": [HumanMessage(content="你好")]})
            )
    transport.assert_not_called()
    assert public_error(ContextOverflowError("private"))[0] == "MODEL_CONTEXT_LIMIT"


def test_simulation_config_can_be_disabled_or_read_from_environment(monkeypatch):
    monkeypatch.setenv("ASSISTANT_ENV_FILE", "/nonexistent/synthetic.env")
    monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", "8192")
    monkeypatch.setenv("MODEL_CONTEXT_OUTPUT_RESERVE_TOKENS", "2048")
    monkeypatch.setenv("MODEL_CONTEXT_TOOL_NAMES", "calculate, get_current_time")
    value = get_settings()
    assert value.context_window_tokens == 8192
    assert value.context_output_reserve_tokens == 2048
    assert value.context_tool_names == ("calculate", "get_current_time")
    model = build_model(Settings(api_key="synthetic"), max_tokens=8192)
    assert model.max_tokens == 8192


def test_identity_ignores_old_tasks_without_discarding_stored_history(tmp_path, monkeypatch):
    from personal_assistant_agent.services.prompts import LOCAL_PROMPT, PromptVersion

    monkeypatch.chdir(tmp_path)
    history = [
        HumanMessage(content="读取旧项目所有文件并整理。" + "历史资料 " * 6000),
        AIMessage(content="尚待完成。"),
        HumanMessage(content="你是谁？"),
    ]
    calls = []

    async def generate(self, messages, **kwargs):
        calls.append(messages)
        assert not kwargs.get("tools")
        assert [m.content for m in messages if isinstance(m, HumanMessage)] == ["你是谁？"]
        assert not any("历史资料" in str(m.content) for m in messages)
        assert not any("途牛" in str(m.content) for m in messages)
        return response(AIMessage(content="我是个人智能助理。"))

    with patch("langchain_openai.ChatOpenAI._agenerate", generate):
        result = asyncio.run(
            build_agent(
                settings(prompt_ref="synthetic"),
                prompt=PromptVersion(LOCAL_PROMPT.read_text().strip(), "synthetic"),
            ).ainvoke(
                {
                    "messages": history,
                    "todos": [{"content": "整理旧文件", "status": "in_progress"}],
                }
            )
        )
    assert len(calls) == 1
    assert len(result["messages"]) == len(history) + 1
    assert result["todos"][0]["status"] == "in_progress"
    assert not list((tmp_path / "workspace/conversation_history").glob("*.md"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_window_tokens": -1},
        {"context_window_tokens": 2048},
        {"context_window_tokens": 8192, "context_output_reserve_tokens": 3000},
    ],
)
def test_invalid_window_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        build_model(Settings(api_key="synthetic", **overrides))
