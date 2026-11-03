"""Generic writing directives, settled-write context and recovery contracts."""

import asyncio
import json
from uuid import uuid4

import httpcore
import httpx
import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from requests.exceptions import ConnectTimeout as RequestsConnectTimeout

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.context import ContextWindowMiddleware
from personal_assistant_agent.agents.model import QwenChatOpenAI, build_model
from personal_assistant_agent.agents.tool_policy import selected_tools
from personal_assistant_agent.agents.workspace import WorkspaceBackend
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.artifacts import index_artifacts
from personal_assistant_agent.interaction.routing import MessageRouter, writing_conditions
from personal_assistant_agent.interaction.runtime import InteractionRuntime
from personal_assistant_agent.interaction.store import TaskStore
from personal_assistant_agent.services.delivery import writing_delivery_issue


def test_rejected_essays_are_archived_with_only_latest_draft_in_next_model_context(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    rejected = "校园的春天。" + "春。" * 395
    second_draft = "春天的校园。" + "春。" * 350
    accepted = "校园的春天。" + "春。" * 295
    seen = []

    async def generate(self, messages, **kwargs):
        seen.append(messages)
        if len(seen) == 2:
            assert sum(m.content == rejected for m in messages) == 1
        if len(seen) == 3:
            assert not any(m.content == rejected for m in messages)
            assert sum(m.content == second_draft for m in messages) == 1
            assert sum(m.id == "writing-delivery-validation" for m in messages) == 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=[rejected, second_draft, accepted][len(seen) - 1])
                )
            ]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)
    state = asyncio.run(
        build_agent(Settings(api_key="synthetic", tracing_enabled=False)).ainvoke(
            {
                "messages": [HumanMessage(content="写一篇600字中文作文，主题是校园的春天。")],
                "todos": [],
            }
        )
    )
    assert len(seen) == 3
    assert rejected in state["delivery_drafts"].values()
    assert second_draft in state["delivery_drafts"].values()
    assert state["messages"][-1].content == accepted


def test_writing_short_controls_keep_all_dimensions_and_status_is_authoritative(tmp_path):
    store = TaskStore(str(tmp_path / "private"))
    cid = str(uuid4())
    receipt = store.receive(
        cid,
        str(uuid4()),
        "写一篇1000字中文作文",
        {
            "intent": "NEW_TASK",
            "goal": "写作文",
            "constraints": {},
        },
        {},
    )
    tid = receipt["task_id"]
    store.state(tid, "RUNNING")
    router = MessageRouter(Settings(api_key="synthetic"))
    for text in ["海洋的主题", "改成英文的", "发生在学校里", "改成800字", "改成中文"]:
        decision = router.quick_route(text, store.tasks(cid), [])
        assert decision["intent"] == "STEER"
        assert decision["target_task_id"] == tid
        store.receive(cid, str(uuid4()), text, decision, {})
    task = store.task(tid)
    assert task["constraints"] == {"字数": "800字", "语言": "中文", "主题": "海洋", "场景": "学校"}
    assert task["revision"] == 5 and task["applied_revision"] == -1
    assert len(store.tasks(cid)) == 1
    assert len(store.messages(cid)) == 6
    assert router.quick_route("任务什么进度", [task], [])["intent"] == "STATUS"
    assert router.quick_route("任务什么进度了", [task], [])["intent"] == "STATUS"
    assert router.quick_route("任务进展怎么样了", [task], [])["intent"] == "STATUS"
    assert router.quick_route("你是谁", [task], [])["intent"] == "CHAT"
    second = {**task, "id": str(uuid4())}
    assert router.quick_route("改成800字", [task, second], [])["intent"] == "CLARIFY"
    store.db.close()


def test_clarification_number_reuses_original_changes_only_for_immediate_answer(tmp_path):
    store = TaskStore(str(tmp_path / "private"))
    cid = str(uuid4())
    router = MessageRouter.__new__(MessageRouter)
    for theme in ["春天", "秋天"]:
        store.receive(
            cid,
            str(uuid4()),
            "写作文",
            {
                "intent": "NEW_TASK",
                "goal": "写作文",
                "constraints": {"主题": theme, "字数": "500字", "语言": "中文"},
            },
            {},
        )
    tasks = store.tasks(cid)
    decision = router.quick_route("改成600字", tasks, [])
    assert decision["intent"] == "CLARIFY"
    store.receive(cid, str(uuid4()), "改成600字", decision, {})
    history = store.messages(cid)
    selected = asyncio.run(router.route("1", tasks, history))
    assert selected["intent"] == "STEER"
    assert selected["target_task_id"] == tasks[0]["id"]
    assert selected["changes"] == {"字数": "600字"}
    assert router.clarification_selection("2", tasks, history)[1] == tasks[1]["id"]
    assert router.clarification_selection("3", tasks, history) is None
    assert (
        router.clarification_selection(
            "1", tasks, [*history, {"role": "user", "kind": "CHAT", "content": "你好"}]
        )
        is None
    )
    store.receive(cid, str(uuid4()), "1", selected, {})
    assert store.tasks(cid)[0]["constraints"] == {"主题": "春天", "字数": "600字", "语言": "中文"}
    assert store.tasks(cid)[1]["constraints"]["字数"] == "500字"
    store.db.close()


def test_revision_source_short_theme_is_not_lost_when_model_omits_it(tmp_path):
    store = TaskStore(str(tmp_path / "private"))
    cid = str(uuid4())
    source = store.receive(
        cid,
        str(uuid4()),
        "写一篇1000字中文作文",
        {
            "intent": "NEW_TASK",
            "goal": "写作文",
            "constraints": {},
        },
        {},
    )["task_id"]
    store.finish(source, 0, "请提供主题。")
    receipt = store.receive(
        cid,
        str(uuid4()),
        "海洋的主题",
        {
            "intent": "STEER",
            "target_task_id": source,
            "changes": {},
        },
        {},
    )
    task = store.task(receipt["task_id"])
    assert task["source_task_id"] == source
    assert task["constraints"] == {"字数": "1000字", "语言": "中文", "主题": "海洋"}
    assert store.task(source)["result"] == "请提供主题。"
    store.db.close()


def test_durable_conversation_directory_keeps_all_user_turns_private(tmp_path):
    store = TaskStore(str(tmp_path / "private"))
    for cid in ["older", "newer"]:
        store.receive(cid, str(uuid4()), cid + "的第一条消息", {"intent": "CHAT"}, {})
        store.receive(cid, str(uuid4()), "不应作为目录标题的后续正文", {"intent": "STATUS"}, {})
    assert store.conversations() == [
        {"id": "newer", "title": "newer的第一条消息"},
        {"id": "older", "title": "older的第一条消息"},
    ]
    assert len(store.messages("older")) == 2
    store.db.close()


def test_explicit_new_composition_does_not_inherit_previous_file_workflow(tmp_path):
    router = MessageRouter(Settings(api_key="synthetic"))
    old = {
        "id": str(uuid4()),
        "goal": "写作文并保存、核验、交付",
        "status": "RUNNING",
        "constraints": {"主题": "海洋"},
    }
    content = "新建独立任务乙：写一篇1000字中文作文，主题是山林，发生在校园里。"
    decision = router.quick_route(content, [old], [])
    assert decision["intent"] == "NEW_TASK" and decision["goal"] == "写作文"
    assert decision["constraints"] == {
        "字数": "1000字",
        "语言": "中文",
        "主题": "山林",
        "场景": "校园",
    }
    store = TaskStore(str(tmp_path / "private"))
    decision["goal"] = old["goal"]
    receipt = store.receive("new-conversation", str(uuid4()), content, decision, {})
    assert store.task(receipt["task_id"])["goal"] == "写作文"
    store.db.close()


def test_compound_and_negated_conditions_leave_conflict_resolution_to_semantic_route():
    assert "字数" not in writing_conditions("不要1000字，改成800字")
    assert "语言" not in writing_conditions("不要英文")
    assert writing_conditions("写一篇关于山林的1000字中文作文") == {
        "主题": "山林",
        "字数": "1000字",
        "语言": "中文",
    }


def test_new_followup_reference_is_real_but_ordinary_based_on_is_not_a_source(monkeypatch):
    tid = str(uuid4())
    task = {
        "id": tid,
        "status": "RECOVERY_REQUIRED",
        "goal": "写作文",
        "constraints": {"字数": "800字"},
    }

    async def generate(self, messages, **kwargs):
        packet = json.loads(messages[-1].content)
        conditions = {"关联任务": tid} if "关联后续" in packet["message"] else {"主题": "物理"}
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=json.dumps(
                            {"intent": "NEW_TASK", "goal": "写作文", "constraints": conditions}
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)

    async def check():
        router = MessageRouter(Settings(api_key="synthetic"))
        decision = await router.route(
            f"新建关联后续任务，基于任务 {tid[:8]} 的已有草稿交付", [task], []
        )
        assert decision["target_task_id"] == tid
        normal = await router.route("基于物理理论写一篇作文", [task], [])
        assert normal["intent"] == "NEW_TASK" and normal["target_task_id"] is None

    asyncio.run(check())


def test_recovery_accepts_conditions_without_resuming_or_discarding_draft(tmp_path):
    store = TaskStore(str(tmp_path / "private"))
    tid = store.receive(
        "c", str(uuid4()), "写800字中文作文", {"intent": "NEW_TASK", "goal": "写作文"}, {}
    )["task_id"]
    store.state(tid, "RECOVERY_REQUIRED", recoverable=True)
    store.control(tid, "c", str(uuid4()), "STEER", "改成600字", {"字数": "600字"})
    assert store.task(tid)["status"] == "RECOVERY_REQUIRED"
    assert store.task(tid)["constraints"]["字数"] == "600字"
    assert store.task(tid)["applied_revision"] == -1
    store.control(tid, "c", str(uuid4()), "RESUME")
    assert store.task(tid)["status"] == "QUEUED"
    store.db.close()


def test_steering_during_model_admission_skips_old_delivery_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def generate(self, messages, **kwargs):
        latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        assert "800字" in latest and "中文" in latest and "山林" in latest
        yield ChatGenerationChunk(message=AIMessageChunk(content="山林的春天。" + "春。" * 390))

    monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)

    async def check():
        runtime = await InteractionRuntime(
            Settings(
                api_key="synthetic",
                tracing_enabled=False,
                interaction_dir=str(tmp_path / "private"),
            )
        ).start()
        await runtime.model_slots.acquire()
        await runtime.model_slots.acquire()
        cid = str(uuid4())
        try:
            tid = (
                await runtime.submit(
                    cid, {"client_message_id": str(uuid4()), "content": "帮我写一篇1000字中文作文"}
                )
            )["task_id"]
            await asyncio.sleep(0.1)
            for text in ["山林的主题", "改成800字", "改成英文的", "改成中文"]:
                await runtime.submit(cid, {"client_message_id": str(uuid4()), "content": text})
            runtime.model_slots.release()
            runtime.model_slots.release()
            async with asyncio.timeout(10):
                while runtime.store.task(tid)["status"] not in {
                    "SUCCEEDED",
                    "FAILED",
                    "RECOVERY_REQUIRED",
                }:
                    await asyncio.sleep(0.01)
            task = runtime.store.task(tid)
            assert task["status"] == "SUCCEEDED" and task["attempt"] == 1
            assert task["revision"] == task["applied_revision"] == 4
            assert task["model_calls"] == 1
        finally:
            runtime.model_slots.release()
            runtime.model_slots.release()
            await runtime.close()

    asyncio.run(check())


def test_failed_source_without_result_still_exposes_imported_draft_to_new_thread(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    seen = []

    async def generate(self, messages, **kwargs):
        seen.extend(str(m.content) for m in messages if isinstance(m, AIMessage))
        yield ChatGenerationChunk(message=AIMessageChunk(content="合成副本已接收。"))

    monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)

    async def check():
        runtime = await InteractionRuntime(
            Settings(
                api_key="synthetic",
                tracing_enabled=False,
                interaction_dir=str(tmp_path / "private"),
            )
        ).start()
        try:
            source = runtime.store.receive(
                "c", str(uuid4()), "保存合成草稿", {"intent": "NEW_TASK", "goal": "整理草稿"}, {}
            )["task_id"]
            runtime.store.state(source, "RECOVERY_REQUIRED", recoverable=False)
            settings = runtime.task_settings(source)
            root = tmp_path / settings.workspace_dir
            root.mkdir(parents=True)
            (root / "draft.txt").write_text("合成草稿。")
            task = runtime.store.task(source)
            task["artifacts"] = index_artifacts(settings)
            runtime.store.save(task)
            child = runtime.store.receive(
                "c",
                str(uuid4()),
                "创建关联后续任务",
                {"intent": "NEW_TASK", "goal": "简短总结", "source_task_id": source},
                {},
            )["task_id"]
            runtime.wakeup.set()
            async with asyncio.timeout(10):
                while runtime.store.task(child)["status"] != "SUCCEEDED":
                    await asyncio.sleep(0.01)
            assert any(f"/sources/{source}/draft.txt" in text for text in seen)
            assert (
                tmp_path
                / runtime.task_settings(child).workspace_dir
                / "sources"
                / source
                / "draft.txt"
            ).read_text() == "合成草稿。"
            assert runtime.store.task(source)["result"] is None
        finally:
            await runtime.close()

    asyncio.run(check())


def test_settled_large_write_uses_archived_preview_without_mutating_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = "合成草稿段落。" * 4000
    call = {
        "name": "write_file",
        "id": "settled-write",
        "args": {"file_path": "/draft.md", "content": body},
    }
    inputs = [
        HumanMessage(content="整理已保存草稿并简短总结"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(
            content="Updated file /draft.md", name="write_file", tool_call_id="settled-write"
        ),
    ]

    async def generate(self, messages, **kwargs):
        for message in messages:
            for observed in getattr(message, "tool_calls", []):
                if observed["id"] == "settled-write":
                    assert "完整调用已归档" in observed["args"]["content"]
                    assert len(observed["args"]["content"]) < 350
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="草稿已保存。"))])

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)
    result = asyncio.run(
        build_agent(Settings(api_key="synthetic", context_window_tokens=8192)).ainvoke(
            {"messages": inputs}
        )
    )
    assert result["messages"][-1].content == "草稿已保存。"
    original = next(m for m in result["messages"] if getattr(m, "tool_calls", []))
    assert original.tool_calls[0]["args"]["content"] == body
    archives = list((tmp_path / "workspace/large_tool_results").glob("settled-write-*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["args"]["content"] == body


def test_provider_overflow_recovery_clips_read_before_generic_eviction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(api_key="synthetic", context_window_tokens=8192)
    model = build_model(settings)
    middleware = ContextWindowMiddleware(model, WorkspaceBackend("workspace"), settings)
    user = HumanMessage(content="核对已保存的合成草稿", id="user")
    call = {"name": "read_file", "id": "read", "args": {"file_path": "/draft.txt"}}
    messages = [
        user,
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="合成正文" * 300, tool_call_id="read", name="read_file"),
    ]
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="合成规则"),
        tools=[],
        state={"messages": messages},
    )

    # The real provider can count substantially more Chinese tokens than the
    # framework estimate, which is exactly the path that previously re-evicted
    # short receipts instead of clipping the large read result.
    async def count(self, packet, **kwargs):
        return 5600 + sum(len(str(m.content)) for m in packet if isinstance(m, ToolMessage))

    monkeypatch.setattr(QwenChatOpenAI, "acount_input_tokens", count)
    sent = []

    async def handler(candidate):
        assert await count(model, [candidate.system_message, *candidate.messages]) <= 6016
        sent.append(candidate)
        return "accepted"

    result, replacements = asyncio.run(
        middleware._acall_with_budget(
            request, handler, error=ContextOverflowError("preflight overflow"), rejected=request
        )
    )
    assert result == "accepted" and len(sent) == 1
    assert replacements and "Full file: /draft.txt" in replacements[0].content
    assert messages[-1].content == "合成正文" * 300


def test_compacted_rejection_count_includes_reinjected_current_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(api_key="synthetic", context_window_tokens=8192)
    model = build_model(settings)
    middleware = ContextWindowMiddleware(model, WorkspaceBackend("workspace"), settings)
    current = HumanMessage(content="最新要求" * 125, id="current")
    call = {"name": "read_file", "id": "read", "args": {"file_path": "/draft.txt"}}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="草稿" * 400, tool_call_id="read", name="read_file"),
    ]
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="合成规则"),
        tools=[],
        state={"messages": [current, *messages]},
    )

    async def count(self, packet, **kwargs):
        return 5000 + sum(
            len(str(m.content)) for m in packet if isinstance(m, ToolMessage) or m.id == "current"
        )

    monkeypatch.setattr(QwenChatOpenAI, "acount_input_tokens", count)

    async def handler(candidate):
        packet, _ = middleware._count_args(candidate)
        assert await count(model, packet) <= 6016
        assert any(m.id == "current" for m in candidate.messages)
        return "accepted"

    result, _ = asyncio.run(
        middleware._acall_with_budget(
            request, handler, error=ContextOverflowError("preflight overflow"), rejected=request
        )
    )
    assert result == "accepted"


@pytest.mark.parametrize(
    "answer,valid", [("海。" * 400, True), ("海。" * 200, False), ("word " * 800 + ".", False)]
)
def test_writing_delivery_checks_target_length_and_language(answer, valid):
    issue = writing_delivery_issue("写作文\n当前要求：字数=800字；语言=中文；主题=海洋", answer)
    assert (issue == "") is valid


def test_complete_essay_with_word_count_note_keeps_its_valid_ending():
    request = "写作文\n当前要求：字数=800字；语言=中文"
    assert writing_delivery_issue(request, "海。" * 390 + "\n（全文约800字）") == ""
    assert (
        writing_delivery_issue(request, "海。" * 390 + "没有写完的最后一句") == "正文缺少完整结尾"
    )


def test_saved_essay_counts_body_instead_of_delivery_receipt_without_relaxing_caps():
    body = "# 校园的春天\n\n" + "春。" * 265
    receipt = (
        "修订完成。正文经 wc -m 实测约540字，已保存到 /essays/acceptance.md。\n以下为完整正文：\n\n"
    )
    answer = receipt + body + "\n\n[下载作文文件](/api/files/acceptance.md)"
    assert writing_delivery_issue("写一篇500字中文作文", answer) == ""
    assert "需要0–500" in writing_delivery_issue("写一篇最多500字中文作文", answer)
    assert "需要450–550" in writing_delivery_issue("写一篇500字中文作文", receipt + body * 2)


def test_native_unnamed_todo_receipt_does_not_expand_writing_tools():
    tools = [{"name": name} for name in ["write_todos", "write_file", "read_file", "execute"]]
    call = {"name": "write_todos", "id": "plan", "args": {"todos": []}}
    messages = [
        HumanMessage(content="写一篇800字中文作文"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="Updated todo list", tool_call_id="plan"),
    ]
    assert messages[-1].name is None
    assert selected_tools(messages, tools) == tools[:1]


@pytest.mark.parametrize(
    "timeout_type", [httpx.ConnectTimeout, httpcore.ConnectTimeout, RequestsConnectTimeout]
)
def test_raw_http_timeout_resumes_same_checkpoint_without_replaying_tools(
    tmp_path, monkeypatch, timeout_type
):
    monkeypatch.chdir(tmp_path)
    failed = False

    async def generate(self, messages, **kwargs):
        nonlocal failed
        if not any(isinstance(m, ToolMessage) for m in messages):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_calls=[{"name": "calculate", "id": "calc", "args": {"expression": "2+2"}}],
                )
            )
        elif not failed:
            failed = True
            raise timeout_type("synthetic transport timeout")
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(content="合成任务已完成。"))

    monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)

    async def check():
        runtime = await InteractionRuntime(
            Settings(
                api_key="synthetic",
                tracing_enabled=False,
                interaction_dir=str(tmp_path / "private"),
            )
        ).start()
        cid = str(uuid4())

        async def wait_status(tid, expected):
            async with asyncio.timeout(10):
                while runtime.store.task(tid)["status"] != expected:
                    await asyncio.sleep(0.01)

        try:
            ack = await runtime.submit(
                cid, {"client_message_id": str(uuid4()), "content": "帮我搜索全球新闻"}
            )
            tid = ack["task_id"]
            await wait_status(tid, "RECOVERY_REQUIRED")
            task = runtime.store.task(tid)
            assert task["recoverable"] and task["error"]["code"] == "MODEL_TIMEOUT"
            assert task["tool_calls"] == 1
            ack = await runtime.submit(cid, {"client_message_id": str(uuid4()), "content": "继续"})
            assert ack["task_id"] == tid
            await wait_status(tid, "SUCCEEDED")
            assert runtime.store.task(tid)["tool_calls"] == 1
            assert runtime.store.task(tid)["attempt"] == 2
        finally:
            await runtime.close()

    asyncio.run(check())
