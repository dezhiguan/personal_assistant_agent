"""Real Harness, disk saver and barrier-driven full-duplex acceptance tests."""

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.middleware import ControlMiddleware
from personal_assistant_agent.interaction.routing import MessageRouter
from personal_assistant_agent.interaction.runtime import InteractionRuntime
from personal_assistant_agent.interaction.store import Conflict


def message(content, **extra):
    return {"client_message_id": str(uuid4()), "content": content, **extra}


async def until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def settings_for(tmp_path):
    return Settings(
        api_key="test-only", interaction_dir=str(tmp_path / "private"), tracing_enabled=False
    )


def test_p1a_continuous_steering_and_independent_chat(tmp_path, monkeypatch):
    """UC01/02/03/05/06, A1/A3: controls never wait for the chat or tool barrier."""
    monkeypatch.chdir(tmp_path)

    async def check():
        started, release, chat_started, chat_release = [asyncio.Event() for _ in range(4)]
        searches, model_inputs = [], []

        @tool
        async def search_news(query: str) -> str:
            """Search synthetic news, waiting at an explicit barrier."""
            searches.append(query)
            started.set()
            await release.wait()
            return "中国科技报道；海外体育报道"

        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            model_inputs.append(latest)
            if "什么" in latest or "六岁" in latest:
                chat_started.set()
                await chat_release.wait()
                answer = "独角兽是估值较高的未上市企业。"
                if "六岁" in latest:
                    assert any("独角兽" in str(m.content) for m in messages[:-1])
                calls = []
            elif not any(isinstance(m, ToolMessage) for m in messages):
                calls = [{"name": "search_news", "id": "global-search", "args": {"query": latest}}]
                answer = ""
            else:
                calls = []
                answer = "最终交付：" + latest
            yield ChatGenerationChunk(message=AIMessageChunk(content=answer, tool_calls=calls))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        settings = settings_for(tmp_path)
        router = MessageRouter(settings)
        # Explicit tasks still use the deterministic news path in this fixture.
        runtime = await InteractionRuntime(
            settings,
            agent_factory=lambda s, **kw: build_agent(s, tools=[search_news], **kw),
            router=router,
        ).start()
        conversation = str(uuid4())
        try:
            ack = await runtime.submit(conversation, message("帮我搜索全球新闻。"))
            task_id = ack["task_id"]
            await asyncio.wait_for(started.wait(), 10)
            original_thread = task_id
            chat = await runtime.submit(conversation, message("什么叫科技独角兽？"))
            await asyncio.wait_for(chat_started.wait(), 10)
            await runtime.submit(conversation, message("就搜索国内的就好了。"))
            await runtime.submit(conversation, message("最近三天的"))
            await runtime.submit(conversation, message("只看科技"))
            current = runtime.store.task(task_id)
            assert current["constraints"] == {
                "范围": "中国大陆",
                "时间": "最近三天",
                "主题": "科技",
            }
            assert current["applied_revision"] < current["revision"]
            assert len(runtime.store.tasks()) == 1
            assert not release.is_set() and not chat_release.is_set()
            runtime.stop_chat(conversation, chat["message_id"])
            await until(lambda: conversation not in runtime.chat_running)
            # Independent chat can finish while the task's slow tool is still blocked.
            chat_release.set()
            chat2 = await runtime.submit(conversation, message("独角兽是什么意思？"))
            await until(
                lambda: any(
                    e["type"] == "chat.settled" and e["chat_message_id"] == chat2["message_id"]
                    for e in runtime.store.events(conversation)
                )
            )
            assert runtime.store.task(task_id)["status"] == "RUNNING"
            release.set()
            await until(lambda: runtime.store.task(task_id)["status"] in {"SUCCEEDED", "FAILED"})
            current = runtime.store.task(task_id)
            assert current["status"] == "SUCCEEDED", current
            assert "中国大陆" in current["result"] and "最近三天" in current["result"]
            assert "主题=科技" in current["result"] and "全球" not in current["result"]
            assert len(searches) == 1 and "全球" in searches[0]
            assert current["attempt"] == 1 and current["id"] == original_thread
            assert current["original_request"] == "帮我搜索全球新闻。"
            assert (
                len(
                    [
                        e
                        for e in runtime.store.events(conversation)
                        if e["type"] == "control.applied"
                    ]
                )
                == 3
            )
            # Terminal steering creates a revision with a reference, leaving the old result intact.
            revised = await runtime.submit(conversation, message("改成最近一周"))
            assert revised["revision_of"] == task_id
            assert revised["task_id"] != task_id
            assert runtime.store.task(task_id)["result"] == current["result"]
        finally:
            release.set()
            chat_release.set()
            await runtime.close()

    asyncio.run(check())


def test_native_pause_resume_cancel_and_bounded_parallelism(tmp_path, monkeypatch):
    """UC04/07/08/09, A2/A4/A5: real interrupts and per-thread ownership."""
    monkeypatch.chdir(tmp_path)

    async def check():
        entered = []
        release = asyncio.Event()

        @tool
        async def slow_read() -> str:
            """Read synthetic data after a barrier."""
            entered.append(1)
            await release.wait()
            return "完成"

        async def generate(self, messages, **kwargs):
            calls = (
                []
                if any(isinstance(m, ToolMessage) for m in messages)
                else [{"name": "slow_read", "id": "read", "args": {}}]
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="已完成数据读取。" if not calls else "", tool_calls=calls
                )
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        settings = settings_for(tmp_path)
        runtime = await InteractionRuntime(
            settings, agent_factory=lambda s, **kw: build_agent(s, tools=[slow_read], **kw)
        ).start()
        conversation = str(uuid4())
        try:
            tasks = []
            for _ in range(3):
                ack = runtime.store.receive(
                    conversation,
                    str(uuid4()),
                    "读取数据",
                    {"intent": "NEW_TASK", "goal": "读取数据"},
                    {},
                )
                tasks.append(ack["task_id"])
            runtime.wakeup.set()
            await until(lambda: len(entered) == 2)
            assert runtime.store.task(tasks[2])["status"] == "QUEUED"
            runtime.store.control(tasks[0], conversation, str(uuid4()), "PAUSE")
            runtime.store.control(tasks[1], conversation, str(uuid4()), "CANCEL")
            assert runtime.store.task(tasks[0])["status"] == "PAUSING"
            assert runtime.store.task(tasks[1])["status"] == "CANCELLING"
            release.set()
            await until(lambda: runtime.store.task(tasks[0])["status"] in {"PAUSED", "FAILED"})
            assert runtime.store.task(tasks[0])["status"] == "PAUSED"
            await until(lambda: runtime.store.task(tasks[1])["status"] == "CANCELLED")
            checkpoint = await runtime.saver.aget_tuple({"configurable": {"thread_id": tasks[0]}})
            assert checkpoint is not None
            runtime.store.control(
                tasks[0], conversation, str(uuid4()), "STEER", changes={"格式": "简短"}
            )
            assert runtime.store.task(tasks[0])["status"] == "PAUSED"
            runtime.store.control(tasks[0], conversation, str(uuid4()), "RESUME")
            runtime.wakeup.set()
            await until(lambda: runtime.store.task(tasks[0])["status"] in {"SUCCEEDED", "FAILED"})
            assert runtime.store.task(tasks[0])["status"] == "SUCCEEDED"
            with pytest.raises(Conflict):
                runtime.store.control(tasks[1], conversation, str(uuid4()), "RESUME")
            with pytest.raises(KeyError):
                runtime.store.task(tasks[0], str(uuid4()))
            assert (
                runtime.task_settings(tasks[0]).workspace_dir
                != runtime.task_settings(tasks[1]).workspace_dir
            )
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


def test_queued_chats_use_preceding_completed_reply(tmp_path, monkeypatch):
    """UC02: all three are accepted while the first reply is still generating."""
    monkeypatch.chdir(tmp_path)

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            if latest == "红眼航班是什么":
                entered.set()
                await release.wait()
                content = "红眼航班在深夜或凌晨起降。"
            elif latest == "带小孩适合吗":
                assert any(m.type == "ai" and "凌晨" in m.content for m in messages)
                content = "儿童可能因夜间休息受影响。"
            else:
                assert any(m.type == "ai" and "儿童" in m.content for m in messages)
                content = "六岁且容易晕车，应考虑孩子的休息与舒适度。"
            seen.append(latest)
            yield ChatGenerationChunk(message=AIMessageChunk(content=content))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        settings = settings_for(tmp_path)
        runtime = await InteractionRuntime(settings).start()
        cid = str(uuid4())
        try:
            await runtime.submit(cid, message("红眼航班是什么", intent="CHAT"))
            await asyncio.wait_for(entered.wait(), 10)
            await runtime.submit(cid, message("带小孩适合吗", intent="CHAT"))
            await runtime.submit(cid, message("孩子六岁，容易晕车", intent="CHAT"))
            assert len(runtime.pending_chat(cid)) == 3
            release.set()
            await until(lambda: len(seen) == 3 and not runtime.pending_chat(cid))
            assert seen == ["红眼航班是什么", "带小孩适合吗", "孩子六岁，容易晕车"]
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


@pytest.mark.parametrize("when", ["model", "delivery"])
def test_steering_wins_model_or_delivery_race(tmp_path, monkeypatch, when):
    """A3/A7: neither stale proposed tools nor stale final text escape the revision barrier."""
    monkeypatch.chdir(tmp_path)

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        invoked = []
        model_calls = []

        @tool
        async def obsolete_query() -> str:
            """A call planned against an obsolete global scope."""
            invoked.append(True)
            return "全球新闻"

        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            model_calls.append(latest)
            if len(model_calls) == 1 and when == "model":
                entered.set()
                await release.wait()
                calls = [{"name": "obsolete_query", "id": "obsolete", "args": {}}]
                text = ""
            else:
                calls = []
                text = "交付：" + latest
            yield ChatGenerationChunk(message=AIMessageChunk(content=text, tool_calls=calls))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(
            settings_for(tmp_path),
            agent_factory=lambda s, **kw: build_agent(s, tools=[obsolete_query], **kw),
        ).start()
        cid = str(uuid4())
        try:
            real_finish = runtime.store.finish
            if when == "delivery":
                won = False

                def competing_finish(task_id, revision, result):
                    nonlocal won
                    if not won:
                        won = True
                        runtime.store.control(
                            task_id, cid, str(uuid4()), "STEER", changes={"范围": "中国大陆"}
                        )
                    return real_finish(task_id, revision, result)

                monkeypatch.setattr(runtime.store, "finish", competing_finish)
            ack = await runtime.submit(cid, message("搜索全球新闻"))
            task_id = ack["task_id"]
            if when == "model":
                await asyncio.wait_for(entered.wait(), 10)
                await runtime.submit(cid, message("就搜索国内的就好了"))
                release.set()
            await until(lambda: runtime.store.task(task_id)["status"] in {"SUCCEEDED", "FAILED"})
            task = runtime.store.task(task_id)
            assert task["status"] == "SUCCEEDED", task
            assert "中国大陆" in task["result"] and "全球" not in task["result"]
            assert not invoked
            assert task["attempt"] == 1 and len(model_calls) == 2
            assert len([e for e in runtime.store.events(cid) if e["type"] == "task.result"]) == 1
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


def test_subagent_steering_and_pause_at_root_boundary(tmp_path, monkeypatch):
    """A3/A4: children read controls independently; only the root durably pauses."""
    monkeypatch.chdir(tmp_path)

    async def check():
        started, release = asyncio.Event(), asyncio.Event()
        child_requests = []

        @tool
        async def baidu_search(query: str) -> str:
            """Synthetic child read."""
            started.set()
            await release.wait()
            return "国内和全球的合成资料"

        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            child = "委派资料收集" in latest
            results = [m for m in messages if isinstance(m, ToolMessage)]
            if child:
                child_requests.append(latest)
                calls = (
                    []
                    if results
                    else [{"name": "baidu_search", "id": "child-query", "args": {"query": latest}}]
                )
            else:
                calls = (
                    []
                    if results
                    else [
                        {
                            "name": "task",
                            "id": "delegate",
                            "args": {
                                "subagent_type": "general-purpose",
                                "description": "委派资料收集",
                            },
                        }
                    ]
                )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=("交付：" + latest) if not calls else "", tool_calls=calls
                )
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(
            settings_for(tmp_path),
            agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search], **kw),
        ).start()
        cid = str(uuid4())
        try:
            ack = await runtime.submit(cid, message("搜索全球新闻"))
            tid = ack["task_id"]
            await asyncio.wait_for(started.wait(), 10)
            await runtime.submit(cid, message("就搜索国内的就好了"))
            runtime.store.control(tid, cid, str(uuid4()), "PAUSE")
            release.set()
            await until(
                lambda: (
                    runtime.store.task(tid)["status"] in {"PAUSED", "FAILED", "RECOVERY_REQUIRED"}
                )
            )
            assert runtime.store.task(tid)["status"] == "PAUSED", runtime.store.task(tid)
            assert "中国大陆" in child_requests[-1]
            operations = runtime.store.db.execute("SELECT status FROM operations").fetchall()
            assert operations and all(row[0] == "DONE" for row in operations)
            runtime.store.control(tid, cid, str(uuid4()), "RESUME")
            runtime.wakeup.set()
            await until(lambda: runtime.store.task(tid)["status"] in {"SUCCEEDED", "FAILED"})
            assert runtime.store.task(tid)["status"] == "SUCCEEDED"
            assert "中国大陆" in runtime.store.task(tid)["result"]
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


def test_cumulative_budget_and_single_runtime_owner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        calls = []

        async def generate(self, messages, **kwargs):
            calls.append(True)
            yield ChatGenerationChunk(message=AIMessageChunk(content="交付"))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        settings = replace(settings_for(tmp_path), max_model_calls=1)
        runtime = await InteractionRuntime(settings).start()
        second = InteractionRuntime(settings)
        try:
            with pytest.raises(RuntimeError, match="另一进程"):
                await second.start()
            cid = str(uuid4())
            tid = runtime.store.receive(
                cid, str(uuid4()), "搜索新闻", {"intent": "NEW_TASK", "goal": "搜索新闻"}, {}
            )["task_id"]
            # Simulate a legitimately exhausted cumulative budget at resume.
            with runtime.store.transaction():
                task = runtime.store.task(tid)
                task["model_calls"] = 1
                runtime.store.save(task)
            runtime.wakeup.set()
            await until(
                lambda: runtime.store.task(tid)["status"] in {"FAILED", "RECOVERY_REQUIRED"}
            )
            assert runtime.store.task(tid)["status"] == "RECOVERY_REQUIRED"
            assert not calls
            assert runtime.store.task(tid)["model_calls"] == 1
        finally:
            await runtime.close()

    asyncio.run(check())


def test_parallel_tool_batch_defers_unstarted_stale_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        executed = []

        @tool
        async def baidu_search(query: str) -> str:
            """Read a synthetic result after the barrier."""
            executed.append(query)
            entered.set()
            await release.wait()
            return "合成国内资料"

        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            calls = (
                []
                if any(isinstance(m, ToolMessage) for m in messages)
                else [
                    {"name": "baidu_search", "id": "first", "args": {"query": "全球第一批"}},
                    {"name": "baidu_search", "id": "second", "args": {"query": "全球第二批"}},
                ]
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="交付：" + latest if not calls else "", tool_calls=calls
                )
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(
            settings_for(tmp_path),
            agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search], **kw),
        ).start()
        runtime.tool_slots = asyncio.Semaphore(1)
        cid = str(uuid4())
        try:
            tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
            await asyncio.wait_for(entered.wait(), 10)
            await runtime.submit(cid, message("就搜索国内的就好了"))
            release.set()
            await until(lambda: runtime.store.task(tid)["status"] in {"SUCCEEDED", "FAILED"})
            assert runtime.store.task(tid)["status"] == "SUCCEEDED"
            assert len(executed) == 1
            assert executed[0] in {"全球第一批", "全球第二批"}
            graph = build_agent(
                runtime.task_settings(tid),
                tools=[baidu_search],
                checkpointer=runtime.saver,
                control=ControlMiddleware(runtime.store, tid, runtime.settings),
            )
            saved = await graph.aget_state({"configurable": {"thread_id": tid}})
            tools = [m for m in saved.values["messages"] if isinstance(m, ToolMessage)]
            assert {m.tool_call_id for m in tools} == {"first", "second"}
            assert len([m for m in tools if "尚未执行" in m.content]) == 1
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


def test_directives_survive_real_context_compaction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        from langchain_core.outputs import ChatGeneration, ChatResult

        entered, release = asyncio.Event(), asyncio.Event()
        summaries = []
        seen = []

        @tool
        async def baidu_search(query: str) -> str:
            """Return enough synthetic text to trigger real context compaction."""
            entered.set()
            await release.wait()
            return "synthetic archive data " * 15000

        async def summarize(self, messages, **kwargs):
            from langchain_core.messages import AIMessage

            summaries.append(True)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="SYNTHETIC_SUMMARY: historical global data collected."
                        )
                    )
                ]
            )

        async def generate(self, messages, **kwargs):
            if not kwargs.get("tools"):
                summaries.append(True)
                yield ChatGenerationChunk(message=AIMessageChunk(content="SYNTHETIC_SUMMARY"))
                return
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            seen.append(latest)
            calls = (
                []
                if len(seen) > 1
                else [{"name": "baidu_search", "id": "large-read", "args": {"query": latest}}]
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="交付：" + latest if not calls else "", tool_calls=calls
                )
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", summarize)
        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        settings = replace(settings_for(tmp_path), context_window_tokens=8192)
        runtime = await InteractionRuntime(
            settings, agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search], **kw)
        ).start()
        cid = str(uuid4())
        try:
            from langchain_core.messages import AIMessage

            tid = runtime.store.receive(
                cid,
                str(uuid4()),
                "搜索全球新闻",
                {"intent": "NEW_TASK", "goal": "搜索新闻", "constraints": {"范围": "全球"}},
                {},
            )["task_id"]
            runtime.store.state(tid, "PAUSED")
            graph = build_agent(
                runtime.task_settings(tid),
                tools=[baidu_search],
                checkpointer=runtime.saver,
                control=ControlMiddleware(runtime.store, tid, settings),
            )
            history = []
            for index in range(12):
                history.extend(
                    [
                        HumanMessage(content=f"合成历史{index}：" + "synthetic context " * 100),
                        AIMessage(content="synthetic answer " * 40),
                    ]
                )
            history.append(HumanMessage(content="搜索全球新闻"))
            await graph.aupdate_state(
                {"configurable": {"thread_id": tid}}, {"messages": history}, as_node="tools"
            )
            runtime.store.control(tid, cid, str(uuid4()), "RESUME")
            runtime.wakeup.set()
            await asyncio.wait_for(entered.wait(), 10)
            for text in ["就搜索国内的就好了", "最近三天的", "只看科技"]:
                await runtime.submit(cid, message(text))
            release.set()
            await until(
                lambda: (
                    runtime.store.task(tid)["status"]
                    in {"SUCCEEDED", "FAILED", "RECOVERY_REQUIRED"}
                )
            )
            current = runtime.store.task(tid)
            assert current["status"] == "SUCCEEDED", current
            assert "中国大陆" in current["result"] and "主题=科技" in current["result"]
            assert "最近三天" in seen[-1]
            # Compaction is verified by its archived tool payload, not a pretend summary.
            archives = list((tmp_path / "workspace-tasks" / tid).rglob("*.md"))
            assert archives and summaries
            assert current["model_calls"] >= len(seen) + len(summaries)
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


def test_checkpoint_failure_blocks_unverified_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        calls = []

        async def generate(self, messages, **kwargs):
            calls.append(True)
            yield ChatGenerationChunk(message=AIMessageChunk(content="已完成合成任务。"))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path)).start()
        original = runtime.saver.aput
        failed = False

        async def failing_put(config, checkpoint, metadata, new_versions):
            nonlocal failed
            if not failed and checkpoint["channel_values"].get("directive_revision") is not None:
                failed = True
                raise OSError("synthetic checkpoint disk failure")
            return await original(config, checkpoint, metadata, new_versions)

        monkeypatch.setattr(runtime.saver, "aput", failing_put)
        cid = str(uuid4())
        try:
            tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
            await until(lambda: runtime.store.task(tid)["status"] == "RECOVERY_REQUIRED")
            assert failed
            assert not runtime.store.task(tid)["recoverable"]
            assert runtime.store.task(tid)["error"]["code"] == "CHECKPOINT_REVIEW_REQUIRED"
            monkeypatch.setattr(runtime.saver, "aput", original)
            with pytest.raises(Conflict):
                runtime.store.control(tid, cid, str(uuid4()), "RESUME")
            assert not [e for e in runtime.store.events(cid) if e["type"] == "task.result"]
        finally:
            await runtime.close()

    asyncio.run(check())


def test_task_files_and_revision_artifacts_are_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            calls = (
                []
                if any(isinstance(m, ToolMessage) for m in messages)
                else [
                    {
                        "name": "write_file",
                        "id": "same-call-id",
                        "args": {
                            "file_path": "/result.txt",
                            "content": latest,
                        },
                    }
                ]
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="已交付合成文件。" if not calls else "", tool_calls=calls
                )
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path)).start()
        cid = str(uuid4())
        try:
            tasks = [
                runtime.store.receive(
                    cid, str(uuid4()), goal, {"intent": "NEW_TASK", "goal": goal}, {}
                )["task_id"]
                for goal in ["保存A", "保存B"]
            ]
            runtime.wakeup.set()
            await until(
                lambda: all(
                    runtime.store.task(t)["status"] in {"SUCCEEDED", "FAILED"} for t in tasks
                )
            )
            assert all(runtime.store.task(t)["status"] == "SUCCEEDED" for t in tasks)
            a, b = [tmp_path / runtime.task_settings(t).workspace_dir / "result.txt" for t in tasks]
            assert a.read_text() == "保存A" and b.read_text() == "保存B"
            assert not (tmp_path / "workspace/result.txt").exists()
            revised = runtime.store.receive(
                cid,
                str(uuid4()),
                "改格式",
                {"intent": "STEER", "target_task_id": tasks[0], "changes": {"格式": "简短"}},
                {},
            )
            rid = revised["task_id"]
            runtime.wakeup.set()
            await until(lambda: runtime.store.task(rid)["status"] in {"SUCCEEDED", "FAILED"})
            assert runtime.store.task(rid)["status"] == "SUCCEEDED"
            copied = (
                tmp_path
                / runtime.task_settings(rid).workspace_dir
                / "sources"
                / tasks[0]
                / "result.txt"
            )
            assert copied.read_text() == "保存A"
            assert a.read_text() == "保存A"
            assert runtime.store.task(rid)["source_task_id"] == tasks[0]
        finally:
            await runtime.close()

    asyncio.run(check())


def test_graceful_drain_finishes_tool_batch_before_releasing_owner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        @tool
        async def baidu_search(query: str) -> str:
            """Read a synthetic result at an explicit shutdown barrier."""
            entered.set()
            await release.wait()
            return "合成资料"

        async def generate(self, messages, **kwargs):
            calls = (
                []
                if any(isinstance(m, ToolMessage) for m in messages)
                else [{"name": "baidu_search", "id": "read", "args": {"query": "合成"}}]
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="已完成。" if not calls else "", tool_calls=calls)
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        settings = settings_for(tmp_path)

        def factory(s, **kw):
            return build_agent(s, tools=[baidu_search], **kw)

        runtime = await InteractionRuntime(settings, agent_factory=factory).start()
        cid = str(uuid4())
        tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
        await asyncio.wait_for(entered.wait(), 10)
        closing = asyncio.create_task(runtime.close())
        try:
            await until(lambda: runtime.stopping)
            assert not closing.done()
        finally:
            release.set()
            await closing
        resumed = await InteractionRuntime(settings, agent_factory=factory).start()
        try:
            await until(lambda: resumed.store.task(tid)["status"] in {"SUCCEEDED", "FAILED"})
            assert resumed.store.task(tid)["status"] == "SUCCEEDED"
            assert resumed.store.task(tid)["model_calls"] == 2
        finally:
            await resumed.close()

    asyncio.run(check())


@pytest.mark.parametrize("failure", ["timeout", "rate_limit"])
def test_provider_errors_are_safe_and_resume_keeps_budget(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)

    async def check():
        import httpx
        from openai import APITimeoutError, RateLimitError

        failing = True

        async def generate(self, messages, **kwargs):
            if failing:
                request = httpx.Request("POST", "https://synthetic.invalid/model")
                if failure == "timeout":
                    raise APITimeoutError(request=request)
                raise RateLimitError(
                    "synthetic-private-provider-body",
                    response=httpx.Response(429, request=request),
                    body=None,
                )
            yield ChatGenerationChunk(message=AIMessageChunk(content="已完成。"))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path)).start()
        cid = str(uuid4())
        try:
            tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
            await until(lambda: runtime.store.task(tid)["status"] == "RECOVERY_REQUIRED")
            current = runtime.store.task(tid)
            assert current["recoverable"] and current["model_calls"] == 1
            assert "synthetic-private-provider-body" not in str(runtime.store.snapshot(cid))
            failing = False
            runtime.store.control(tid, cid, str(uuid4()), "RESUME")
            runtime.wakeup.set()
            await until(lambda: runtime.store.task(tid)["status"] in {"SUCCEEDED", "FAILED"})
            assert runtime.store.task(tid)["status"] == "SUCCEEDED"
            assert runtime.store.task(tid)["model_calls"] >= 2
        finally:
            await runtime.close()

    asyncio.run(check())


def test_cancel_bypasses_saturated_model_admission_and_queued_terminal_stays_closed(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    async def check():
        calls = []

        async def generate(self, messages, **kwargs):
            calls.append(True)
            yield ChatGenerationChunk(message=AIMessageChunk(content="不应调用"))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path)).start()
        runtime.model_slots = asyncio.Semaphore(0)
        cid = str(uuid4())
        try:
            tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
            await until(lambda: runtime.store.task(tid)["applied_revision"] == 0)
            runtime.store.control(tid, cid, str(uuid4()), "CANCEL")
            await until(lambda: runtime.store.task(tid)["status"] == "CANCELLED")
            assert not calls
            # A worker queued by the scheduler cannot resurrect a terminal task.
            original = runtime.store.task(tid)
            await runtime.execute(tid)
            assert runtime.store.task(tid) == original
        finally:
            await runtime.close()

    asyncio.run(check())


def test_cancel_waits_for_real_synchronous_write_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        import threading

        entered, release = threading.Event(), threading.Event()
        paths = []

        def factory(settings, **kwargs):
            from pathlib import Path

            path = Path(settings.workspace_dir) / "result.txt"
            paths.append(path)

            @tool
            def fixture_write() -> str:
                """Synchronous synthetic write; cannot be interrupted once issued."""
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("fixture barrier timeout")
                path.write_text("已发生的写入")
                return "已保存"

            return build_agent(settings, tools=[fixture_write], **kwargs)

        async def generate(self, messages, **kwargs):
            calls = [{"name": "fixture_write", "id": "write", "args": {}}]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=calls))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path), agent_factory=factory).start()
        cid = str(uuid4())
        try:
            tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
            await until(entered.is_set)
            runtime.store.control(tid, cid, str(uuid4()), "CANCEL")
            assert runtime.store.task(tid)["status"] == "CANCELLING"
            assert not paths[0].exists()
            release.set()
            await until(lambda: runtime.store.task(tid)["status"] == "CANCELLED")
            assert paths[0].read_text() == "已发生的写入"
            assert runtime.store.task(tid)["effects"] == [
                {"name": "fixture_write", "status": "DONE"}
            ]
        finally:
            release.set()
            await runtime.close()

    asyncio.run(check())


def test_durable_ordinary_history_is_not_silently_truncated_at_legacy_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def check():
        async def generate(self, messages, **kwargs):
            assert any(m.type == "human" and m.content == "暗号=A" for m in messages)
            assert len(messages) > 60
            yield ChatGenerationChunk(message=AIMessageChunk(content="暗号是A。"))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path)).start()
        cid = str(uuid4())
        try:
            for index in range(31):
                mid = str(uuid4())
                runtime.store.receive(
                    cid, mid, "暗号=A" if index == 0 else f"合成历史{index}", {"intent": "CHAT"}, {}
                )
                runtime.complete_chat(cid, mid, "合成回复")
            reply = await runtime.submit(cid, message("第一个暗号是什么", intent="CHAT"))
            await until(lambda: not runtime.pending_chat(cid))
            result = next(
                m for m in runtime.store.messages(cid) if m["id"] == "chat:" + reply["message_id"]
            )
            assert result["content"] == "暗号是A。"
        finally:
            await runtime.close()

    asyncio.run(check())


@pytest.mark.parametrize("inferred", ["CHAT", "STEER"])
def test_fast_steering_bypasses_slow_classifier_and_rejects_stale_merge(
    tmp_path, monkeypatch, inferred
):
    monkeypatch.chdir(tmp_path)

    async def check():
        tool_started, tool_release, route_started, route_release = [
            asyncio.Event() for _ in range(4)
        ]

        @tool
        async def baidu_search(query: str) -> str:
            """Hold a real task tool while another message is still being classified."""
            tool_started.set()
            await tool_release.wait()
            return "合成国内资料"

        async def generate(self, messages, **kwargs):
            latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            calls = (
                []
                if "搜索新闻" not in latest or any(isinstance(m, ToolMessage) for m in messages)
                else [{"name": "baidu_search", "id": "query", "args": {"query": latest}}]
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="答复：" + latest if not calls else "", tool_calls=calls
                )
            )

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(
            settings_for(tmp_path),
            agent_factory=lambda s, **kw: build_agent(s, tools=[baidu_search], **kw),
        ).start()
        cid = str(uuid4())
        pending = None
        try:
            tid = (await runtime.submit(cid, message("搜索全球新闻")))["task_id"]
            await asyncio.wait_for(tool_started.wait(), 10)

            async def classify(*args):
                route_started.set()
                await route_release.wait()
                return {"intent": inferred, "target_task_id": tid, "changes": {"范围": "国际"}}

            monkeypatch.setattr(runtime.router, "infer", classify)
            pending = asyncio.create_task(runtime.submit(cid, message("换个范围吧")))
            await asyncio.wait_for(route_started.wait(), 10)
            await asyncio.wait_for(runtime.submit(cid, message("就搜索国内的就好了")), 1)
            assert runtime.store.task(tid)["constraints"]["范围"] == "中国大陆"
            route_release.set()
            if inferred == "STEER":
                with pytest.raises(Conflict, match="路由期间"):
                    await pending
            else:
                assert (await pending)["intent"] == "CHAT"
            tool_release.set()
            await until(lambda: runtime.store.task(tid)["status"] in {"SUCCEEDED", "FAILED"})
            assert "中国大陆" in runtime.store.task(tid)["result"]
        finally:
            route_release.set()
            tool_release.set()
            if pending:
                await asyncio.gather(pending, return_exceptions=True)
            await runtime.close()

    asyncio.run(check())


def test_ordinary_reply_can_discuss_completed_results_without_task_execution_state(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    async def check():
        async def generate(self, messages, **kwargs):
            assert any(m.type == "ai" and "ORBIT" in m.content for m in messages)
            assert not any(isinstance(m, ToolMessage) for m in messages)
            yield ChatGenerationChunk(message=AIMessageChunk(content="项目代号是ORBIT。"))

        monkeypatch.setattr("langchain_openai.ChatOpenAI._astream", generate)
        runtime = await InteractionRuntime(settings_for(tmp_path)).start()
        cid = str(uuid4())
        try:
            tid = runtime.store.receive(
                cid,
                str(uuid4()),
                "生成合成报告",
                {"intent": "NEW_TASK", "goal": "生成合成报告"},
                {},
            )["task_id"]
            runtime.store.finish(tid, 0, "合成报告：项目代号ORBIT。")
            original = runtime.store.task(tid)
            await runtime.submit(cid, message("刚才报告的项目代号是什么", intent="CHAT"))
            await until(lambda: not runtime.pending_chat(cid))
            assert runtime.store.task(tid) == original
            assert len(runtime.store.tasks()) == 1
            assert runtime.store.messages(cid, "CHAT")[-1]["content"] == "项目代号是ORBIT。"
        finally:
            await runtime.close()

    asyncio.run(check())


def test_complex_news_requirements_do_not_use_lossy_fast_path(tmp_path):
    router = MessageRouter(settings_for(tmp_path))
    assert router.quick_route("帮我搜索全球新闻", [], [], None, None)["intent"] == "NEW_TASK"
    assert (
        router.quick_route(
            "搜索最近三天国内科技新闻，分四类，每类两条，附来源链接", [], [], "NEW_TASK", None
        )
        is None
    )
    tasks = [{"id": "a", "goal": "搜索新闻", "status": "RUNNING"}]
    assert router.quick_route("只看科技，至少十条并附链接", tasks, [], None, None) is None
    assert router.quick_route("继续", tasks, [], None, None)["intent"] == "RESUME"


def test_router_accepts_null_unused_fields_without_erasing_condition_deletions():
    from personal_assistant_agent.interaction.routing import Decision

    parsed = Decision.model_validate(
        {
            "intent": "CHAT",
            "goal": None,
            "constraints": None,
            "changes": None,
            "question": None,
            "candidates": None,
        }
    )
    assert parsed.goal == "" and parsed.constraints == {} and parsed.candidates == []
    steer = Decision.model_validate({"intent": "STEER", "changes": {"时间": None}})
    assert steer.changes == {"时间": None}


def test_news_goal_and_aliases_cannot_keep_conflicting_old_requirements(tmp_path):
    from personal_assistant_agent.interaction.store import TaskStore

    store = TaskStore(str(tmp_path / "canonical"))
    try:
        cid, mid = str(uuid4()), str(uuid4())
        original = "搜索全球科技新闻，分十类，每类两条，附日期和来源链接"
        tid = store.receive(
            cid,
            mid,
            original,
            {
                "intent": "NEW_TASK",
                "goal": "搜索全球科技新闻",
                "constraints": {
                    "分类数量": "十类",
                    "每类结果数": "两条",
                    "输出要求": "附日期和来源链接",
                },
            },
            {},
        )["task_id"]
        store.control(
            tid,
            cid,
            str(uuid4()),
            "STEER",
            content="改为国内最近三天",
            changes={"地域": "中国大陆", "时间范围": "最近三天"},
        )
        task = store.task(tid)
        assert task["goal"] == "搜索新闻"
        assert task["constraints"] == {
            "范围": "中国大陆",
            "主题": "科技",
            "时间": "最近三天",
            "分类数量": "十类",
            "每类结果数": "两条",
            "输出要求": "附日期和来源链接",
        }
        assert "全球" not in task["effective_task_request"]
        assert task["original_request"] == original
        assert store.messages(cid)[0]["content"] == original
        store.finish(tid, task["revision"], "原成果")
        revised = store.receive(
            cid,
            str(uuid4()),
            "改成最近一周",
            {"intent": "STEER", "target_task_id": tid, "changes": {"时间范围": "最近一周"}},
            {},
        )
        new_task = store.task(revised["task_id"])
        assert new_task["source_task_id"] == tid and store.task(tid)["result"] == "原成果"
        assert new_task["constraints"]["时间"] == "最近一周"
        assert new_task["constraints"]["范围"] == "中国大陆"
    finally:
        store.db.close()


def test_router_preserves_structured_conditions_and_explicit_deletions():
    from personal_assistant_agent.interaction.routing import Decision

    parsed = Decision.model_validate(
        {
            "intent": "NEW_TASK",
            "constraints": {
                "categories": ["芯片", "人工智能", "航天", "生物科技"],
                "min_items_per_category": 2,
                "required_fields": ["日期", "来源", "链接"],
                "deduplication": True,
            },
        }
    )
    import json

    assert json.loads(parsed.constraints["categories"]) == ["芯片", "人工智能", "航天", "生物科技"]
    assert parsed.constraints["min_items_per_category"] == "2"
    assert parsed.constraints["deduplication"] == "true"
    assert Decision.model_validate({"intent": "STEER", "changes": {"时间": None}}).changes == {
        "时间": None
    }
