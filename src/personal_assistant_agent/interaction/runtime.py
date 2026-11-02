"""Durable bounded workers around the existing Deep Agent, never a second Agent Loop."""

import asyncio
import fcntl
import hashlib
import json
import logging
import re
import sqlite3
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command
from langsmith import Client, tracing_context
from openai import APIConnectionError, APITimeoutError, RateLimitError

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.artifacts import import_artifacts, index_artifacts
from personal_assistant_agent.interaction.middleware import (
    READ_ONLY,
    ControlMiddleware,
    TaskBudgetCallback,
    TaskBudgetExceeded,
)
from personal_assistant_agent.interaction.routing import MessageRouter
from personal_assistant_agent.interaction.store import ACTIVE, TERMINAL, Conflict, TaskStore
from personal_assistant_agent.observability.timeline import InvocationCollector, measured_stream
from personal_assistant_agent.observability.tracer import CostTracer
from personal_assistant_agent.schemas.chat import StoredChatRequest
from personal_assistant_agent.services.chat import ChatService, _state_updates, public_error

logger = logging.getLogger(__name__)


class CheckpointIntegrityError(RuntimeError):
    pass


def execution_fingerprint(settings):
    from personal_assistant_agent.services.prompts import load_prompt

    digest = hashlib.sha256(load_prompt(settings).text.encode())
    for directory in ["agents", "interaction"]:
        for path in sorted((Path(__file__).parents[1] / directory).glob("*.py")):
            digest.update(path.read_bytes())
    for package in ("deepagents", "langgraph", "langchain", "langgraph-checkpoint-sqlite"):
        digest.update(f"{package}={version(package)}".encode())
    digest.update(
        json.dumps(
            {
                "model": settings.model_name,
                "context": settings.context_window_tokens,
                "output_reserve": settings.context_output_reserve_tokens,
                "max_output": settings.max_output_tokens,
                "provider_url": settings.base_url,
                "travel_provider": settings.travel_provider,
                "travel_environment": settings.amadeus_environment,
                "tools": settings.context_tool_names,
            }
        ).encode()
    )
    return digest.hexdigest()


def cumulative_summary(base, current):
    result = {**current}
    for key in (
        "duration_ms",
        "model_calls",
        "tool_calls",
        "measured_model_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "model_cost_estimate_cny",
        "priced_model_calls",
        "unknown_tool_costs",
    ):
        result[key] += base.get(key, 0)
    for key in ("usage_complete", "model_cost_complete", "cost_complete"):
        result[key] = base.get(key, True) and current[key]
    return result


class InteractionRuntime:
    def __init__(
        self, settings: Settings, *, agent_factory=build_agent, router=None, chat_factory=None
    ):
        self.settings, self.agent_factory = settings, agent_factory
        self.router = router or MessageRouter(settings)
        self.chat_factory = chat_factory
        self.store = None
        self.stack = AsyncExitStack()
        self.running = {}
        self.controls = {}
        self.active_clocks = {}
        self.chat_running = {}
        self.routing_locks = {}
        self.wakeup = asyncio.Event()
        self.stopping = False
        self.persistence_fault = None
        self.scheduler = None
        self.fingerprint = None
        self.owner = None
        self.model_slots = asyncio.Semaphore(settings.task_concurrency)
        self.tool_slots = asyncio.Semaphore(settings.task_concurrency * 2)
        self.chat_slots = asyncio.Semaphore(2)
        self.trace_client = (
            Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
            if settings.tracing_configured
            else None
        )

    async def start(self):
        if not 1 <= self.settings.interaction_queue_limit <= 512:
            raise ValueError("接收队列上限必须为1–512。")
        if not 1 <= self.settings.task_concurrency <= 8:
            raise ValueError("后台任务并发必须为1–8。")
        private = Path(self.settings.interaction_dir).resolve()
        workspace = Path(self.settings.workspace_dir).resolve()
        roots = [workspace, Path(f"{workspace}-tasks"), Path(f"{workspace}-chats")]
        if any(private.is_relative_to(root) for root in roots):
            raise ValueError("任务数据库必须位于 Agent 工作区外。")
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        private.chmod(0o700)
        self.owner = (private / "runtime.lock").open("a")
        (private / "runtime.lock").chmod(0o600)
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            self.owner = None
            raise RuntimeError("任务运行时已被另一进程持有；仅支持一个服务进程。") from None
        try:
            self.store = TaskStore(
                str(private), control_limit=self.settings.interaction_queue_limit
            )
            checkpoint = private / "checkpoints.sqlite3"
            checkpoint.touch(mode=0o600, exist_ok=True)
            checkpoint.chmod(0o600)
            self.saver = await self.stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(checkpoint))
            )
            await self.saver.setup()
            self.fingerprint = await asyncio.to_thread(execution_fingerprint, self.settings)
            await self.recover()
            self.scheduler = asyncio.create_task(self.schedule())
            self.wakeup.set()
        except BaseException:
            await self.stack.aclose()
            if self.store:
                self.store.db.close()
            self.owner.close()
            self.owner = None
            raise
        return self

    async def recover(self):
        for task in self.store.tasks():
            if task["status"] in TERMINAL:
                continue
            if task.get("fingerprint") and task["fingerprint"] != self.fingerprint:
                self.store.state(
                    task["id"],
                    "RECOVERY_REQUIRED",
                    reason="执行版本改变，需迁移检查。",
                    recoverable=False,
                )
                continue
            unknown = self.store.db.execute(
                "SELECT name FROM operations WHERE task=? AND status='STARTED'",
                (task["id"],),
            ).fetchall()
            unsafe = [r[0] for r in unknown if r[0] not in READ_ONLY]
            if unsafe:
                self.store.state(
                    task["id"],
                    "RECOVERY_REQUIRED",
                    reason="存在结果不明的写操作，须核对后新建修订任务。",
                    unknown=unsafe,
                    recoverable=False,
                )
                continue
            config = {"configurable": {"thread_id": task["id"]}}
            checkpoint = await self.saver.aget_tuple(config)
            if checkpoint:
                revision = checkpoint.checkpoint["channel_values"].get("directive_revision", -1)
                self.store.applied(
                    task["id"],
                    revision,
                    checkpoint.config["configurable"]["checkpoint_id"],
                    checkpoint.checkpoint["channel_values"].get("effective_task_request"),
                )
            task = self.store.task(task["id"])
            if task["status"] in ACTIVE or task["status"] == "RECOVERING":
                # The last heartbeat reserved the next second of active work.
                # Process downtime is not task execution and must not consume budget.
                if task["desired_action"] == "CANCEL":
                    self.store.state(task["id"], "CANCELLED")
                elif task["status"] != "RECOVERY_REQUIRED":
                    self.store.state(task["id"], "QUEUED")
        # Accepted chat turns are durable too. A crash ends only that reply; no blind replay.
        for row in self.store.db.execute(
            "SELECT * FROM messages WHERE kind='CHAT' AND role='user'"
        ):
            receipt = json.loads(row["response"])
            if receipt.get("pending"):
                self.complete_chat(row["conversation"], row["id"], "", "服务重启，本条回复已中止。")

    async def submit(self, conversation, payload):
        if self.persistence_fault and payload.get("intent") != "STATUS":
            raise RuntimeError("任务数据库异常，需修复后重启。")
        request = {"conversation": conversation, **payload}
        if duplicate := self.store.duplicate(payload["client_message_id"], request):
            return duplicate
        # Explicit controls bypass any outstanding semantic routing or chat generation.
        lock = self.routing_locks.setdefault(conversation, asyncio.Lock())

        async def accept(quick=None):
            tasks = self.store.tasks(conversation)
            content = payload["content"]
            confirmation = re.fullmatch(r"确认预订 ([a-f0-9]{32})", content.strip())
            if confirmation and payload.get("intent") != "CHAT":
                candidates = [
                    t
                    for t in tasks
                    if t.get("result")
                    and (
                        confirmation[1] in t["result"]
                        or t["original_request"].strip() == content.strip()
                    )
                    and (not payload.get("target_task_id") or t["id"] == payload["target_task_id"])
                ]
                in_progress = [
                    t
                    for t in tasks
                    if t["status"] not in TERMINAL
                    and t["original_request"].strip() == content.strip()
                ]
                roots = {t.get("travel_owner_task_id", t["id"]) for t in candidates}
                if in_progress and not payload.get("target_task_id"):
                    decision = {
                        "intent": "CLARIFY",
                        "question": "此预订任务尚未结束，请先明确恢复或取消目标。",
                        "candidates": [t["id"] for t in in_progress],
                    }
                elif len(roots) == 1:
                    source = max(candidates, key=lambda t: t["created_at"])
                    decision = {
                        "intent": "NEW_TASK",
                        "source_task_id": source["id"],
                        "goal": "继续处理已明确确认的原预订请求",
                        "constraints": {**source["constraints"], "预订确认": confirmation[1]},
                    }
                else:
                    decision = {
                        "intent": "CLARIFY",
                        "question": "请指定已交付确认编号的原预订任务。",
                        "candidates": [t["id"] for t in candidates],
                    }
            else:
                decision = (
                    quick
                    if quick is not None
                    else await self.router.route(
                        content,
                        tasks,
                        self.store.messages(conversation),
                        payload.get("intent"),
                        payload.get("target_task_id"),
                    )
                )
            if target_id := decision.get("target_task_id"):
                prior = next((t for t in tasks if t["id"] == target_id), None)
                if prior:
                    decision["expected_revision"] = prior["revision"]
                    if decision["intent"] == "NEW_TASK" and prior["status"] in TERMINAL:
                        decision["source_task_id"] = target_id
            result = self.store.receive(
                conversation,
                payload["client_message_id"],
                payload["content"],
                decision,
                request,
                queue_limit=self.settings.interaction_queue_limit,
            )
            self.wakeup.set()
            return result

        if payload.get("intent") in {"STATUS", "PAUSE", "RESUME", "CANCEL"}:
            return await accept()
        if hasattr(self.router, "quick_route"):
            quick = self.router.quick_route(
                payload["content"],
                self.store.tasks(conversation),
                [],
                payload.get("intent"),
                payload.get("target_task_id"),
            )
            if quick and quick["intent"] in {
                "STEER",
                "STATUS",
                "PAUSE",
                "RESUME",
                "CANCEL",
                "CLARIFY",
            }:
                return await accept(quick)
        async with lock:
            return await accept()

    async def schedule(self):
        while not self.stopping:
            self.wakeup.clear()
            for task_id in list(self.active_clocks):
                try:
                    self.account_active(task_id, reserve=1)
                except (OSError, sqlite3.Error) as error:
                    self.persistence_fault = type(error).__name__
                    for control in self.controls.values():
                        control.request_drain("persistence unavailable")
            for task in [] if self.persistence_fault else self.store.tasks():
                if len(self.running) >= self.settings.task_concurrency:
                    break
                if task["status"] != "QUEUED" or task["id"] in self.running:
                    continue
                task_id = task["id"]
                self.running[task_id] = asyncio.create_task(self.execute(task_id))
            conversations = {
                r[0]
                for r in self.store.db.execute(
                    "SELECT DISTINCT conversation FROM messages WHERE kind='CHAT' AND role='user'"
                )
            }
            for conversation in conversations:
                if (
                    not self.persistence_fault
                    and conversation not in self.chat_running
                    and self.pending_chat(conversation)
                ):
                    self.chat_running[conversation] = asyncio.create_task(
                        self.respond(conversation)
                    )
            try:
                await asyncio.wait_for(self.wakeup.wait(), 0.25)
            except TimeoutError:
                pass

    def task_settings(self, task_id):
        # A sibling root prevents the legacy root backend from racing with task files.
        return replace(
            self.settings, workspace_dir=f"{self.settings.workspace_dir}-tasks/{task_id}"
        )

    async def execute(self, task_id):
        collector = None
        base_summary = {}
        control = RunControl()
        self.controls[task_id] = control
        try:
            task = self.store.task(task_id)
            base_summary = task.get("trace_summary", {})
            if task["status"] != "QUEUED":
                return
            remaining = self.settings.workflow_timeout - task["active_seconds"]
            if (
                remaining <= 0
                or task["model_calls"] >= self.settings.max_model_calls
                or task["tool_calls"] >= self.settings.max_tool_calls
            ):
                raise TaskBudgetExceeded("Task active time budget exhausted")
            run_id = uuid4()
            self.store.state(
                task_id,
                "RUNNING",
                attempt_id=str(run_id),
                tracing_enabled=self.settings.tracing_configured,
                attempt=task["attempt"] + 1,
                attempt_started=time.time(),
                fingerprint=self.fingerprint,
                active_seconds=task["active_seconds"] + 1,
            )
            self.active_clocks[task_id] = (time.monotonic(), task["active_seconds"])
            scoped = self.task_settings(task_id)
            graph = await asyncio.to_thread(
                self.agent_factory,
                scoped,
                checkpointer=self.saver,
                control=ControlMiddleware(
                    self.store,
                    task_id,
                    self.settings,
                    model_slots=self.model_slots,
                    tool_slots=self.tool_slots,
                ),
            )
            queue = asyncio.Queue()
            collector = InvocationCollector(scoped, queue)
            source = (
                self.store.task(task["source_task_id"], task["conversation_id"])
                if task["source_task_id"]
                else None
            )
            owner_task = source.get("travel_owner_task_id", source["id"]) if source else task_id
            confirmation = re.fullmatch(
                r"确认预订 ([a-f0-9]{32})", task["original_request"].strip()
            )
            with self.store.transaction():
                current = self.store.task(task_id)
                current["travel_owner_task_id"] = owner_task
                authorized = bool(
                    confirmation
                    and source
                    and source.get("result")
                    and (
                        confirmation[1] in source["result"]
                        or source["original_request"].strip() == task["original_request"].strip()
                    )
                    and current["revision"] == 0
                )
                if authorized:
                    current["authorization"] = {
                        "revision": 0,
                        "task_id": task_id,
                        "message_id": current["authorization_message_id"],
                        "source_task_id": source["id"],
                    }
                self.store.save(current)
            config = {
                "configurable": {
                    "thread_id": task_id,
                    "travel_context": {
                        "conversation_id": task["conversation_id"] + ":" + owner_task,
                        "confirmation": confirmation[1] if authorized else "",
                        "task_id": task_id,
                        "authorization_message_id": current["authorization_message_id"],
                    },
                },
                "recursion_limit": self.settings.recursion_limit,
                "run_id": run_id,
                "callbacks": [
                    collector,
                    TaskBudgetCallback(
                        self.store,
                        task_id,
                        self.settings,
                        summary=lambda: cumulative_summary(base_summary, collector.summary()),
                    ),
                    *(
                        [
                            CostTracer(
                                settings=scoped,
                                client=self.trace_client,
                                project_name=scoped.langsmith_project,
                            )
                        ]
                        if self.trace_client
                        else []
                    ),
                ],
                "metadata": {"task_id": task_id, "conversation_id": task["conversation_id"]},
            }
            async with asyncio.timeout(remaining):
                state = await graph.aget_state(config)
                if state.values and not any(
                    isinstance(m, HumanMessage) for m in state.values.get("messages", [])
                ):
                    raise CheckpointIntegrityError("Checkpoint message chain is incomplete")
                if state.interrupts:
                    input_value = self.resume_command(state)
                elif not state.values:
                    input_value = {
                        "messages": [
                            HumanMessage(
                                content=task["effective_task_request"], id=f"task-request:{task_id}"
                            )
                        ],
                    }
                    imported = []
                    if source and source.get("artifacts"):
                        operation = "import:" + task_id
                        with self.store.transaction():
                            self.store.db.execute(
                                "INSERT OR IGNORE INTO operations VALUES(?,?,?,?,NULL)",
                                (operation, task_id, "import_artifacts", "STARTED"),
                            )
                        imported = await asyncio.to_thread(
                            import_artifacts,
                            source,
                            self.task_settings(source["id"]),
                            scoped,
                        )
                        with self.store.transaction():
                            self.store.db.execute(
                                "UPDATE operations SET status='DONE',result=? WHERE id=?",
                                (json.dumps(imported), operation),
                            )
                    ancestry = []
                    cursor = source
                    seen = set()
                    while cursor:
                        if cursor["id"] in seen:
                            raise CheckpointIntegrityError("Cyclic result reference")
                        seen.add(cursor["id"])
                        if cursor.get("result"):
                            ancestry.append(cursor["result"])
                        cursor = (
                            self.store.task(cursor["source_task_id"], task["conversation_id"])
                            if cursor["source_task_id"]
                            else None
                        )
                    if ancestry:
                        from langchain_core.messages import AIMessage

                        history = [
                            AIMessage(content="关联原成果（历史资料，须按最新要求修订）：\n" + text)
                            for text in reversed(ancestry)
                        ]
                        history.append(
                            AIMessage(
                                content="原成果文件副本："
                                + json.dumps(imported, ensure_ascii=False)
                            )
                        )
                        input_value["messages"] = history + input_value["messages"]
                else:
                    input_value = None
                while True:
                    if not state.next and state.values and input_value is None:
                        # Finished checkpoint, but final transaction may have been lost.
                        if (
                            state.values.get("directive_revision", -1)
                            != self.store.task(task_id)["revision"]
                        ):
                            # Only this worker updates the same thread. Force another decision
                            # from the native before_model boundary without resetting history.
                            await graph.aupdate_state(
                                config,
                                {
                                    "messages": [RemoveMessage(id=state.values["messages"][-1].id)],
                                },
                                as_node="tools",
                            )
                    stream_source = self.stream_graph(
                        graph,
                        input_value,
                        config,
                        stream_mode=["messages", "updates", "checkpoints"],
                        durability="sync",
                        control=control,
                    )
                    async for mode, data in measured_stream(stream_source, queue):
                        if mode == "trace":
                            with self.store.transaction():
                                self.store.event(task["conversation_id"], "trace", data, task_id)
                        elif mode == "updates":
                            for value in data.values():
                                for update in _state_updates(value):
                                    if "todos" in update:
                                        with self.store.transaction():
                                            current = self.store.task(task_id)
                                            current["progress"]["todos"] = update["todos"]
                                            self.store.save(current)
                                            self.store.event(
                                                task["conversation_id"],
                                                "plan.updated",
                                                {"todos": update["todos"]},
                                                task_id,
                                            )
                        elif mode == "checkpoints":
                            # Verify the persisted checkpoint before publishing APPLIED.
                            saved = await self.saver.aget_tuple(data["config"])
                            if saved:
                                rev = saved.checkpoint["channel_values"].get(
                                    "directive_revision", -1
                                )
                                self.store.applied(
                                    task_id,
                                    rev,
                                    saved.config["configurable"]["checkpoint_id"],
                                    saved.checkpoint["channel_values"].get(
                                        "effective_task_request"
                                    ),
                                )
                    state = await graph.aget_state(config)
                    revision = state.values.get("directive_revision", -1)
                    self.store.applied(
                        task_id,
                        revision,
                        state.config["configurable"].get("checkpoint_id"),
                        state.values.get("effective_task_request"),
                    )
                    with self.store.transaction():
                        current = self.store.task(task_id)
                        current["trace_summary"] = cumulative_summary(
                            base_summary, collector.summary()
                        )
                        self.store.save(current)
                        self.store.event(
                            task["conversation_id"], "task.usage", current["trace_summary"], task_id
                        )
                    current = self.store.task(task_id)
                    if current["desired_action"] == "CANCEL":
                        self.store.state(task_id, "CANCELLED")
                        break
                    if state.interrupts:
                        valid = all(i.value.get("kind") == "task_pause" for i in state.interrupts)
                        self.store.state(task_id, "PAUSED" if valid else "WAITING_INPUT")
                        if valid and current["desired_action"] == "RUN":
                            self.store.state(task_id, "QUEUED")
                        break
                    if state.next:
                        raise RuntimeError("Graph stopped with pending nodes")
                    result = next(
                        (
                            m.content
                            for m in reversed(state.values["messages"])
                            if m.type == "ai" and not getattr(m, "tool_calls", None)
                        ),
                        "",
                    )
                    if not result:
                        raise RuntimeError("Graph completed without final delivery")
                    artifacts = await asyncio.to_thread(index_artifacts, scoped)
                    with self.store.transaction():
                        current = self.store.task(task_id)
                        current["artifacts"] = artifacts
                        self.store.save(current)
                    if self.store.finish(task_id, revision, result):
                        break
                    # A control won the final transaction. Continue this same native graph.
                    await graph.aupdate_state(
                        config,
                        {
                            "messages": [RemoveMessage(id=state.values["messages"][-1].id)],
                        },
                        as_node="tools",
                    )
                    state = await graph.aget_state(config)
                    input_value = None
        except GraphDrained:
            self.store.state(task_id, "QUEUED")
        except asyncio.CancelledError:
            self.store.state(task_id, "RECOVERING", reason="执行中断，重启时核对现场。")
            raise
        except Exception as error:
            code, message = public_error(error)
            if isinstance(error, (OSError, sqlite3.Error, CheckpointIntegrityError)):
                code = "CHECKPOINT_REVIEW_REQUIRED"
                message = "持久化写入或消息链异常，需核对执行现场后再创建修订任务。"
            elif isinstance(error, TaskBudgetExceeded):
                code = "TASK_BUDGET_EXHAUSTED"
                message = "任务累计执行预算已耗尽，请核对已有成果后创建后续任务。"
            try:
                unknown = self.store.db.execute(
                    "SELECT name FROM operations WHERE task=? AND status='STARTED'",
                    (task_id,),
                ).fetchall()
            except (OSError, sqlite3.Error):
                unknown = [("unavailable",)]
            status = (
                "RECOVERY_REQUIRED"
                if isinstance(
                    error,
                    (
                        TaskBudgetExceeded,
                        CheckpointIntegrityError,
                        TimeoutError,
                        OSError,
                        sqlite3.Error,
                        APIConnectionError,
                        APITimeoutError,
                        RateLimitError,
                    ),
                )
                or any(r[0] not in READ_ONLY for r in unknown)
                else "FAILED"
            )
            recoverable = (
                isinstance(
                    error,
                    (
                        APIConnectionError,
                        APITimeoutError,
                        RateLimitError,
                    ),
                )
                and not unknown
            )
            try:
                self.store.state(
                    task_id,
                    status,
                    error={"code": code, "message": message},
                    recoverable=recoverable,
                    **(
                        {"trace_summary": cumulative_summary(base_summary, collector.summary())}
                        if collector
                        else {}
                    ),
                )
            except (OSError, sqlite3.Error) as persistence_error:
                self.persistence_fault = type(persistence_error).__name__
                logger.warning("Task persistence unavailable; claiming suspended")
        finally:
            try:
                self.account_active(task_id)
            except (OSError, sqlite3.Error) as persistence_error:
                self.persistence_fault = type(persistence_error).__name__
                logger.warning("Task persistence unavailable; claiming suspended")
            finally:
                self.active_clocks.pop(task_id, None)
                self.controls.pop(task_id, None)
                self.running.pop(task_id, None)
                self.wakeup.set()

    def account_active(self, task_id, reserve=0):
        clock = self.active_clocks.get(task_id)
        if clock:
            started, base = clock
            with self.store.transaction():
                task = self.store.task(task_id)
                task["active_seconds"] = base + time.monotonic() - started + reserve
                self.store.save(task)

    async def stream_graph(self, graph, input_value, config, **kwargs):
        with tracing_context(enabled=self.settings.tracing_configured):
            async for event in graph.astream(input_value, config, **kwargs):
                yield event

    @staticmethod
    def resume_command(state):
        if not state.interrupts or any(
            i.value.get("kind") != "task_pause" for i in state.interrupts
        ):
            raise Conflict("不允许通过任务恢复批准其他人工审核。")
        return Command(resume={i.id: {"resume": True} for i in state.interrupts})

    def pending_chat(self, conversation):
        return [
            m
            for m in self.store.messages(conversation, "CHAT")
            if m["role"] == "user" and json.loads(m["response"]).get("pending")
        ]

    async def respond(self, conversation):
        try:
            async with self.chat_slots:
                await self._respond(conversation)
        except asyncio.CancelledError:
            pending = self.pending_chat(conversation)
            if pending:
                self.complete_chat(conversation, pending[0]["id"], "", "已停止本条回复。")
        finally:
            self.chat_running.pop(conversation, None)
            self.wakeup.set()

    async def _respond(self, conversation):
        row = self.pending_chat(conversation)[0]
        message_id = row["id"]
        service = None
        try:
            scoped = replace(
                self.settings, workspace_dir=f"{self.settings.workspace_dir}-chats/{conversation}"
            )
            service = (
                self.chat_factory(scoped)
                if self.chat_factory
                else ChatService(
                    await asyncio.to_thread(self.agent_factory, scoped, read_only=True), scoped
                )
            )
            # Pair replies by the user ID, not by reply insertion time: a queued
            # follow-up may have arrived before the preceding reply was written.
            turns = self.store.messages(conversation, "CHAT")
            replies = {m["id"]: m for m in turns if m["role"] == "assistant"}
            # Completed results are conversation facts, not execution history.
            # A follow-up may discuss them without inheriting tools, todos or authority.
            messages = [
                {
                    "role": "assistant",
                    "content": "已交付的任务结果（仅作问答资料，不接手原任务）：\n" + m["content"],
                }
                for m in self.store.messages(conversation, "RESULT")
                if m["seq"] <= row["seq"]
            ]
            for turn in turns:
                if turn["role"] != "user" or turn["seq"] > row["seq"]:
                    continue
                messages.append({"role": "user", "content": turn["content"]})
                if reply := replies.get("chat:" + turn["id"]):
                    messages.append({"role": "assistant", "content": reply["content"]})
            # These are validated durable rows, not client-supplied history. Let
            # the existing Harness archive/compress them instead of silently
            # dropping early turns to fit the legacy HTTP request's item limit.
            payload = StoredChatRequest(conversation_id=conversation, messages=messages)
            result = ""
            error = None
            async for event in service.stream(payload, str(uuid4())):
                kind = event["type"]
                if kind == "delta":
                    result += event["text"]
                if kind == "error":
                    error = event["message"]
                with self.store.transaction():
                    self.store.event(conversation, "chat." + kind, event, chat=message_id)
            self.complete_chat(conversation, message_id, result, error)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.complete_chat(conversation, message_id, "", "本条回复失败，请重试。")
        finally:
            if service:
                await service.close()
            self.chat_running.pop(conversation, None)
            self.wakeup.set()

    def complete_chat(self, conversation, message_id, text, error=None):
        with self.store.transaction():
            row = self.store.db.execute(
                "SELECT response FROM messages WHERE id=?", (message_id,)
            ).fetchone()
            receipt = json.loads(row[0])
            if not receipt.get("pending"):
                return
            receipt["pending"] = False
            receipt["error"] = error
            self.store.db.execute(
                "UPDATE messages SET response=? WHERE id=?",
                (
                    json.dumps(receipt),
                    message_id,
                ),
            )
            if text and not error:
                self.store.db.execute(
                    "INSERT OR IGNORE INTO messages"
                    "(conversation,id,content,role,kind) VALUES(?,?,?,?,?)",
                    (
                        conversation,
                        "chat:" + message_id,
                        text,
                        "assistant",
                        "CHAT",
                    ),
                )
            self.store.event(
                conversation, "chat.settled", {"text": text, "error": error}, chat=message_id
            )

    def stop_chat(self, conversation, message_id):
        pending = self.pending_chat(conversation)
        if not any(m["id"] == message_id for m in pending):
            raise Conflict("此回复已结束或不属于当前会话。")
        if pending[0]["id"] == message_id and conversation in self.chat_running:
            self.chat_running[conversation].cancel()
        else:
            self.complete_chat(conversation, message_id, "", "已停止本条回复。")

    async def close(self):
        self.stopping = True
        self.wakeup.set()
        if self.scheduler:
            await asyncio.gather(self.scheduler, return_exceptions=True)
        for control in self.controls.values():
            control.request_drain("shutdown")
        # Existing tools finish before the saver closes; process kill uses crash recovery.
        await asyncio.gather(*list(self.running.values()), return_exceptions=True)
        for chat in list(self.chat_running.values()):
            chat.cancel()
        await asyncio.gather(*list(self.chat_running.values()), return_exceptions=True)
        try:
            await self.stack.aclose()
        finally:
            try:
                if self.store:
                    self.store.db.close()
            finally:
                if self.owner:
                    self.owner.close()
        if self.trace_client:
            await asyncio.to_thread(self.trace_client.flush)
            await asyncio.to_thread(self.trace_client.close)
