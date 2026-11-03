"""Native Harness hooks: durable directives, safe tool gates and root interrupts."""

import asyncio
import base64
import hashlib
import json
import time
from contextvars import ContextVar
from functools import wraps
from typing import NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import ModelResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import interrupt

# A stable operation ID reaches local calendar creation without becoming model input.
operation_id = ContextVar("interaction_operation_id", default=None)

# Unknown tools are conservatively treated as effects on crash.
READ_ONLY = {
    "baidu_search",
    "get_current_time",
    "calculate",
    "validate_time_budget",
    "ls",
    "read_file",
    "glob",
    "grep",
    "get_city_weather",
    "get_weather_alert",
    "geocode",
    "ip_locate",
    "check_cn_holiday",
    "convert_currency",
    "convert_units",
    "fetch_url",
    "fetch_rss",
    "translate",
    "route_plan",
    "search_flights",
    "search_hotels",
    "amap_maps",
    "write_todos",
}


class TaskBudgetExceeded(RuntimeError):
    pass


class ControlState(AgentState):
    directive_revision: NotRequired[int]
    effective_task_request: NotRequired[str]
    pause_consumed: NotRequired[int]


class ControlMiddleware(AgentMiddleware):
    state_schema = ControlState

    def __init__(self, store, task_id, settings, *, root=True, model_slots=None, tool_slots=None):
        self.store, self.task_id, self.settings, self.root = store, task_id, settings, root
        self.model_slots, self.tool_slots = model_slots, tool_slots

    @property
    def name(self):
        return "TaskControl" if self.root else "BranchControl"

    def branch(self):
        return ControlMiddleware(
            self.store,
            self.task_id,
            self.settings,
            root=False,
            model_slots=self.model_slots,
            tool_slots=self.tool_slots,
        )

    def task(self):
        return self.store.task(self.task_id)

    def directive(self, state):
        task = self.task()
        update = {
            "directive_revision": task["revision"],
            "effective_task_request": task["effective_task_request"],
        }
        if state.get("directive_revision", -1) != task["revision"]:
            update.update(delivery_checks=0, plan_checks=0)
            update["messages"] = [
                RemoveMessage(id=m.id)
                for m in state["messages"]
                if (
                    m.type == "system"
                    and str(m.content).startswith(("正文尚未交付：", "写作交付校验："))
                )
                or (m.type == "ai" and m.additional_kwargs.get("writing_rejected_draft"))
            ]
        # A stable user message is the effective task view, not a new user turn.
        # Raw user turns and superseded constraints remain immutable in TaskStore.
        user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if user:
            update.setdefault("messages", []).extend(
                [
                    HumanMessage(
                        content=task["effective_task_request"]
                        if self.root
                        else (
                            str(user.additional_kwargs.get("delegation", user.content))
                            + "\n父任务最新要求（约束本次委派）："
                            + task["effective_task_request"]
                        ),
                        id=user.id,
                        additional_kwargs={
                            **user.additional_kwargs,
                            "original_request": user.additional_kwargs.get(
                                "original_request", user.content
                            ),
                            **(
                                {
                                    "delegation": user.additional_kwargs.get(
                                        "delegation", user.content
                                    )
                                }
                                if not self.root
                                else {}
                            ),
                        },
                    )
                ]
            )
        return update

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        task = self.task()
        if task["desired_action"] == "CANCEL":
            return {"jump_to": "end"}
        update = self.directive(state)
        if self.root:
            pauses = [
                c
                for c in self.store.controls(self.task_id)
                if c["kind"] == "PAUSE" and c["seq"] > state.get("pause_consumed", 0)
            ]
            if pauses:
                epoch = pauses[-1]["seq"]
                # Replayed node still visits this same interrupt after RESUME.
                interrupt({"kind": "task_pause", "task_id": self.task_id, "epoch": epoch})
                update["pause_consumed"] = epoch
        return update

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        task = self.task()
        if task["desired_action"] == "CANCEL":
            return {"jump_to": "end"}
        if state.get("directive_revision", -1) != task["revision"]:
            update = self.directive(state)
            last = state["messages"][-1]
            # Discard the stale candidate before ToolNode sees its calls.
            if last.type == "ai":
                update.setdefault("messages", []).append(RemoveMessage(id=last.id))
            return {**update, "jump_to": "model"}
        return None

    def stale(self, request):
        task = self.task()
        return (
            task["desired_action"] == "CANCEL"
            or (self.root and task["desired_action"] == "PAUSE")
            or request.state.get("directive_revision", -1) != task["revision"]
        )

    async def acquire(self, slots, request):
        # Resource admission is a safe boundary too. Controls must not wait for
        # an unrelated task to release a saturated model/tool semaphore.
        while not self.stale(request):
            try:
                await asyncio.wait_for(slots.acquire(), 0.1)
                return True
            except TimeoutError:
                pass
        return False

    @staticmethod
    def model_control_response():
        return ModelResponse(
            result=[
                AIMessage(
                    content="控制已更新，等待安全边界。",
                    additional_kwargs={"stale_task_candidate": True},
                )
            ]
        )

    @staticmethod
    def deferred_tool(request):
        call = request.tool_call
        return ToolMessage(
            content="尚未执行：收到新控制，请按最新任务要求重新决策。",
            tool_call_id=call["id"],
            name=call["name"],
        )

    async def awrap_model_call(self, request, handler):
        if self.model_slots:
            if not await self.acquire(self.model_slots, request):
                return self.model_control_response()
            try:
                return await self.call_model(request, handler)
            finally:
                self.model_slots.release()
        return await self.call_model(request, handler)

    async def call_model(self, request, handler):
        if self.stale(request):
            return self.model_control_response()
        response = await handler(request)
        if self.stale(request):
            # Settle the transport, but never run delivery validators or tools
            # against an obsolete candidate. The after_model gate revisits model.
            return ModelResponse(
                result=[
                    AIMessage(
                        content="控制已更新，等待安全边界。",
                        additional_kwargs={"stale_task_candidate": True},
                    )
                ]
            )
        return response

    def operation_key(self, request):
        namespace = request.runtime.config.get("configurable", {}).get("checkpoint_ns", "root")
        return hashlib.sha256(
            f"{self.task_id}:{namespace}:{request.tool_call['id']}".encode()
        ).hexdigest()

    @staticmethod
    def call_hash(request):
        call = request.tool_call
        return hashlib.sha256(
            json.dumps(
                {"name": call["name"], "args": call["args"]}, sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()

    def cached_tool(self, request):
        prior = self.store.db.execute(
            "SELECT * FROM operations WHERE id=?", (self.operation_key(request),)
        ).fetchone()
        if prior and prior["status"] == "DONE" and prior["result"]:
            packed = json.loads(prior["result"])
            if packed["call_hash"] != self.call_hash(request):
                raise RuntimeError("Operation ID reused with different tool arguments")
            return JsonPlusSerializer().loads_typed(
                (
                    packed["encoding"],
                    base64.b64decode(packed["data"]),
                )
            )
        return None

    async def awrap_tool_call(self, request, handler):
        # Settled facts precede admission/control gates. A replay must never say
        # "not executed" about a side effect whose durable receipt already exists.
        if cached := self.cached_tool(request):
            return cached
        if self.tool_slots and request.tool_call["name"] != "task":
            if not await self.acquire(self.tool_slots, request):
                return self.deferred_tool(request)
            try:
                return await self.call_tool(request, handler)
            finally:
                self.tool_slots.release()
        return await self.call_tool(request, handler)

    async def call_tool(self, request, handler):
        call = request.tool_call
        namespace = request.runtime.config.get("configurable", {}).get("checkpoint_ns", "root")
        key = self.operation_key(request)
        with self.store.transaction():
            if cached := self.cached_tool(request):
                return cached
            task = self.task()
            if (
                task["desired_action"] == "CANCEL"
                or (self.root and task["desired_action"] == "PAUSE")
                or request.state.get("directive_revision", -1) != task["revision"]
            ):
                return ToolMessage(
                    content="尚未执行：收到新控制，请按最新任务要求重新决策。",
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            authorization = task.get("authorization")
            if (
                call["name"] in {"book_flight", "book_hotel"}
                and authorization
                and authorization["revision"] != task["revision"]
            ):
                return ToolMessage(
                    content="任务或控制版本已经变化，请用户重新单独确认原报价。",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            prior = self.store.db.execute("SELECT * FROM operations WHERE id=?", (key,)).fetchone()
            if prior and prior["status"] == "STARTED" and call["name"] not in READ_ONLY:
                raise RuntimeError("Unsettled side effect requires reconciliation")
            if task["tool_calls"] >= self.settings.max_tool_calls:
                raise TaskBudgetExceeded("Task tool call budget exhausted")
            task["tool_calls"] += 1
            task["progress"].setdefault("active_tools", {})[key] = {
                "name": call["name"],
                "call_id": call["id"],
                "namespace": namespace,
            }
            task["progress"]["last_progress_at"] = time.time()
            self.store.save(task)
            self.store.db.execute(
                "INSERT OR REPLACE INTO operations VALUES(?,?,?,?,NULL)",
                (
                    key,
                    self.task_id,
                    call["name"],
                    "STARTED",
                ),
            )
            self.store.event(
                task["conversation_id"],
                "tool.started",
                {
                    "name": call["name"],
                    "call_id": call["id"],
                    "namespace": namespace,
                },
                self.task_id,
            )
        token = operation_id.set(key)
        try:
            result = await handler(request)
        finally:
            operation_id.reset(token)
        # Persist complete native Commands as well as messages so a replay restores
        # reducers/todos without repeating the actual tool or delegated work.
        encoding, data = JsonPlusSerializer().dumps_typed(result)
        packed = json.dumps(
            {
                "encoding": encoding,
                "data": base64.b64encode(data).decode(),
                "call_hash": self.call_hash(request),
            }
        )
        with self.store.transaction():
            task = self.task()
            task["progress"].setdefault("active_tools", {}).pop(key, None)
            task["progress"]["last_progress_at"] = time.time()
            self.store.save(task)
            self.store.db.execute(
                "UPDATE operations SET status='DONE',result=? WHERE id=?",
                (
                    packed,
                    key,
                ),
            )
            self.store.event(
                task["conversation_id"],
                "tool.finished",
                {
                    "name": call["name"],
                    "call_id": call["id"],
                    "namespace": namespace,
                },
                self.task_id,
            )
        return result


def owner_callback(method):
    @wraps(method)
    async def routed(self, *args, **kwargs):
        if asyncio.get_running_loop() is self.owner_loop:
            return await method(self, *args, **kwargs)
        # Some SDK error paths run async callbacks on a helper loop/thread.
        # SQLite and the collector must still have exactly one owning writer.
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(method(self, *args, **kwargs), self.owner_loop)
        )

    return routed


class TaskBudgetCallback(AsyncCallbackHandler):
    """Count every transport invocation, including compaction and nested models."""

    raise_error = True
    # LangChain 1.7 builds inline callback coroutines eagerly; raising there
    # leaves later coroutines unawaited. Async gather safely propagates our gate.
    run_inline = False

    def __init__(self, store, task_id, settings, summary=None):
        self.store, self.task_id, self.settings = store, task_id, settings
        self.summary = summary
        self.owner_loop = asyncio.get_running_loop()

    def observed(self, task):
        if self.summary:
            task["trace_summary"] = self.summary()

    @owner_callback
    async def persist_observation(self):
        with self.store.transaction():
            task = self.store.task(self.task_id)
            self.observed(task)
            self.store.save(task)

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        await self.persist_observation()

    async def on_tool_end(self, output, *, run_id, **kwargs):
        await self.persist_observation()

    async def on_llm_error(self, error, *, run_id, **kwargs):
        await self.persist_observation()

    @owner_callback
    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.store.transaction():
            task = self.store.task(self.task_id)
            if task["model_calls"] >= self.settings.max_model_calls:
                raise TaskBudgetExceeded("Task model call budget exhausted")
            task["model_calls"] += 1
            self.observed(task)
            self.store.save(task)

    @owner_callback
    async def on_llm_end(self, response, *, run_id, **kwargs):
        usages = [
            getattr(g.message, "usage_metadata", None)
            for group in response.generations
            for g in group
        ]
        with self.store.transaction():
            task = self.store.task(self.task_id)
            measured = task.setdefault("measured_usage", {})
            for usage in usages:
                if usage:
                    measured[str(run_id)] = usage
            self.observed(task)
            self.store.save(task)


class ChatReadOnlyMiddleware(AgentMiddleware):
    """An ordinary reply cannot take over a task or issue external writes."""

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name not in READ_ONLY or name == "write_todos":
            return ToolMessage(
                content="普通聊天通道只支持只读查询；请将执行操作作为任务提交。",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return await handler(request)
