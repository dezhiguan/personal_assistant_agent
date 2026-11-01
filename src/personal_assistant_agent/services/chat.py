"""Stream Deep Agents output with native LangSmith tracing."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.ai import add_usage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from langsmith import Client, tracing_context
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import TypeAdapter

from personal_assistant_agent.agents.token_budget import TokenBudgetViolation, TokenCountUnavailable
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.observability.costs import estimate_calls
from personal_assistant_agent.observability.timeline import InvocationCollector, measured_stream
from personal_assistant_agent.observability.tracer import CostTracer
from personal_assistant_agent.schemas.chat import ChatRequest, TodoList, ToolProgress
from personal_assistant_agent.services.plan_edits import explicit_edit

logger = logging.getLogger(__name__)
todo_adapter = TypeAdapter(TodoList)


def _state_updates(value):
    """Compaction can make a model node emit Commands instead of a plain update."""
    if isinstance(value, dict):
        yield value
    elif isinstance(value, Command) and isinstance(value.update, dict):
        yield value.update
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_updates(item)


def _tool_failed(message: ToolMessage) -> bool:
    if message.status == "error":
        return True
    if isinstance(message.content, str):
        try:
            payload = json.loads(message.content)
            return isinstance(payload, dict) and bool(payload.get("error"))
        except ValueError:
            pass
    return False


def public_error(error: Exception) -> tuple[str, str]:
    """Return safe errors without provider bodies, keys, or stack traces."""
    from langchain_core.exceptions import ContextOverflowError

    if isinstance(error, TokenCountUnavailable):
        return "MODEL_TOKEN_COUNT_UNAVAILABLE", "无法核验本次 Token 预算，已停止生成，请稍后重试。"
    if isinstance(error, TokenBudgetViolation):
        return "MODEL_CONTEXT_ACCOUNTING_ERROR", "模型实际用量超过模拟窗口，已停止本轮调用。"

    if isinstance(error, ContextOverflowError):
        return (
            "MODEL_CONTEXT_LIMIT",
            "压缩后仍超过上下文预算，请缩短本次输入、减少启用工具或调大模拟窗口。",
        )
    if isinstance(error, AuthenticationError):
        return "MODEL_AUTH_ERROR", "Qwen 认证失败，请检查服务端的 API Key 和地域。"
    if isinstance(error, RateLimitError):
        return "MODEL_RATE_LIMIT", "模型服务暂时繁忙或额度不足，请稍后重试。"
    if isinstance(error, GraphRecursionError):
        return (
            "WORKFLOW_STEP_LIMIT",
            "任务达到执行步骤上限，已保存的文件和计划仍保留。请核对现有成果后继续。",
        )
    if isinstance(error, (ModelCallLimitExceededError, ToolCallLimitExceededError)):
        return (
            "WORKFLOW_CALL_LIMIT",
            "本轮任务达到调用次数上限，已停止继续调用。请核对已完成成果后继续。",
        )
    if isinstance(error, APITimeoutError):
        return "MODEL_TIMEOUT", "等待模型回复超时，请重试。"
    if isinstance(error, TimeoutError):
        return (
            "WORKFLOW_TIMEOUT",
            "本轮任务达到总时限，已保存的文件和计划仍保留。请核对现有成果后继续。",
        )
    if isinstance(error, APIConnectionError):
        return "MODEL_CONNECTION_ERROR", "暂时无法连接 Qwen 服务，请稍后重试。"
    return "MODEL_ERROR", "这次回复未能完成，请重试或开启新对话。"


class ChatService:
    def __init__(self, agent: Any, settings: Settings) -> None:
        if not 0 < settings.workflow_timeout <= 3600:
            raise ValueError("WORKFLOW_TIMEOUT_SECONDS 必须大于0且不超过3600。")
        if not 16 <= settings.recursion_limit <= 10000:
            raise ValueError("AGENT_RECURSION_LIMIT 必须为16–10000。")
        self.agent = agent
        self.settings = settings
        self.trace_client = (
            Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
            if settings.tracing_configured
            else None
        )

    async def stream(self, payload: ChatRequest, request_id: str) -> AsyncIterator[dict]:
        queue = asyncio.Queue()
        usage_collector = InvocationCollector(self.settings, queue)
        callbacks = (
            [
                CostTracer(
                    settings=self.settings,
                    client=self.trace_client,
                    project_name=self.settings.langsmith_project,
                )
            ]
            if self.trace_client
            else []
        )
        config = {
            "configurable": {
                "travel_context": {
                    "conversation_id": str(payload.conversation_id),
                    "confirmation": (
                        match.group(1)
                        if (
                            match := re.fullmatch(
                                r"确认预订 ([a-f0-9]{32})", payload.messages[-1].content.strip()
                            )
                        )
                        else ""
                    ),
                }
            },
            "run_id": UUID(request_id),
            "run_name": "personal-assistant-chat",
            "recursion_limit": self.settings.recursion_limit,
            "callbacks": [usage_collector, *callbacks],
            "metadata": {
                "conversation_id": str(payload.conversation_id),
                "session_id": str(payload.conversation_id),
                "request_id": request_id,
                "prompt_ref": self.settings.prompt_ref or "local",
            },
            "tags": ["personal-assistant", "mvp", "qwen"],
        }
        yield {
            "type": "meta",
            "request_id": request_id,
            "model": self.settings.model_name,
            "tracing_enabled": self.settings.tracing_configured,
            "started_at": usage_collector.started_at,
            "timeline_version": 1,
        }
        received_text = False
        buffered_text = ""
        buffered_chunks = []
        final_text = ""
        usage = None
        call_usage: dict[str, dict] = {}
        todos = [item.model_dump() for item in payload.todos]
        edit = explicit_edit(payload.messages[-1].content, todos)
        if edit is not None:
            todos, acknowledgement = edit
            yield {"type": "plan", "todos": todos}
            if acknowledgement:
                yield {"type": "delta", "text": acknowledgement}
                yield {
                    "type": "done",
                    "request_id": request_id,
                    "trace_summary": usage_collector.summary(),
                }
                return
        tool_calls: dict[str, str] = {}
        try:
            with tracing_context(enabled=self.settings.tracing_configured):
                # A workflow makes several model calls; each call already has
                # its own request_timeout enforced by the model transport.
                async with asyncio.timeout(self.settings.workflow_timeout):
                    source = self.agent.astream(
                        {
                            "messages": [m.model_dump() for m in payload.messages],
                            "todos": todos,
                        },
                        config=config,
                        stream_mode=["messages", "updates"],
                    )
                    async for mode, data in measured_stream(source, queue):
                        if mode == "trace":
                            yield data
                            continue
                        if mode == "updates":
                            for update in (
                                update
                                for value in data.values()
                                for update in _state_updates(value)
                            ):
                                for message in update.get("messages", []):
                                    if isinstance(message, AIMessage):
                                        buffered_text = ""
                                        buffered_chunks = []
                                        if not message.tool_calls and isinstance(
                                            message.content, str
                                        ):
                                            final_text = message.content
                                        else:
                                            final_text = ""
                                        for call in message.tool_calls:
                                            if call["name"] == "write_todos":
                                                continue
                                            tool_calls[call["id"]] = call["name"]
                                            yield ToolProgress(
                                                name=call["name"],
                                                call_id=call["id"],
                                                status="running",
                                            ).model_dump()
                                    elif isinstance(message, ToolMessage):
                                        name = tool_calls.pop(message.tool_call_id, None)
                                        if name:
                                            yield ToolProgress(
                                                name=name,
                                                call_id=message.tool_call_id,
                                                status=(
                                                    "error" if _tool_failed(message) else "finished"
                                                ),
                                            ).model_dump()
                                if "todos" not in update:
                                    continue
                                updated = [
                                    item.model_dump()
                                    for item in todo_adapter.validate_python(update["todos"])
                                ]
                                if updated != todos:
                                    todos = updated
                                    yield {"type": "plan", "todos": todos}
                            continue
                        if mode != "messages":
                            continue
                        chunk, metadata = data
                        if (
                            metadata.get("langgraph_node") != "model"
                            or metadata.get("lc_source") == "summarization"
                            or metadata.get("lc_internal_call")
                        ):
                            continue
                        if chunk_usage := getattr(chunk, "usage_metadata", None):
                            usage = add_usage(usage, chunk_usage)
                            call_id = getattr(chunk, "id", None) or str(
                                metadata.get("langgraph_step", "model")
                            )
                            call_usage[call_id] = add_usage(call_usage.get(call_id), chunk_usage)
                        content = chunk.content
                        text = (
                            content
                            if isinstance(content, str)
                            else "".join(
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        )
                        if text:
                            received_text = True
                            buffered_text += text
                            buffered_chunks.append(text)
            delivered = final_text or buffered_text
            if delivered:
                for part in [final_text] if final_text else buffered_chunks:
                    yield {"type": "delta", "text": part}
            if not received_text and not delivered:
                raise RuntimeError("Model completed without text")
            if usage_collector.calls:
                call_usage = usage_collector.calls
                usage = None
                for measured in call_usage.values():
                    usage = add_usage(usage, measured)
            if usage:
                cost_estimate = estimate_calls(call_usage.values(), self.settings)
                model_spans = [s for s in usage_collector.spans.values() if s["kind"] == "model"]
                if any(
                    s["name"] != self.settings.model_name or s.get("usage") is None
                    for s in model_spans
                ):
                    # The legacy usage estimate assumes one model. The timeline
                    # prices actual models individually and flags missing usage.
                    cost_estimate = None
                yield {
                    "type": "usage",
                    **usage,
                    "cost_estimate": cost_estimate,
                }
            yield {
                "type": "done",
                "request_id": request_id,
                "trace_summary": usage_collector.summary(),
            }
        except asyncio.CancelledError:
            logger.info("Chat cancelled request_id=%s", request_id)
            raise
        except Exception as error:
            usage_collector.close_open_spans("error")
            while not queue.empty():
                mode, data = queue.get_nowait()
                if mode == "trace":
                    yield data
            code, message = public_error(error)
            logger.warning(
                "Chat failed request_id=%s code=%s error_type=%s",
                request_id,
                code,
                type(error).__name__,
            )
            for call_id, name in tool_calls.items():
                yield ToolProgress(name=name, call_id=call_id, status="error").model_dump()
            yield {
                "type": "error",
                "code": code,
                "message": message,
                "request_id": request_id,
                "trace_summary": usage_collector.summary(),
            }

    async def close(self) -> None:
        if self.trace_client:
            await asyncio.to_thread(self.trace_client.flush)
            await asyncio.to_thread(self.trace_client.close)
