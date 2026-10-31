"""Stream Deep Agents output with native LangSmith tracing."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.ai import add_usage
from langsmith import Client, tracing_context
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import TypeAdapter

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.observability.costs import estimate_calls
from personal_assistant_agent.observability.tracer import CostTracer
from personal_assistant_agent.schemas.chat import ChatRequest, TodoList, ToolProgress
from personal_assistant_agent.services.plan_edits import explicit_edit

logger = logging.getLogger(__name__)
todo_adapter = TypeAdapter(TodoList)


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


class _ModelUsageCollector(BaseCallbackHandler):
    """Collect final usage from every model call, including nested task agents.

    Nested agent chunks are intentionally absent from the public message stream.
    Callback inheritance still provides their measured usage without exposing text.
    """

    run_inline = True

    def __init__(self):
        self.calls: dict[str, dict] = {}

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = None
        for generations in response.generations:
            for generation in generations:
                metadata = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if metadata:
                    usage = add_usage(usage, metadata)
        if usage:
            self.calls[str(run_id)] = usage


def public_error(error: Exception) -> tuple[str, str]:
    """Return safe errors without provider bodies, keys, or stack traces."""
    if isinstance(error, AuthenticationError):
        return "MODEL_AUTH_ERROR", "Qwen 认证失败，请检查服务端的 API Key 和地域。"
    if isinstance(error, RateLimitError):
        return "MODEL_RATE_LIMIT", "模型服务暂时繁忙或额度不足，请稍后重试。"
    if isinstance(error, (TimeoutError, APITimeoutError)):
        return "MODEL_TIMEOUT", "等待模型回复超时，请重试。"
    if isinstance(error, APIConnectionError):
        return "MODEL_CONNECTION_ERROR", "暂时无法连接 Qwen 服务，请稍后重试。"
    return "MODEL_ERROR", "这次回复未能完成，请重试或开启新对话。"


class ChatService:
    def __init__(self, agent: Any, settings: Settings) -> None:
        self.agent = agent
        self.settings = settings
        self.trace_client = (
            Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
            if settings.tracing_configured
            else None
        )

    async def stream(self, payload: ChatRequest, request_id: str) -> AsyncIterator[dict]:
        usage_collector = _ModelUsageCollector()
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
            "run_id": UUID(request_id),
            "run_name": "personal-assistant-chat",
            "recursion_limit": 100,
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
                yield {"type": "done", "request_id": request_id}
                return
        tool_calls: dict[str, str] = {}
        try:
            with tracing_context(enabled=self.settings.tracing_configured):
                # A workflow makes several model calls; each call already has
                # its own request_timeout enforced by the model transport.
                async with asyncio.timeout(max(300, self.settings.request_timeout + 10)):
                    async for mode, data in self.agent.astream(
                        {
                            "messages": [m.model_dump() for m in payload.messages],
                            "todos": todos,
                        },
                        config=config,
                        stream_mode=["messages", "updates"],
                    ):
                        if mode == "updates":
                            for update in data.values():
                                if not isinstance(update, dict):
                                    continue
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
                        if metadata.get("langgraph_node") != "model":
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
                yield {
                    "type": "usage",
                    **usage,
                    "cost_estimate": estimate_calls(call_usage.values(), self.settings),
                }
            yield {"type": "done", "request_id": request_id}
        except asyncio.CancelledError:
            logger.info("Chat cancelled request_id=%s", request_id)
            raise
        except Exception as error:
            code, message = public_error(error)
            logger.warning("Chat failed request_id=%s code=%s", request_id, code)
            yield {"type": "error", "code": code, "message": message, "request_id": request_id}

    async def close(self) -> None:
        if self.trace_client:
            await asyncio.to_thread(self.trace_client.flush)
            await asyncio.to_thread(self.trace_client.close)
