"""Stream Deep Agents output with native LangSmith tracing."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, tracing_context
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)


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
        callbacks = (
            [
                LangChainTracer(
                    client=self.trace_client, project_name=self.settings.langsmith_project
                )
            ]
            if self.trace_client
            else []
        )
        config = {
            "run_id": UUID(request_id),
            "run_name": "personal-assistant-chat",
            "recursion_limit": 12,
            "callbacks": callbacks,
            "metadata": {
                "conversation_id": str(payload.conversation_id),
                "session_id": str(payload.conversation_id),
                "request_id": request_id,
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
        try:
            with tracing_context(enabled=self.settings.tracing_configured):
                async with asyncio.timeout(self.settings.request_timeout + 10):
                    async for chunk, metadata in self.agent.astream(
                        {"messages": [m.model_dump() for m in payload.messages]},
                        config=config,
                        stream_mode="messages",
                    ):
                        if metadata.get("langgraph_node") != "model":
                            continue
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
                            yield {"type": "delta", "text": text}
            if not received_text:
                raise RuntimeError("Model completed without text")
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
