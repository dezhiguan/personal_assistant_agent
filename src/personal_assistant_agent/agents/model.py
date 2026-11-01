"""One Qwen transport configuration shared by the assistant and evaluators."""

import hashlib
import json
import time
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx
from langchain_core.exceptions import ContextOverflowError
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr

from personal_assistant_agent.agents.token_budget import (
    PROTOCOL_MARGIN,
    TokenBudgetViolation,
    TokenCountUnavailable,
    count_payload,
)
from personal_assistant_agent.core.config import Settings

PRICING_PROVIDER = "dashscope-beijing-nonthinking"


class QwenChatOpenAI(ChatOpenAI):
    """Keep real model identity while isolating Beijing pricing from other regions."""

    context_window_tokens: int = Field(default=0, exclude=True)
    _token_counts: dict = PrivateAttr(default_factory=dict)

    def _count_request(self, messages, **kwargs):
        return count_payload(self._get_request_payload(messages, **kwargs))

    def _count_url(self):
        url = urlsplit(str(self.openai_api_base))
        if not url.path.rstrip("/").endswith("/compatible-mode/v1"):
            raise TokenCountUnavailable("当前模型地址不支持严格 Token 预检。")
        path = url.path.rstrip("/")[: -len("/compatible-mode/v1")]
        return urlunsplit(
            (url.scheme, url.netloc, path + "/apps/anthropic/v1/messages/count_tokens", "", "")
        )

    def _cached_count(self, payload):
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        entry = self._token_counts.get(key)
        return key, entry[1] if entry and time.monotonic() - entry[0] < 60 else None

    def _remember_count(self, key, response):
        try:
            response.raise_for_status()
            count = response.json()["input_tokens"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Invalid token count")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise TokenCountUnavailable("百炼 Token 预检失败，本轮未继续生成。") from None
        if len(self._token_counts) >= 256:
            self._token_counts.clear()
        self._token_counts[key] = (time.monotonic(), count)
        return count

    def count_input_tokens(self, messages, **kwargs):
        payload = self._count_request(messages, **kwargs)
        key, cached = self._cached_count(payload)
        if cached is not None:
            return cached
        try:
            response = self.root_client._client.post(
                self._count_url(),
                json=payload,
                timeout=15,
                headers={
                    "x-api-key": self.openai_api_key.get_secret_value(),
                    "anthropic-version": "2023-06-01",
                },
            )
        except httpx.HTTPError:
            raise TokenCountUnavailable("无法连接百炼 Token 预检接口，本轮未继续生成。") from None
        return self._remember_count(key, response)

    async def acount_input_tokens(self, messages, **kwargs):
        payload = self._count_request(messages, **kwargs)
        key, cached = self._cached_count(payload)
        if cached is not None:
            return cached
        try:
            response = await self.root_async_client._client.post(
                self._count_url(),
                json=payload,
                timeout=15,
                headers={
                    "x-api-key": self.openai_api_key.get_secret_value(),
                    "anthropic-version": "2023-06-01",
                },
            )
        except httpx.HTTPError:
            raise TokenCountUnavailable("无法连接百炼 Token 预检接口，本轮未继续生成。") from None
        return self._remember_count(key, response)

    def _check_preflight(self, count, messages, **kwargs):
        payload = self._get_request_payload(messages, **kwargs)
        output = payload.get("max_completion_tokens", payload.get("max_tokens", self.max_tokens))
        if not isinstance(output, int) or output <= 0:
            raise TokenCountUnavailable("模拟窗口要求明确设置单次输出上限。")
        if count + output + PROTOCOL_MARGIN > self.context_window_tokens:
            raise ContextOverflowError("输入、输出预留与协议余量之和超过模拟窗口。")

    def _preflight(self, messages, **kwargs):
        if self.context_window_tokens:
            self._check_preflight(self.count_input_tokens(messages, **kwargs), messages, **kwargs)

    async def _apreflight(self, messages, **kwargs):
        if self.context_window_tokens:
            self._check_preflight(
                await self.acount_input_tokens(messages, **kwargs), messages, **kwargs
            )

    def _check_measured(self, message):
        usage = message.usage_metadata
        if (
            self.context_window_tokens
            and usage
            and usage["total_tokens"] > self.context_window_tokens
        ):
            raise TokenBudgetViolation(usage)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._preflight(messages, stop=stop, **kwargs)
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations:
            self._check_measured(generation.message)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await self._apreflight(messages, stop=stop, **kwargs)
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations:
            self._check_measured(generation.message)
        return result

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._preflight(messages, stop=stop, **kwargs)
        for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            self._check_measured(chunk.message)
            yield chunk

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        await self._apreflight(messages, stop=stop, **kwargs)
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            self._check_measured(chunk.message)
            yield chunk

    def _get_ls_params(self, stop=None, **kwargs):
        params = super()._get_ls_params(stop=stop, **kwargs)
        if urlparse(str(self.openai_api_base)).hostname == "dashscope.aliyuncs.com":
            params["ls_provider"] = PRICING_PROVIDER
        return params


def build_model(settings: Settings, *, max_tokens: int = 2048) -> ChatOpenAI:
    if not settings.model_configured:
        raise ValueError("请在服务端 .env 中配置 MODEL_NAME 和 DASHSCOPE_API_KEY。")
    if settings.context_window_tokens:
        if settings.context_window_tokens < 4096:
            raise ValueError("MODEL_CONTEXT_WINDOW_TOKENS 必须为0（关闭）或至少4096。")
        if not 256 <= settings.context_output_reserve_tokens <= settings.context_window_tokens // 4:
            raise ValueError("上下文输出预留须至少256 Token，且不超过窗口的25%。")
        max_tokens = min(max_tokens, settings.context_output_reserve_tokens)
    model = QwenChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.request_timeout,
        max_retries=1,
        max_tokens=max_tokens,
        use_responses_api=False,
        stream_usage=True,
        extra_body={"enable_thinking": False},
        context_window_tokens=settings.context_window_tokens,
    )
    if settings.context_window_tokens:
        model.profile = {
            **(model.profile or {}),
            "max_input_tokens": settings.context_window_tokens,
        }
    return model
