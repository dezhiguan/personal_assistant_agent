"""One Qwen transport configuration shared by the assistant and evaluators."""

from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from personal_assistant_agent.core.config import Settings

PRICING_PROVIDER = "dashscope-beijing-nonthinking"


class QwenChatOpenAI(ChatOpenAI):
    """Keep real model identity while isolating Beijing pricing from other regions."""

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
    )
    if settings.context_window_tokens:
        model.profile = {
            **(model.profile or {}),
            "max_input_tokens": settings.context_window_tokens,
        }
    return model
