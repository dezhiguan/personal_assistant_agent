"""Server-side Qwen and LangSmith settings; credentials are excluded from repr."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model_name: str = "qwen-plus"
    api_key: str = field(default="", repr=False)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    request_timeout: float = 90
    tracing_enabled: bool = False
    langsmith_api_key: str = field(default="", repr=False)
    langsmith_project: str = "personal-assistant-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @property
    def model_configured(self) -> bool:
        return bool(self.model_name and self.api_key.strip())

    @property
    def tracing_configured(self) -> bool:
        return self.tracing_enabled and bool(self.langsmith_api_key.strip())


def get_settings() -> Settings:
    load_dotenv(Path(os.getenv("ASSISTANT_ENV_FILE", ".env")), override=False)
    return Settings(
        model_name=os.getenv("MODEL_NAME", "qwen-plus").strip(),
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip(),
        request_timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "90")),
        tracing_enabled=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "personal-assistant-agent"),
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )
