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
    prompt_ref: str = ""
    workspace_dir: str = "workspace"
    baidu_timeout: float = 10
    baidu_proxy: str = field(default="", repr=False)
    tool_timeout: float = 15
    amap_api_key: str = field(default="", repr=False)
    qweather_host: str = field(default="", repr=False)
    qweather_api_key: str = field(default="", repr=False)
    travel_provider: str = "amadeus"
    tuniu_api_key: str = field(default="", repr=False)
    tuniu_mcp_base_url: str = "https://openapi.tuniu.cn/mcp"
    amadeus_client_id: str = field(default="", repr=False)
    amadeus_client_secret: str = field(default="", repr=False)
    amadeus_environment: str = "test"
    travel_private_dir: str = ".assistant-private"

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
        prompt_ref=os.getenv("LANGSMITH_PROMPT_REF", "").strip(),
        workspace_dir=os.getenv("ASSISTANT_WORKSPACE_DIR", "workspace").strip(),
        baidu_timeout=float(os.getenv("BAIDU_TIMEOUT_SECONDS", "10")),
        baidu_proxy=os.getenv("BAIDU_HTTP_PROXY", "").strip(),
        tool_timeout=float(os.getenv("TOOL_TIMEOUT_SECONDS", "15")),
        amap_api_key=os.getenv("AMAP_API_KEY", ""),
        qweather_host=os.getenv("QWEATHER_API_HOST", "").strip(),
        qweather_api_key=os.getenv("QWEATHER_API_KEY", ""),
        travel_provider=os.getenv("TRAVEL_PROVIDER", "amadeus").strip(),
        tuniu_api_key=os.getenv("TUNIU_API_KEY", ""),
        tuniu_mcp_base_url=os.getenv("TUNIU_MCP_BASE_URL", "https://openapi.tuniu.cn/mcp").strip(),
        amadeus_client_id=os.getenv("AMADEUS_CLIENT_ID", ""),
        amadeus_client_secret=os.getenv("AMADEUS_CLIENT_SECRET", ""),
        amadeus_environment=os.getenv("AMADEUS_ENVIRONMENT", "test").strip(),
        travel_private_dir=os.getenv("TRAVEL_PRIVATE_DIR", ".assistant-private").strip(),
    )
