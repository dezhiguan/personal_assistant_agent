"""Load one immutable private LangSmith prompt version at application startup."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import SystemMessage
from langsmith import Client

from personal_assistant_agent.core.config import Settings

LOCAL_PROMPT = Path(__file__).parents[1] / "prompts" / "system.txt"


@dataclass(frozen=True)
class PromptVersion:
    text: str
    reference: str


def load_prompt(settings: Settings, client: Client | None = None) -> PromptVersion:
    if not settings.prompt_ref:
        text = LOCAL_PROMPT.read_text().strip()
        return PromptVersion(text, "local:" + hashlib.sha256(text.encode()).hexdigest())
    if not re.search(r":[0-9a-f]{8,64}$", settings.prompt_ref):
        raise ValueError("LANGSMITH_PROMPT_REF 必须固定到提交哈希，不能使用 latest。")
    if not settings.langsmith_api_key:
        raise ValueError("加载托管提示词需要 LANGSMITH_API_KEY。")
    owned = client is None
    client = client or Client(
        api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint
    )
    try:
        prompt = client.pull_prompt(settings.prompt_ref, include_model=False)
        if getattr(prompt, "input_variables", None):
            raise ValueError("系统提示词不能依赖用户输入变量。")
        messages = prompt.invoke({}).to_messages()
        if len(messages) != 1 or not isinstance(messages[0], SystemMessage):
            raise ValueError("托管提示词必须只包含一条 system 消息。")
        text = messages[0].content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("托管系统提示词不能为空。")
        return PromptVersion(text, settings.prompt_ref)
    finally:
        if owned:
            client.close()
