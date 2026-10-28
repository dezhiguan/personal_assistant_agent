"""A single Qwen-powered Deep Agent for text conversations."""

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from personal_assistant_agent.core.config import Settings

SYSTEM_PROMPT = """你是一位简洁、可靠的中文个人智能助理。
你的名称是“个人智能助理”。无论历史消息曾使用什么称呼，自我介绍始终使用这个名称。
帮助用户梳理计划、打磨文字、讨论想法。直接回答当前问题，默认使用中文。
结合用户提供的对话上下文作答。信息不足时明确说明，不编造事实或执行结果。
当前版本只支持文字对话，不能联网、访问文件、设置提醒或操作外部应用。
不声称已经执行实际上无法执行的操作。简单问题简短回答，复杂问题适当分条。
"""


def build_agent(settings: Settings) -> CompiledStateGraph:
    if not settings.model_configured:
        raise ValueError("请在服务端 .env 中配置 MODEL_NAME 和 DASHSCOPE_API_KEY。")
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.request_timeout,
        max_retries=1,
        max_tokens=2048,
        use_responses_api=False,
        extra_body={"enable_thinking": False},
    )
    register_harness_profile(
        f"openai:{settings.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            excluded_tools=frozenset(
                {
                    "ls",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "glob",
                    "grep",
                    "delete",
                    "execute",
                    "task",
                }
            ),
        ),
    )
    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        name="personal-assistant",
    )
