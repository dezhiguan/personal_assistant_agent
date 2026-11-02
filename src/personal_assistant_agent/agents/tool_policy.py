"""Choose tool access from the current request, without another model call."""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

READ_ONLY = {"ls", "glob", "grep", "read_file", "baidu_search"}
MUTATING = {"write_file", "edit_file", "delete", "execute", "task"}


def latest_user(messages):
    return next(
        (
            m
            for m in reversed(messages)
            if isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") != "summarization"
        ),
        None,
    )


def standalone_question(text):
    """Only complete, history-independent greetings and identity questions qualify."""
    value = re.sub(r"[\s，,。.!！?？：:]+", "", text).lower()
    return bool(
        re.fullmatch(
            r"(?:请|请问)?(?:你是谁|您是谁|你叫什么(?:名字)?|您的名字是什么|你的名字是什么|"
            r"介绍(?:一下|下)?你自己|你好|您好|谢谢(?:你)?|多谢|hello|hi|whoareyou|"
            r"whatisyourname|what'syourname|introduceyourself)"
            r"(?:(?:请)?只(?:回答|回复|输出)(?:你的)?(?:名字|名称|身份|一句话)(?:即可)?)?",
            value,
        )
    )


def _turn_messages(messages):
    user = latest_user(messages)
    start = next((i for i, m in enumerate(messages) if m is user), 0)
    return messages[start:]


def _signature(call):
    return call["name"], json.dumps(call.get("args", {}), sort_keys=True, ensure_ascii=False)


def _completed_reads(messages):
    calls, reads = {}, []
    for message in _turn_messages(messages):
        if isinstance(message, AIMessage):
            if any(c["name"] in MUTATING for c in message.tool_calls):
                reads.clear()
            calls.update({c["id"]: c for c in message.tool_calls})
        elif isinstance(message, ToolMessage):
            call = calls.get(message.tool_call_id)
            if not call:
                continue
            if call["name"] in MUTATING:
                reads.clear()
            elif call["name"] in READ_ONLY and message.status == "success":
                reads.append((_signature(call), message))
    return reads


def repeated_read(messages):
    reads = _completed_reads(messages)
    return len(reads) >= 2 and reads[-1][0] == reads[-2][0]


def advice_only(text):
    return bool(
        re.search(
            r"(?:未经.{0,10}确认|确认之前|先.{0,12}建议|只.{0,6}建议)"
            r".{0,40}(?:不要|禁止|不).{0,8}(?:修改|删除|写入)",
            text,
        )
    )


def authorized_tools(messages, tools):
    """Apply explicit user authorization independently of token-window settings."""
    user = latest_user(messages)
    if not user or not advice_only(str(user.content)):
        return tools
    return [t for t in tools if tool_name(t) not in MUTATING]


def selected_tools(messages, tools):
    user = latest_user(messages)
    text = str(user.content) if user else ""
    if standalone_question(text) or re.search(r"(?:不|不要|禁止|无需)调用(?:任何)?工具", text):
        return []
    if repeated_read(messages):
        return []
    return authorized_tools(messages, tools)


def tool_name(tool):
    return tool.name if hasattr(tool, "name") else tool.get("function", tool).get("name", "")


def cached_read(messages, call):
    if call["name"] not in READ_ONLY:
        return None
    key = _signature(call)
    return next(
        (
            result
            for signature, result in reversed(_completed_reads(messages))
            if signature == key and result.tool_call_id != call["id"]
        ),
        None,
    )
