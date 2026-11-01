"""Provider token-count preflight for text-only Qwen requests."""

import json

PROTOCOL_MARGIN = 128  # OpenAI and Anthropic adapters differ slightly in framing.


class TokenCountUnavailable(RuntimeError):
    """A strict budget cannot be checked; generation must not proceed."""


class TokenBudgetViolation(RuntimeError):
    """Measured usage violated a successful preflight; stop without retrying."""

    def __init__(self, usage):
        super().__init__("模型实际用量超过模拟窗口，已停止本轮调用。")
        self.usage_metadata = usage


def count_payload(payload):
    """Convert the actual OpenAI packet to the provider's count_tokens schema."""
    messages, system = [], []
    for message in payload["messages"]:
        role = message["role"]
        content = message.get("content") or ""
        if isinstance(content, list):
            if any(block.get("type") != "text" for block in content):
                raise TokenCountUnavailable("8K 模拟暂只支持文本 Token 预检。")
            content = "\n".join(block.get("text", "") for block in content)
        if role in {"system", "developer"}:
            system.append(content)
            continue
        if role == "tool":
            converted = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": content,
                    }
                ],
            }
        elif message.get("tool_calls"):
            blocks = [{"type": "text", "text": content}] if content else []
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function["arguments"]
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": json.loads(arguments) if isinstance(arguments, str) else arguments,
                    }
                )
            converted = {"role": role, "content": blocks}
        else:
            converted = {"role": role, "content": content}
        messages.append(converted)
    result = {
        "model": payload["model"],
        "messages": messages,
        "thinking": {"type": "disabled"},
    }
    if system:
        result["system"] = "\n\n".join(system)
    if payload.get("tools"):
        result["tools"] = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"]["parameters"],
            }
            for tool in payload["tools"]
        ]
    return result
