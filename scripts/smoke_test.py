"""Verify the running app with real Qwen responses and, when enabled, real traces."""

import argparse
import json
import time
from uuid import uuid4

import httpx
import truststore
from langsmith import Client

from personal_assistant_agent.core.config import get_settings


def chat(client: httpx.Client, base_url: str, conversation_id: str, messages: list) -> tuple:
    answer = ""
    request_id = None
    done = False
    with client.stream(
        "POST",
        base_url + "/api/chat",
        json={"conversation_id": conversation_id, "messages": messages},
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(f"聊天接口返回 HTTP {response.status_code}")
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "meta":
                request_id = event["request_id"]
            elif event["type"] == "delta":
                answer += event["text"]
            elif event["type"] == "error":
                raise RuntimeError(event["message"])
            elif event["type"] == "done":
                done = True
    if not done or not answer:
        raise RuntimeError("未收到完整的模型回复")
    return answer, request_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    truststore.inject_into_ssl()
    settings = get_settings()
    try:
        with httpx.Client(timeout=110) as client:
            status = client.get(args.url + "/healthz").raise_for_status().json()
            if not status["model_configured"]:
                raise RuntimeError("请先配置 DASHSCOPE_API_KEY 并重启服务。")
            conversation_id = str(uuid4())
            messages = [{"role": "user", "content": "请记住本次测试代号是青竹。仅回复收到。"}]
            first, _ = chat(client, args.url, conversation_id, messages)
            messages += [
                {"role": "assistant", "content": first},
                {"role": "user", "content": "刚才的测试代号是什么？只回答代号。"},
            ]
            second, request_id = chat(client, args.url, conversation_id, messages)
            if "青竹" not in second:
                raise RuntimeError("模型未能结合第一轮上下文回答")
            print(f"PASS: 真实 Qwen 两轮对话与上下文，request_id={request_id}")
            if status["tracing_enabled"]:
                trace_client = Client(
                    api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint
                )
                try:
                    project = next(
                        trace_client.list_projects(name=settings.langsmith_project, limit=1)
                    )
                    for _ in range(10):
                        response = client.get(
                            settings.langsmith_endpoint + "/api/v2/runs/" + request_id,
                            params={"project_id": str(project.id), "selects": ["ID", "NAME"]},
                            headers={"x-api-key": settings.langsmith_api_key},
                        )
                        if response.is_success and response.json().get("id") == request_id:
                            print("PASS: LangSmith 中查到本次真实调用记录")
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError("真实回复成功，但 LangSmith 追踪记录尚未查到")
                finally:
                    trace_client.close()
    except Exception as error:
        if isinstance(error, RuntimeError):
            raise SystemExit(str(error)) from None
        raise SystemExit(f"验证失败：{type(error).__name__}") from None


if __name__ == "__main__":
    main()
