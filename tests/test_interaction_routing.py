"""Context-aware general routing uses the real model adapter with controlled transport."""

import asyncio
import json
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.interaction.routing import MessageRouter


def test_generic_context_classification_keeps_chat_and_task_scopes(monkeypatch):
    tid = str(uuid4())
    task = {
        "id": tid,
        "goal": "安排旅行",
        "status": "RUNNING",
        "constraints": {"目的地": "日本", "预算": "5000元", "时长": "一周"},
    }
    observed = []

    async def generate(self, messages, **kwargs):
        packet = json.loads(messages[-1].content)
        observed.append(packet)
        if packet["message"] == "预算改成三千，日期保持":
            result = {"intent": "STEER", "target_task_id": tid, "changes": {"预算": "3000元"}}
        else:
            result = {"intent": "CHAT"}
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=json.dumps(result)))]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)

    async def check():
        router = MessageRouter(Settings(api_key="synthetic-only"))
        history = [
            {"role": "user", "content": "红眼航班是什么"},
            {"role": "assistant", "content": "深夜起降的航班"},
        ]
        steer = await router.route("预算改成三千，日期保持", [task], history)
        assert steer["changes"] == {"预算": "3000元"}
        chat = await router.route("孩子六岁，容易晕车", [task], history)
        assert chat["intent"] == "CHAT"
        assert all(p["tasks"][0]["constraints"]["目的地"] == "日本" for p in observed)
        assert observed[-1]["history"] == history

    asyncio.run(check())


def test_ordinary_mention_is_not_a_condition_and_explicit_controls_need_no_model(monkeypatch):
    task = {
        "id": str(uuid4()),
        "goal": "搜索新闻",
        "status": "RUNNING",
        "constraints": {"范围": "全球"},
    }
    calls = []

    async def generate(self, messages, **kwargs):
        calls.append(messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content='{"intent":"CHAT"}'))]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)

    async def check():
        router = MessageRouter(Settings(api_key="synthetic-only"))
        assert (await router.route("科技公司也挺有意思的", [task], []))["intent"] == "CHAT"
        assert len(calls) == 1
        for text, expected in [
            ("暂停任务", "PAUSE"),
            ("恢复任务", "RESUME"),
            ("取消任务", "CANCEL"),
            ("所有任务现在怎么样", "STATUS"),
        ]:
            assert (await router.route(text, [task], []))["intent"] == expected
        assert len(calls) == 1

    asyncio.run(check())


def test_conflicting_news_mentions_require_semantic_merge(monkeypatch):
    task = {
        "id": str(uuid4()),
        "goal": "搜索新闻",
        "status": "RUNNING",
        "constraints": {"范围": "中国大陆", "时间": "最近三天", "主题": "科技"},
    }
    observed = []

    async def generate(self, messages, **kwargs):
        observed.append(json.loads(messages[-1].content))
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=json.dumps(
                            {
                                "intent": "STEER",
                                "target_task_id": task["id"],
                                "changes": {"范围": "全球"},
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr("langchain_openai.ChatOpenAI._agenerate", generate)

    async def check():
        router = MessageRouter(Settings(api_key="synthetic-only"))
        result = await router.route("不要国内，改为全球新闻", [task], [])
        assert result["changes"] == {"范围": "全球"}
        assert observed[0]["tasks"][0]["constraints"] == task["constraints"]

    asyncio.run(check())
