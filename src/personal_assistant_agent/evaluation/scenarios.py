"""Project-specific test environment; execution stays in the real Agent Harness."""

import asyncio
import json
import os
import time
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import get_current_run_tree
from pydantic import ValidationError

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.tools import (
    build_baidu_search,
    calculate,
    get_current_time,
    validate_time_budget,
)
from personal_assistant_agent.agents.workspace import WorkspaceBackend
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.evaluation.golden import Case
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService

ROOT = Path(__file__).resolve().parents[3]


def fixture_tools(case, settings=None):
    async def search(query: str, max_results: int = 5) -> dict:
        return case.search_results or {"source": "baidu", "error": "合成故障：检索服务不可用"}

    def frozen_time(timezone: str = "Asia/Shanghai") -> dict:
        try:
            now = datetime(2026, 10, 30, 8, tzinfo=UTC).astimezone(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError):
            return {"error": "无效时区，请使用 Asia/Shanghai、UTC 等 IANA 时区名称。"}
        monday = now.date() - timedelta(days=now.weekday())
        return {
            "timezone": timezone,
            "date": now.date().isoformat(),
            "weekday": now.isoweekday(),
            "datetime": now.isoformat(timespec="seconds"),
            "week_start": monday.isoformat(),
            "week_end": (monday + timedelta(days=6)).isoformat(),
        }

    return [
        build_baidu_search(settings or Settings(), search=search),
        get_current_time.model_copy(update={"func": frozen_time}),
        calculate,
        validate_time_budget,
    ]


def boundary(case, directory):
    backend = WorkspaceBackend(str(directory.relative_to(ROOT)))
    if case.operation == "execute":
        return [] if backend.execute(case.argument).exit_code != 0 else ["unsafe_command_accepted"]
    if case.operation in {"symlink", "hardlink"}:
        outside = directory.parent / (directory.name + "-outside.txt")
        outside.write_text("SYNTHETIC-OUTSIDE-CANARY")
        if case.operation == "symlink":
            (directory / "link.txt").symlink_to(outside)
        else:
            os.link(outside, directory / "link.txt")
        argument = "/link.txt"
    else:
        argument = case.argument
    try:
        result = backend.read(argument)
        return [] if result.error else ["outside_read_accepted"]
    except (PermissionError, ValueError):
        return []


def redacted_record(record, secrets):
    serialized = json.dumps(record, ensure_ascii=False)
    for secret in secrets:
        if secret:
            serialized = serialized.replace(
                json.dumps(secret, ensure_ascii=False)[1:-1], "[redacted]"
            )
    return json.loads(serialized)


def graph_tool_receipts(runs, secrets=()):
    """Committed ToolMessages include middleware returns without a tool span."""
    receipts = []
    for run in runs:
        messages = (run.outputs or {}).get("messages", [])
        calls = {
            call["id"]: call
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            receipts.append(
                redacted_record(
                    {
                        "name": message.name,
                        "tool_call_id": message.tool_call_id,
                        "inputs": calls.get(message.tool_call_id, {}).get("args"),
                        "content": message.content,
                        "status": message.status,
                        "cached": bool(message.additional_kwargs.get("assistant_cached_tool")),
                    },
                    secrets,
                )
            )
    return receipts


def tool_receipts(runs, secrets=()):
    """Use native trace records, retaining actual outcomes and removing credential values."""
    pending, seen, records = list(runs), set(), []
    while pending:
        run = pending.pop()
        if run.id in seen:
            continue
        seen.add(run.id)
        pending.extend(run.child_runs or [])
        if run.run_type != "tool":
            continue
        record = run.model_dump(
            mode="json", include={"id", "name", "inputs", "outputs", "start_time"}
        )
        record["error"] = bool(run.error)
        records.append(redacted_record(record, secrets))
    return sorted(records, key=lambda record: record["start_time"])


class ObservedGraph:
    def __init__(self, graph, collector):
        self.graph, self.collector = graph, collector

    def astream(self, value, *, config, **kwargs):
        return self.graph.astream(
            value,
            config={**config, "callbacks": [*config.get("callbacks", []), self.collector]},
            **kwargs,
        )


class ScenarioTarget:
    """One SDK target invocation runs one isolated scenario, including all its turns."""

    def __init__(self, settings, prompt, output, *, timeout=180, trace_client=None):
        self.settings, self.prompt, self.output, self.timeout = settings, prompt, output, timeout
        self.trace_client = trace_client
        self.repetitions = {}

    async def __call__(self, inputs):
        case = Case.model_validate(inputs["case"])
        repeat = self.repetitions.get(case.id, 0) + 1
        self.repetitions[case.id] = repeat
        directory = self.output / "workspaces" / f"{case.id}-{repeat}"
        directory.mkdir(parents=True, mode=0o700)
        started = time.monotonic()
        result = {
            "id": case.id,
            "repeat": repeat,
            "workspace": str(directory),
            "turns": [],
            "engine_failures": [],
            "error": None,
        }
        service = None
        try:
            for name, content in case.fixtures.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            if case.engine == "schema":
                try:
                    ChatRequest.model_validate({"conversation_id": str(uuid4()), **case.request})
                    result["engine_failures"].append("invalid_request_accepted")
                except ValidationError:
                    pass
            elif case.engine == "boundary":
                result["engine_failures"] = boundary(case, directory)
            else:
                tree = get_current_run_tree()
                scoped = replace(
                    self.settings,
                    workspace_dir=str(directory.relative_to(ROOT)),
                    langsmith_project=tree.session_name
                    if tree
                    else self.settings.langsmith_project,
                )
                graph = await asyncio.to_thread(
                    build_agent, scoped, tools=fixture_tools(case, scoped), prompt=self.prompt
                )
                collector = RunCollectorCallbackHandler()
                secrets = [
                    getattr(scoped, field.name) for field in fields(scoped) if not field.repr
                ]
                service = ChatService(
                    ObservedGraph(graph, collector), scoped, trace_client=self.trace_client
                )
                history, todos, conversation = [], case.todos, uuid4()
                for turn in case.turns:
                    collector.traced_runs.clear()
                    payload = ChatRequest(
                        conversation_id=conversation,
                        messages=[*history, {"role": "user", "content": turn.prompt}],
                        todos=todos,
                    )
                    async with asyncio.timeout(self.timeout):
                        events = [e async for e in service.stream(payload, str(uuid4()))]
                    answer = "".join(e["text"] for e in events if e["type"] == "delta")
                    plans = [e["todos"] for e in events if e["type"] == "plan"]
                    todos = plans[-1] if plans else todos
                    errors = [
                        e.get("code", "service_error") for e in events if e["type"] == "error"
                    ]
                    receipts = tool_receipts(collector.traced_runs, secrets)
                    result["turns"].append(
                        {
                            "answer": answer,
                            "todos": todos,
                            "tools": [receipt["name"] for receipt in receipts],
                            "tool_receipts": receipts,
                            "graph_tool_receipts": graph_tool_receipts(
                                collector.traced_runs, secrets
                            ),
                            "done": any(e["type"] == "done" for e in events),
                            "errors": errors,
                        }
                    )
                    history = [m.model_dump() for m in payload.messages] + [
                        {"role": "assistant", "content": answer or "（回复失败）"}
                    ]
                    if errors:
                        result["error"] = "service_error"
                        break
        except Exception as error:
            # A provider's exception text can contain credentials. Record only its type.
            result["error"] = type(error).__name__
        finally:
            if service:
                await service.close()
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        return result
