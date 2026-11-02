"""Project-specific test environment; execution stays in the real Agent Harness."""

import asyncio
import json
import os
import time
from dataclasses import fields, replace
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import get_current_run_tree
from pydantic import ValidationError

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.agents.tools import calculate, validate_time_budget
from personal_assistant_agent.agents.workspace import WorkspaceBackend
from personal_assistant_agent.evaluation.golden import Case
from personal_assistant_agent.schemas.chat import ChatRequest
from personal_assistant_agent.services.chat import ChatService

ROOT = Path(__file__).resolve().parents[3]


def fixture_tools(case):
    @tool
    def baidu_search(query: str, max_results: int = 5) -> dict:
        """Return synthetic search evidence; never contact a search engine in evaluation."""
        return case.search_results or {"source": "baidu", "error": "合成故障：检索服务不可用"}

    @tool
    def get_current_time(timezone: str = "Asia/Shanghai") -> dict:
        """Return the benchmark's fixed time; never infer the real current date."""
        return {
            "timezone": timezone,
            "date": "2026-10-30",
            "weekday": 5,
            "datetime": "2026-10-30T16:00:00+08:00",
            "week_start": "2026-10-26",
            "week_end": "2026-11-01",
        }

    return [baidu_search, get_current_time, calculate, validate_time_budget]


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
        serialized = json.dumps(record, ensure_ascii=False)
        for secret in secrets:
            if secret:
                serialized = serialized.replace(
                    json.dumps(secret, ensure_ascii=False)[1:-1], "[redacted]"
                )
        records.append(json.loads(serialized))
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

    def __init__(self, settings, prompt, output, *, timeout=180):
        self.settings, self.prompt, self.output, self.timeout = settings, prompt, output, timeout
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
                    build_agent, scoped, tools=fixture_tools(case), prompt=self.prompt
                )
                collector = RunCollectorCallbackHandler()
                secrets = [
                    getattr(scoped, field.name) for field in fields(scoped) if not field.repr
                ]
                service = ChatService(ObservedGraph(graph, collector), scoped)
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
