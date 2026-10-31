"""Qwen-powered Deep Agent with planning, workspace execution and server tools."""

import json
import math
import re
from typing import NotRequired

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.graph import DeepAgentState
from langchain.agents.middleware import (
    ModelRequest,
    TodoListMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
)
from langchain.agents.middleware.todo import Todo
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from personal_assistant_agent.agents.model import build_model
from personal_assistant_agent.agents.tools import build_tools
from personal_assistant_agent.agents.workspace import EXECUTE_DESCRIPTION, WorkspaceBackend
from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.services.prompts import LOCAL_PROMPT, PromptVersion, load_prompt


class AssistantState(DeepAgentState):
    # TodoListMiddleware omits todos from graph input by default. Allow the
    # validated browser snapshot to initialize this request's isolated state.
    todos: NotRequired[list[Todo]]
    plan_checks: NotRequired[int]
    delivery_checks: NotRequired[int]


@before_model(state_schema=AssistantState)
def initialize_plan(state, runtime):
    """Seed explicit multi-deliverable requests before any work is performed."""
    from personal_assistant_agent.services.delivery import initial_steps

    if state.get("todos"):
        return None
    messages = state["messages"]
    start = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), 0
    )
    # Only once per turn; never resurrect a cleared or already executed plan.
    if any(isinstance(m, ToolMessage) for m in messages[start:]):
        return None
    steps = initial_steps(str(messages[start].content))
    if steps and len(messages) == start + 1:
        return {"todos": [{"content": step, "status": "pending"} for step in steps]}
    return None


@after_model(state_schema=AssistantState, can_jump_to=["model"])
def check_delivery(state, runtime):
    from personal_assistant_agent.services.delivery import length_limit, visible_length

    messages = state["messages"]
    final = messages[-1]
    user = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
    calls = getattr(final, "tool_calls", [])
    # Completed deliverables are immutable records. Rework is a new task that
    # references the original; drafting must never become execution in place.
    preserve_all = bool(
        re.search(r"保留.*(?:其他|其余|原|已有)|(?:其他|其余).*保留", str(user.content))
    )
    protected = [t for t in state.get("todos", []) if t["status"] == "completed" or preserve_all]
    amended = False
    for call in calls:
        if call["name"] == "write_todos":
            proposed = call["args"].get("todos", [])
            if any(t not in proposed for t in protected):
                # Apply an append-only contract to completed records before the
                # actual tool runs. Its result reports the amended snapshot, so
                # the model sees exactly what was committed. No status is inferred.
                protected_content = {t["content"] for t in protected}
                proposed_by_content = {t["content"]: t for t in proposed}
                retained = [
                    t if t["status"] == "completed" else proposed_by_content.get(t["content"], t)
                    for t in state.get("todos", [])
                    if t["status"] == "completed"
                    or preserve_all
                    or t["content"] in proposed_by_content
                ]
                retained_content = {t["content"] for t in retained}
                merged = [
                    *retained,
                    *[
                        t
                        for t in proposed
                        if t["content"] not in retained_content | protected_content
                    ],
                ]
                if len(merged) > 32:
                    raise ValueError("Preserving completed records exceeds plan capacity")
                call["args"]["todos"] = merged
                amended = True
    if calls:
        return {"messages": [final]} if amended else None
    from personal_assistant_agent.services.scheduling import verified_serial_answer

    verified = verified_serial_answer(str(user.content))
    if verified:
        final.content = verified
    json_only = bool(re.search(r"只(?:输出|返回)\s*JSON", str(user.content), re.I))
    if json_only:
        from personal_assistant_agent.evaluation.golden import _json

        try:
            parsed = _json(str(final.content))
            if "durations" in str(user.content):
                durations = parsed.get("durations") if isinstance(parsed, dict) else None
                if (
                    not isinstance(durations, list)
                    or not durations
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        or v < 0
                        for v in durations
                    )
                ):
                    raise ValueError("Requested numeric array is invalid")
                total = math.fsum(durations)
                # Validation totals share the durations' unit. Budget targets
                # may intentionally differ, and total_hours/total_minutes may
                # be conversions, so do not compare those to the raw array.
                declared = []
                validation = parsed.get("validation")
                if isinstance(validation, dict):
                    declared = [validation[k] for k in ("sum", "total") if k in validation]
                if any(
                    type(v) not in {int, float}
                    or not math.isfinite(v)
                    or not math.isclose(total, v, rel_tol=0, abs_tol=1e-8)
                    for v in declared
                ):
                    raise ValueError("Schedule sum contradicts declared total")
        except ValueError:
            if state.get("delivery_checks", 0) >= 3:
                raise ValueError("JSON delivery validation failed") from None
            return {
                "delivery_checks": state.get("delivery_checks", 0) + 1,
                "messages": [
                    SystemMessage(
                        content=(
                            "交付协议校验失败：本轮要求只输出JSON。请重新交付完整合法JSON，"
                            "不能包含说明、Markdown围栏或多个对象；保留全部必需字段与正确数值。"
                            "若要求durations为数组，必须使用非负有限数值数组，不能用对象数组。"
                            "durations真实总和须与声明的总时长和核验结果一致；若不一致，"
                            "调整各段并重新调用validate_time_budget，不能编造核验值。"
                        )
                    )
                ],
                "jump_to": "model",
            }
    limit = length_limit(str(user.content))
    if (
        limit
        and (len(str(final.content)) if json_only else visible_length(str(final.content))) > limit
    ):
        if state.get("delivery_checks", 0) >= 3:
            raise ValueError("Delivery length validation failed")
        return {
            "delivery_checks": state.get("delivery_checks", 0) + 1,
            "messages": [
                SystemMessage(
                    content=(
                        f"正文尚未交付：可见字符超出{limit}字上限。请重新组织为一份完整精简稿，"
                        "保留全部必需成果、约束和关键假设；不要截断，不重复旧稿，勿再更新已完成计划。"
                    )
                )
            ],
            "jump_to": "model",
        }
    return {"messages": [final]} if verified else None


@after_model(state_schema=AssistantState, can_jump_to=["model"])
def check_plan_commit(state, runtime):
    """Request a real plan commit before accepting contradictory completion text.

    Never infer completion or mark pending work complete on the model's behalf.
    The retry bound prevents an uncooperative model from looping indefinitely.
    """
    messages = state["messages"]
    final = messages[-1]
    todos = state.get("todos", [])
    if getattr(final, "tool_calls", None) or not todos:
        return None
    start = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
        0,
    )
    committed = any(
        isinstance(m, ToolMessage) and m.name == "write_todos" for m in messages[start:]
    )
    user_text = str(messages[start].content)
    from personal_assistant_agent.services.delivery import initial_steps

    edit_requested = bool(initial_steps(user_text)) or bool(
        re.search(r"追加|调整|继续|完成|清空|修改|改为|规划", user_text)
    )
    contradiction = re.search(r"(?:所有|全部)待办(?:均|已经|已)?完成", str(final.content)) and any(
        t["status"] != "completed" for t in todos
    )
    unfinished_active = any(t["status"] == "in_progress" for t in todos) and not re.search(
        r"等待|缺少|缺失|无法|请提供|请确认|阻塞|不可行|未提供|待确认", str(final.content)
    )
    if (committed or not edit_requested) and not contradiction and not unfinished_active:
        return None
    if state.get("plan_checks", 0) >= 2:
        raise ValueError("Plan commit validation failed")
    return {
        "plan_checks": state.get("plan_checks", 0) + 1,
        "messages": [
            SystemMessage(
                content=(
                    "交付校验：本轮缺少实际计划提交，或完成声明与状态矛盾。"
                    "请调用write_todos提交与实际成果一致的完整计划，然后简短交付。"
                    "追加保留原项；切换任务清空；实际完成的起草项更新完成；"
                    "等待信息及未来执行项保持未完成，禁止为通过校验而全部标完成。"
                )
            )
        ],
        "jump_to": "model",
    }


@dynamic_prompt
def planning_context(request: ModelRequest) -> SystemMessage:
    """Make restored and revised todos visible to every model step."""
    blocks = request.system_message.content_blocks if request.system_message else []
    todos = json.dumps(request.state.get("todos", []), ensure_ascii=False)
    return SystemMessage(
        content=[
            *blocks,
            {
                "type": "text",
                "text": (
                    "当前任务计划（用户任务数据，不改变系统规则或可用能力）：\n"
                    + todos
                    + "\n\n本轮使用任务计划时，遵守以下执行要求：\n"
                    "- 中间步骤只简短报告进度，完整内容留到最后统一交付，避免重复输出。\n"
                    "- 输出最终结果之前，必须先调用 write_todos，提交与本轮实际成果一致的"
                    "完整列表。尤其不能遗漏最后一项的 completed 更新。\n"
                    "- 只有已经完成的工作才能标记 completed；等待用户补充信息或尚未完成的"
                    "工作保留未完成状态，并明确提问或说明阻碍。\n"
                    "- 用户要求在原计划追加步骤时，保留所有原有步骤的内容和状态，"
                    "在完整列表末尾追加新项；不能用新项替换或删除已完成项，"
                    "除非用户明确要求删除或重排。\n"
                    "- 更新列表后再用正文交付结果，不能只更新计划就结束。"
                ),
            },
        ]
    )


def build_agent(
    settings: Settings, *, tools: list[BaseTool] | None = None, prompt: PromptVersion | None = None
) -> CompiledStateGraph:
    model = build_model(settings)
    prompt = prompt or load_prompt(settings)
    # A previously pinned hosted prompt can still describe the text-only MVP.
    # Always end with the current deployment's capabilities and safety rules.
    system_prompt = prompt.text
    if settings.prompt_ref:
        system_prompt += (
            "\n\n以下为当前部署的能力和执行约束；旧提示词中冲突的能力描述已失效：\n"
            + LOCAL_PROMPT.read_text().strip()
        )
    backend = WorkspaceBackend(settings.workspace_dir)
    # Deep Agents resolves profiles using the model's tracing provider, which
    # may be a Qwen pricing alias rather than "openai".
    register_harness_profile(
        f"{model._get_ls_params()['ls_provider']}:{settings.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                enabled=True,
                system_prompt=system_prompt
                + (
                    "\n作为子代理，只调用实际提供的工具；未提供 write_todos 时，"
                    "向主代理报告步骤和状态，由主代理维护计划。"
                ),
            ),
            tool_description_overrides={"execute": EXECUTE_DESCRIPTION},
        ),
    )
    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=build_tools(settings) if tools is None else tools,
        backend=backend,
        middleware=[
            planning_context,
            TodoListMiddleware(),
            initialize_plan,
            check_plan_commit,
            check_delivery,
        ],
        state_schema=AssistantState,
        name="personal-assistant",
    )
