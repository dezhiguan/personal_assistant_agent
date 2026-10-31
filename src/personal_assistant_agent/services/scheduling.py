"""Validate every allocation, including fixed work, capacity and precedence."""

import math
import re


def validate_schedule(tasks, days, capacity, *, resources=1, allow_split=False):
    errors = []
    totals = {name: 0.0 for name, _ in tasks}
    positions = {name: [] for name, _ in tasks}
    order = [name for name, _ in tasks]
    if resources != 1:
        errors.append("资源约束：仅允许单人")
    for day, entries in enumerate(days):
        used = 0.0
        for index, (name, hours) in enumerate(entries):
            if name not in totals or not math.isfinite(hours) or hours <= 0:
                errors.append("未知任务或非法工时")
                continue
            used += hours
            totals[name] += hours
            positions[name].append((day, index))
        if used > capacity + 1e-9:
            errors.append("每日容量超限")
    for name, duration in tasks:
        if not math.isclose(totals[name], duration, abs_tol=1e-9):
            errors.append("固定工时不符：" + name)
        if len(positions[name]) > 1 and not allow_split:
            errors.append("不允许跨天暂停：" + name)
    for a, b in zip(order, order[1:]):
        if positions[a] and positions[b] and max(positions[a]) >= min(positions[b]):
            errors.append("违反依赖：" + a + "→" + b)
    return errors


def serial_contract(text):
    """Recognize explicit fixed single-person serial feasibility questions only."""
    if not re.search(r"单人|一个人|1人|一名工程师|一个工程师", text) or not re.search(
        r"依赖|串行|完成后开始", text
    ):
        return None
    capacity = re.search(r"每天(?:最多|上限|不超过|工作)?\s*(\d+(?:\.\d+)?)\s*小时", text)
    tasks = re.findall(r"([A-Z])\s*[^A-Z\d\n→]{0,12}?(\d+(?:\.\d+)?)\s*小时", text)
    if not capacity or len(tasks) < 2 or len(set(n for n, _ in tasks)) != len(tasks):
        return None
    capacity = float(capacity[1])
    tasks = [(name, float(duration)) for name, duration in tasks]
    if capacity <= 0 or any(d <= 0 for _, d in tasks):
        return None
    return tasks, capacity


def verified_serial_answer(text):
    if re.search(r"JSON", text, re.I):
        return None  # JSON delivery has its own strict protocol validator.
    contract = serial_contract(text)
    if not contract:
        return None
    tasks, capacity = contract
    total = math.fsum(d for _, d in tasks)
    parts = [f"固定串行工时共{total:g}小时，每日上限{capacity:g}小时。"]
    parts.append(
        "当天可行。" if total <= capacity else f"当天不可行，缺口{total - capacity:g}小时。"
    )
    split = not bool(re.search(r"(?:不能|不允许|禁止|不可)跨天(?:暂停|拆分)?", text))
    days = [[]]
    used = 0.0
    for name, duration in tasks:
        remaining = duration
        if not split and duration > capacity:
            return (
                "".join(parts) + f"{name}单项超过每日容量且不允许跨天暂停，现有约束下无可行排程。"
            )
        if not split and used + duration > capacity:
            days.append([])
            used = 0
        while remaining > 1e-9:
            if used >= capacity - 1e-9:
                days.append([])
                used = 0
            work = min(remaining, capacity - used)
            days[-1].append((name, work))
            used += work
            remaining -= work
    errors = validate_schedule(tasks, days, capacity, allow_split=split)
    if errors:
        raise ValueError("Schedule validation failed")
    if split and re.search(r"允许.*(?:暂停|跨天)|暂停任务.*继续", text):
        parts.append("按已允许的跨天暂停安排，固定工时和依赖不变。")
    elif split:
        parts.append("以下以允许任务跨天暂停为前提（需用户确认）；暂停不改变工时或依赖。")
    else:
        parts.append("各任务不跨天拆分。")
    for i, day in enumerate(days, 1):
        parts.append(f"第{i}天：" + "、".join(f"{name}{hours:g}小时" for name, hours in day) + "。")
    parts.append(
        f"最早需{len(days)}天；所有任务按依赖完成后再开始下一项，单人且无加班、减工时或并行。增加人手不会缩短此固定串行关键路径。以上仅评估，未执行任务。"
    )
    return "".join(parts)
