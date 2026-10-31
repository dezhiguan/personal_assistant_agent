"""Runtime delivery contracts shared with the visible-content golden rule."""

import re


def length_limit(text: str) -> int | None:
    matches = re.findall(
        r"(?:不超过|最多|不多于|限|控制在|全文|正文|回答|要求)?\s*(\d+)\s*(?:字|字符)(?:以内|以下|内)?",
        text,
    )
    return min(map(int, matches)) if matches else None


def visible_length(text: str) -> int:
    # Keep precisely the existing golden/UI Markdown counting semantics.
    from personal_assistant_agent.evaluation.golden import answer_length

    return answer_length(text, "visible")


def initial_steps(text: str) -> list[str]:
    # Explicit ordered work clauses: avoid creating plans for explanations or
    # quoted examples with no requested deliverable.
    if re.search(r"(?:不要|无需|禁止)(?:创建|建立|生成)(?:待办|计划)", text):
        return []
    if not re.search(r"请|帮我|设计|起草|制定|完成", text):
        return []
    markers = list(re.finditer(r"(?:首先|先|然后|再|接着|最后)", text))
    if len(markers) < 3:
        return []
    steps = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        clause = re.split(r"[。；;！\n]", text[marker.end() : end])[0].strip(" ，,：:")
        if not re.search(
            r"明确|分配|设计|起草|撰写|整理|核验|读取|创建|生成|分析|检查|计算", clause
        ):
            return []
        steps.append(clause[:500])
    return steps[:32]
