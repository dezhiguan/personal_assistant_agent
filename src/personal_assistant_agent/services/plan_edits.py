"""Handle unambiguous, explicit plan edits without relying on model assertions."""

import re


def explicit_edit(text: str, todos: list[dict]) -> tuple[list[dict], str | None] | None:
    # Only imperative, explicitly scoped edits. General replanning stays with the agent.
    if re.search(
        r"(?:^|[。；，\n]|并)(?:请|现在|立即|先)?清空"
        r"(?:当前|旧|原有)?(?:任务计划|待办列表|待办)(?=$|[。；，！\n])",
        text,
    ):
        if not re.search(r"(?:不要|不|不能|别)清空", text):
            return [], None
    match = re.search(r"只在原待办末尾追加一项[“「]([^”」]+)[”」]", text)
    if match and todos and len(todos) < 32 and len(match[1]) <= 500:
        return [
            *todos,
            {"content": match[1], "status": "pending"},
        ], "已追加待办，原有步骤和状态保留。"
    return None
