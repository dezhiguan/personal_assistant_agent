"""Runtime delivery contracts shared with the visible-content golden rule."""

import re


def requests_arithmetic(request: str) -> bool:
    """Identify explicit numeric arithmetic questions, without matching calendar dates."""
    expression_text = re.sub(r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)", "", request)
    return bool(
        re.search(r"计算|算一下|算出|等于多少|求值", request)
        and re.search(r"\d+(?:\.\d+)?\s*[*×/÷+%\-]\s*\d+(?:\.\d+)?", expression_text)
    )


def requests_file_read(request: str) -> bool:
    """Recognize an imperative to read a concrete, ordinary workspace file."""
    match = re.search(
        r"(?:^|[。；;\n])(?:请|帮我)?先?(?:读取|读一下|查看)\s*[`“「\"]?"
        r"(/[^\s`“”「」\"'，。；;]+?\.[a-zA-Z0-9]{1,12})"
        r"(?=$|[\s`”」\"，。；;]|并|再|把|将|然后|的|中)",
        request,
    )
    if not match:
        return False
    path = match[1]
    return ".." not in path.split("/") and not re.search(
        r"\.env(?:$|/)|secret|credential|id_rsa|id_ed25519", path, re.I
    )


def requests_listing_tool(request: str) -> bool:
    """Keep explicitly named ls calls distinct from execute('ls ...')."""
    if re.search(r"恶意样本|仅识别|只识别", request):
        return False
    visible = re.sub(
        r'```[\s\S]*?```|<(system|developer|assistant)>[\s\S]*?</\1>|“[^”]*”|「[^」]*」|"[^"]*"',
        lambda match: " " * len(match[0]),
        request,
        flags=re.I,
    )
    for match in re.finditer(r"(?:调用|使用|再用)\s*`?ls`?(?![A-Za-z0-9_])", visible):
        before = visible[max(0, match.start() - 8) : match.start()]
        clause = re.split(r"[。；;\n]", request[match.start() :], maxsplit=1)[0]
        if not re.search(
            r"(?:不要|禁止|不准|无需|不必|不能|没有|别|不)\s*$", before
        ) and not re.search(r"\.\./|工作区外|密钥|\.env|credential", clause, re.I):
            return True
    return False


def delivery_monologue(request: str, answer: str) -> bool:
    """Recognize self-directed reasoning before an explicitly restricted answer."""
    text = answer.strip()
    if not text or text in request or not re.search(r"(?:只|仅)(?:回答|回复|输出|返回)", request):
        return False
    return bool(
        re.match(r"(?:用户|我(?:应该|需要|应当)|The user|I (?:need|should|must))", text, re.I)
        and re.search(
            r"我(?:应该|需要|应当|必须)|按照系统提示|I (?:need|should|must)", text[:500], re.I
        )
    )


def unsupported_report_progress(request: str, answer: str) -> str | None:
    """Check explicit closed-source status reports for unsupported progress claims.

    This narrow contract checks completion, project stage and action commitments;
    it does not attempt general semantic fact checking.
    """
    if not re.search(
        r"(?:写|起草|生成|改写|整理).{0,20}(?:周报|日报|月报|工作报告|进展简报)", request
    ):
        return None
    facts = re.search(r"唯一已知(?:事实|信息)?\s*[:：]\s*([^。\n]+)", request)
    if not facts:
        return None

    def affirmative(text, pattern):
        for match in re.finditer(pattern, text):
            before = text[max(0, match.start() - 8) : match.start()]
            if not re.search(r"(?:未|没有|尚无|不曾|建议)(?:\w{0,3})$", before):
                return True
        return False

    for pattern in (
        r"完成|已开发|已验证|验收通过|测试通过|已交付|已发布|已上线",
        r"(?:处于|进入|转入).{0,12}阶段",
        r"(?:下周|下月|明天|后续)(?:的)?计划|计划(?:开展|推进|补充|完成|安排)|将(?:开展|推进|完成)",
    ):
        if affirmative(answer, pattern) and not affirmative(facts[1], pattern):
            return facts[1]
    return None


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
