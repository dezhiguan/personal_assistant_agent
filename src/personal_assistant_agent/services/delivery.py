"""Runtime delivery contracts shared with the visible-content golden rule."""

import re


def requests_sensitive_disclosure(text: str) -> bool:
    """Recognize direct secret retrieval, not security explanations or negated reads.

    WorkspaceBackend remains the authority for every actual path operation.
    This narrow front guard prevents an affirmative answer from authorizing an
    explicit server-credential request before the model can launch tools.
    """
    if re.search(r"(?:仅|只)(?:识别|分析|解释|审查)|如何保护|怎样保护|如何防止", text):
        return False
    for clause in re.split(r"[，。；;\n]", text):
        if not re.search(
            r"\.env(?![A-Za-z0-9_])|id_rsa(?![A-Za-z0-9_])|id_ed25519(?![A-Za-z0-9_])"
            r"|credentials?\b|api[ _-]?key"
            r"|(?:API|服务端|服务器).{0,12}(?:密钥|密码|令牌)",
            clause,
            re.I,
        ):
            continue
        for verb in re.finditer(
            r"读取|读出|查看|打印|输出|泄露|告诉我|发给我|获取|\bread\b|\bprint\b|\breveal\b",
            clause,
            re.I,
        ):
            if not re.search(
                r"(?:不要|禁止|无需|别|不允许|do not|don't)\s*$",
                clause[max(0, verb.start() - 12) : verb.start()],
                re.I,
            ):
                return True
    return False


def name_only_request(request: str) -> bool:
    return bool(
        re.search(
            r"(?:只|仅)(?:回答|回复|输出)\s*(?:你的?)?(?:正式)?(?:名称|名字|姓名)"
            r"(?:即可)?\s*[。.!！?？]?\s*$",
            request,
        )
        and re.search(r"你(?:的)?(?:正式)?名|你叫什么", request)
    )


def name_only_answer(request: str, answer: str) -> str | None:
    """Strip a self-introduction wrapper only when the user requested a name alone."""
    if not re.search(r"(?:只|仅)(?:回答|回复|输出)", request) or not re.search(
        r"你(?:的)?(?:正式)?名|你叫什么", request
    ):
        return None
    match = re.fullmatch(
        r"我(?:的)?(?:正式)?(?:名字|名称)?(?:是|叫|为)\s*[“\"「]?"
        r"([^，。！？\n“”\"「」]{1,40})[”\"」]?[。.!！]?",
        answer.strip(),
    )
    return match[1].strip() if match else None


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
    if name_only_request(request) and (
        "\n" in text or re.search(r"[，。；！？]", text.rstrip("。.!！?？"))
    ):
        return True
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


def writing_body(answer: str) -> str:
    """Separate a short explicit delivery receipt from the requested composition."""
    heading = re.search(r"(?m)^#{1,6}\s+\S", answer)
    if heading and heading.start() <= 300:
        preamble = answer[: heading.start()]
        if re.search(r"完整正文|以下.{0,8}正文|已保存|经.{0,20}实测", preamble):
            answer = answer[heading.start() :]
    else:
        marker = re.search(r"(?:以下|下面)(?:为|是)(?:完整|作文)?正文[：:]\s*\n", answer[:300])
        if marker:
            answer = answer[marker.end() :]
    # A download link is a receipt, not the ending of the essay. Only standalone,
    # explicitly labelled receipt lines are excluded; arbitrary prose is kept.
    answer = re.sub(
        r"\n+(?:---\s*\n+)?(?:\[(?:下载|查看)(?:作文|文件|成稿)[^\]\n]*\]\([^\n]+\)|"
        r"(?:文件)?已保存(?:至|到)[^\n]+)[。.]?\s*$",
        "",
        answer,
    )
    return answer.strip()


def writing_delivery_issue(request: str, answer: str) -> str | None:
    """None is not a writing contract; empty string means the contract is met."""
    if not re.search(r"作文|散文|故事|文章", request):
        return None
    match = re.search(r"(?:字数\s*=\s*)?(\d+)\s*(字|词)", request)
    if not match:
        return None
    target, unit = int(match[1]), match[2]
    if target < 80:
        return None
    answer = writing_body(answer)
    english = bool(re.search(r"语言\s*=\s*(?:英文|英语)", request))
    chinese = bool(re.search(r"语言\s*=\s*(?:中文|汉语)", request))
    count = (
        len(re.findall(r"\b[A-Za-z]+(?:['’-][A-Za-z]+)*\b", answer))
        if english or unit == "词"
        else visible_length(answer)
    )
    capped = bool(re.search(r"不超过|最多|以内|以下|上限", request))
    low, high = (0, target) if capped else (int(target * 0.9), int(target * 1.1))
    if not low <= count <= high:
        return (
            f"当前正文实测{count}{'词' if english or unit == '词' else '字'}，需要{low}–{high}，"
            "不能通过删除结尾或截断正文凑字数"
        )
    han = len(re.findall(r"[\u4e00-\u9fff]", answer))
    letters = len(re.findall(r"[A-Za-z]", answer))
    if chinese and han < letters:
        return "最新语言要求是中文，请使用中文正文"
    if english and han > max(20, letters * 0.1):
        return "最新语言要求是英文，请使用英文正文"
    ending = re.sub(r"\s*[（(](?:全文|正文)?(?:共|约)?\s*\d+\s*[字词][）)]\s*$", "", answer)
    ending = re.sub(r"[*_`]+\s*$", "", ending)
    if not re.search(r"[。！？.!?][”’\"'」』]?\s*$", ending):
        return "正文缺少完整结尾"
    return ""


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


def requests_budget_verification(text: str) -> bool:
    """Require a real verification when the user explicitly requests it."""
    from personal_assistant_agent.services.scheduling import serial_contract

    if serial_contract(text) and re.search(r"核验|校验|验证", text):
        return True
    return bool(
        re.search(
            r"(?:用|使用|调用).{0,8}validate_time_budget"
            r"|(?:核验|校验|验证).{0,6}(?:时长|时间预算)|调用时长核验工具",
            text,
        )
    )
