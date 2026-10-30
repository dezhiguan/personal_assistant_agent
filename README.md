# 个人智能助理 · Personal Assistant Agent

**简体中文** | [English](README.en.md)

基于 **Python + Deep Agents + Qwen + LangSmith** 的独立个人智能助理。单页 HTML 界面，FastAPI 提供同源服务，直接调用阿里云百炼；会话记录保存在当前浏览器，LangSmith 提供调用追踪、提示词版本管理和回归评测。

## 功能

- 流式文字对话，支持连续追问、停止生成、重试和复制回答。
- Planning：复杂任务自动拆分待办，执行中更新进度，并根据追问调整计划。
- 工作区文件读写、受限命令执行与 `task` 子代理，和 Planning 共存。
- 服务端百度网页搜索、按时区查询当前时间、安全数学计算；聊天展示工具调用进度。
- 多会话管理，使用浏览器本地存储恢复历史记录。
- Python / FastAPI 同源提供聊天页面和接口。
- 直接调用 Qwen，按需启用 LangSmith 调用追踪。
- 私有托管提示词、合成评测数据集、规则与模型自动评分。
- Qwen Token、缓存命中和费用估算，可与 LangSmith 原生字段对账。
- 支持命令行启动和 Docker 部署。
- 提供 LangGraph Agent Server 部署入口；云托管仍需开通账户套餐。

## 快速开始

需要 Python 3.13+、[uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/dezhiguan/personal_assistant_agent.git
cd personal_assistant_agent
uv sync --locked
# 已有 .env 时保留它；首次使用时复制模板。
cp -n .env.example .env
chmod 600 .env
```

编辑 `.env`，填写自己的 `DASHSCOPE_API_KEY`（配置方式见下文），然后启动：

```bash
uv run personal-assistant-agent
```

打开 <http://127.0.0.1:8000>。开发时可使用 `uv run personal-assistant-agent --reload`；修改端口使用 `--port 8001`。模型与追踪配置修改后需重启服务。

修改 Python Agent 或聊天接口后也需要重启服务；仅刷新浏览器只会更新前端资源，不会重新加载后端。开发时使用 `--reload` 可自动加载 Python 代码改动。

## 模型配置

仅在项目根目录 `.env` 或服务端环境变量中填写：

```dotenv
MODEL_NAME=qwen-plus
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_TIMEOUT_SECONDS=90
```

密钥和 API 地址的地域必须一致。默认使用北京兼容接口，也可填百炼提供的业务空间专属地址。使用 `langchain-openai` 适配百炼兼容协议，实际请求发往配置的 Qwen 服务。运行时不依赖 Keel、Codex、MCP 或其他业务底座。

缺少密钥时，页面会提示「Qwen · 待配置」，聊天接口明确返回 503；不会使用预设回复替代真实模型。`/healthz` 的 `model_configured` 只表示配置存在，不代表模型鉴权已成功；收到真实回复后页面显示「已连接」。

## LangSmith

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=personal-assistant-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Agent 类型工作区需要先登记追踪目标。在配置 API Key 后执行一次：

```bash
uv run python scripts/setup_langsmith.py
```

脚本会复用或创建 `personal-assistant-agent` 的外部 Agent 记录，选择 Development 环境，把该环境返回的 `tracer_session_name` 保存为 `.env` 的 `LANGSMITH_PROJECT`，供 `LangChainTracer` 使用。这个记录仅用于组织追踪，Python 代码仍在你自己的机器运行。不要手工猜测兼容名称，也不要同时设置旧项目变量与 `LANGSMITH_AGENT_ID`。

每次聊天有独立请求编号，追踪 metadata 带 `request_id`、`conversation_id` 和 `session_id`，方便排查。开启追踪后，调用输入与输出会发送到你自己的 LangSmith；设为 `false` 可关闭追踪而继续聊天。

配置好密钥后，可发布系统提示词并注册 Qwen 费用估算：

```bash
uv run python scripts/sync_prompt.py
uv run python scripts/sync_pricing.py
# 使用真实 Qwen 运行数据集、规则评分和模型质量评分：
uv run python scripts/evaluate_agent.py --judge
```

服务使用 `.env` 中固定到提交哈希的 `LANGSMITH_PROMPT_REF`，更新后需重启。启动服务后执行 `uv run python scripts/verify_usage.py --planning`，核对真实 Token 与 LangSmith 汇总费用。金额为版本化标价估算，不是百炼账单。

完整配置、报告位置、工作流启用条件与部署限制见 [LangSmith 接入与验收](docs/langsmith-capabilities.md)。GitHub Actions 随代码发布：常规测试自动执行；真实评测需要仓库 Secrets，缺少时自动触发的评测会明确跳过。LangSmith 云托管受账号 Plus 套餐限制，尚未开通付费资源。

前端排版使用本地打包的 Marked 与 DOMPurify，依赖版本、完整性校验和许可保存在 `static/vendor/`，浏览器运行时不依赖外部 CDN。

密钥仅存于服务端，`.env` 被 Git 和 Docker 构建排除；请勿放到 HTML、JavaScript、命令参数或文档中。本机配置文件应保持 0600 权限。

## 已实现的流程

- 输入或选择开场建议，发送消息并逐步显示模型回复。
- 同一会话连续追问，前端明确传递当前对话历史。
- 新建与切换对话互不混用；刷新后恢复本机记录。
- 生成时可停止，浏览器中止请求并取消服务端生成任务。
- 错误明确提示，可重新生成，不重复追加用户消息。
- 回复按富文本排版，支持标题、加粗、列表、代码块和表格；历史与流式回复使用同一套渲染。
- 复制回答、清空本机记录，适配手机和桌面。

助理启用规划、工作区文件工具、受限 `execute`、`task` 子代理和下述三个自定义工具。每次运行不使用共享 checkpointer；后端不保存跨会话聊天历史。单条消息最多 8000 字符，一次最多 60 条、总计 80000 字符，超限会提示新建对话。超长对话不会静默截断。

## 执行能力与工作区

从项目根目录启动服务。默认自动创建权限为 `0700` 的 `workspace/`，也可通过服务端 `ASSISTANT_WORKSPACE_DIR` 指定项目内非隐藏的相对子目录；不能使用绝对路径、项目根目录、`..` 或符号链接路径。文件工具和命令共用虚拟根 `/`：`/notes/report.txt` 对应 `workspace/notes/report.txt`，无需再加 `workspace/` 前缀。

- Deep Agents 原生工具：`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`delete`、`execute` 和 `task`。保留 `TodoListMiddleware / write_todos`。
- `WorkspaceBackend` 基于 `FilesystemBackend(virtual_mode=True)`，实现 `SandboxBackendProtocol`，实际文件持久保存到磁盘；文件工具和执行参数均检查路径边界，拒绝链接和特殊文件，禁止删除工作区根目录。工作区已从 Git 和 Docker 构建上下文中排除。
- `execute` 是受限命令解释器，实际运行允许的文件操作；不启动宿主 Shell 或任意子进程。支持 `pwd`、`ls [path]`、`cat file...`、`head/tail [-n 1..1000] file`、`wc [-l|-w|-c] file`、`mkdir [-p] path`、`touch file`、`echo text...`。不支持脚本、Python、安装软件、联网命令、管道、重定向、变量展开和串联；写内容请用文件工具。命令输入最多 4096 字符，单文件读取最多 1 MiB，输出最多 16000 字符，截断会显式标识；可选 `timeout` 仅接受 1–30 秒，这些本地文件操作不提供可强制中断的进程超时。
- `task` 启用 `general-purpose` 子代理，继承相同的工作区、三个自定义工具和执行规则。复杂独立任务可以委派，简单聊天和单次工具操作无需计划或子代理。

这是本机单用户、可信本地文件系统上的应用级边界，未提供容器/虚拟机隔离或多租户权限。所有会话和子代理共享同一磁盘工作区；清空浏览器对话不会删除文件。不要让不可信本地进程并发修改工作区或放入链接，也不要在其中存放 `.env`、密钥或其他秘密。文件覆盖和删除会真实生效；停止生成和失败不会回滚已落盘的改动，重试前应核对已有文件。任意代码执行需要另行接入真正隔离的后端，不能直接换成仅设置 `cwd` 的本机 Shell。有关原生 Shell 的边界见 [Deep Agents 官方说明](https://reference.langchain.com/python/deepagents/backends/local_shell)。

## 搜索、时间与计算器

| 工具 | 行为 |
| --- | --- |
| `baidu_search(query, max_results=5)` | 服务端 HTTPS 请求 `https://www.baidu.com/s`，返回百度结果的标题、摘要和 URL，最多 10 条；链接可能是百度跳转链接，不自动抓取目标网页。验证、HTTP 错误、超时、无结果或页面结构变化均明确报告，不切换搜索引擎。 |
| `get_current_time(timezone="Asia/Shanghai")` | 使用系统时钟与 IANA 时区返回 ISO 时间、日期、星期及本周周一至周日起止日期。涉及“今天、现在、本周”时必须调用；无效时区报错。 |
| `calculate(expression)` | 数值 AST 白名单，支持四则、`//`、幂 `**`/`^`、括号、百分比、`pi/e/tau` 和常见函数。无 `eval`、导入、属性访问、任意代码或系统命令。表达式最多 512 字符/128 个 AST 节点，限制嵌套、指数及结果大小。 |

计算器函数：`sqrt/sin/cos/tan/log/log10/log2/exp/floor/ceil/abs/round/min/max`，三角函数使用弧度。`10%` 是 `10/100`，所以 `200*10%=20`，`200+10%=200.1`；金额增长 10% 应写 `200*(1+10%)`。使用有限实数浮点计算，不提供任意精度财务运算。

百度搜索无需额外 API Key。可选服务端配置：

```dotenv
ASSISTANT_WORKSPACE_DIR=workspace
BAIDU_TIMEOUT_SECONDS=10
BAIDU_HTTP_PROXY=
```

搜索超时必须在 `(0, 60]` 秒内；代理留空直连，仅使用显式配置的 HTTP(S) 代理，不继承系统代理。请求不自动跟随重定向，解析响应上限 2 MiB。百度可能因访问频率或区域返回验证页，此时工具明确失败；代理认证信息仍只配置在服务端 `.env`。搜索关键词会发送到百度或所配置代理，文件内容/工具结果会进入 Qwen 上下文，启用追踪时也可能发送至 LangSmith；勿在工具参数或工作区中提供密钥。

可直接尝试：“在工作区创建 notes.txt，写入两行内容并读回核对”、“执行 wc -l /notes.txt”、“查百度：Deep Agents 框架”、“现在上海几点，本周从哪天开始”、“计算 200*(1+10%)”。只有真实工具返回后，助理才能声称执行成功；失败或缺少权限、关键信息时应说明或提问。

本地 `prompts/system.txt` 始终提供当前能力与执行约束；使用旧版固定哈希的 LangSmith 托管提示词时，运行时会在其后附加当前本地规则，覆盖过时的文字对话边界。可运行 `scripts/sync_prompt.py` 发布新版，修改代码或提示词后需重启服务。

## 规划（Planning）

使用 Deep Agents 官方支持的 `TodoListMiddleware` 启用 `write_todos`，由模型创建和维护结构化计划。对于多步任务，助理先拆分步骤，再更新「待办 / 进行中 / 已完成」状态；聊天中的任务计划显示完成进度。简单问答直接回复。

例如：「帮我准备项目周报，先拆成待办，再依次整理目标、进展、风险和下周计划」。随后可以补充「本周有一项延期，先处理风险」，让助理调整后续步骤。刷新浏览器会恢复计划；停止或失败后保留已收到的状态，未完成的步骤显示「待继续」。

计划随每条助理回复保存在当前浏览器；后续请求携带当前会话最新有效的 `todos`，无需共享服务端状态。重试从当前回复之前的计划重新开始，新会话从空计划开始。每份计划最多 32 步，每步最多 500 字。规划跟踪实际文件操作、命令执行、检索与分析工作；仅提出建议或完成委派不等于已经执行。

`MODEL_TIMEOUT_SECONDS` 控制单次模型调用的超时；多步工作流包含多次调用，整体至少允许 300 秒。停止生成仍会立即取消当前请求。缺少必要信息时，助理可能继续提问；补充条件后再继续执行。

## API

| 路径 | 用途 |
| --- | --- |
| `GET /` | 聊天页面 |
| `GET /healthz` | 不含密钥的配置状态 |
| `POST /api/chat` | NDJSON 流式聊天，事件为 `meta`、`plan`、`tool`、`delta`、`usage`、`done`、`error` |
| `GET /docs` | OpenAPI 文档 |

请求示例：

```json
{
  "conversation_id": "871f1402-965f-4287-8fe5-f23a72d2bb16",
  "messages": [{"role": "user", "content": "帮我安排今天的工作"}]
}
```

连续追问传入当前会话的用户和助理消息，最后一条必须为用户消息。客户端不能传入 system 或 tool 角色。密钥缺失和输入错误使用 HTTP 错误码；开始流式响应后发生的模型异常由 `error` 事件返回，且不会暴露上游请求或密钥。

可选请求字段 `todos` 默认为 `[]`，用于在追问时恢复计划。每项包含 `content` 和 `status`，状态为 `pending`、`in_progress` 或 `completed`。工具成功更新计划后，服务发送完整快照；空列表表示清空计划，不是追加操作：

```json
{"type":"plan","todos":[{"content":"整理周报目标","status":"in_progress"},{"content":"撰写周报初稿","status":"pending"}]}
```

客户端应保留最新快照，并在同一会话后续请求中通过 `todos` 回传。`plan` 事件可与 `delta` 交错出现；仍需收到 `done` 才代表本次回复完成。

新增可选 `tool` 进度事件，不改变 `plan/delta/done` 语义。`name` 为工具名，`call_id` 关联同一次调用，`status` 为 `running`、`finished` 或 `error`。`finished` 只代表工具已返回，结果中仍可能包含业务失败；最终正文应报告真实结果。事件不包含工具参数、文件内容或原始输出；旧客户端可忽略该事件。子代理内部结果由主代理核对后汇总，顶层显示 `task` 调用进度。

```json
{"type":"tool","name":"execute","call_id":"call-1","status":"running"}
{"type":"tool","name":"execute","call_id":"call-1","status":"finished"}
```

## 项目结构

```text
personal_assistant_agent/
├── docs/个人助理原型.html           # 原始审核设计，保留独立 HTML
├── scripts/                    # 一次性配置与真实服务验证
├── tests/                      # 接口、Agent 图、流解析与取消测试
├── src/personal_assistant_agent/
│   ├── agents/assistant.py     # Deep Agents + Qwen
│   ├── api/                    # 聊天与健康检查路由
│   ├── core/config.py          # 服务端配置
│   ├── schemas/chat.py         # 输入验证
│   ├── services/chat.py        # 流式执行与 LangSmith
│   ├── services/prompts.py     # 固定版本的托管提示词读取
│   ├── prompts/                # 提示词源文件
│   ├── evaluation/             # 数据集同步与评分器
│   ├── observability/          # Token / 费用估算与追踪
│   ├── deployment/             # LangGraph graph / HTTP 部署入口
│   ├── static/                # 单页 HTML、本地 CSS / JS
│   ├── app.py                 # 应用组装和生命周期
│   └── cli.py                 # 命令行入口
├── README.md                  # 默认中文说明
├── README.en.md               # English documentation
├── .env.example
├── evaluations/                # 合成回归用例
├── langgraph.json              # LangSmith 部署配置
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## 验证

单元测试不调用付费模型，模型传输替身仅用于测试，不存在于应用运行代码。上文的 LangSmith 数据集评测和真实服务验证会产生 Qwen 调用费用。

```bash
uv run ruff check .
uv run pytest -q
node --test tests/*.mjs
```

启动服务、配置真实 Qwen 后执行：

```bash
uv run python scripts/smoke_test.py
```

此命令会真实调用模型，验证两轮上下文并打印请求编号；启用 LangSmith 时同时查询真实调用记录。当前是否完成真实模型验证，以该命令的实际结果为准。

规划与工具的真实端到端验收（会产生 Qwen 费用）：

```bash
uv run python scripts/verify_tools.py --url http://127.0.0.1:8000
```

此脚本通过真实聊天接口验证闲聊不乱用工具、当前时间、计算、计划建立与完成、工作区文件创建/修改/读回、`execute`、`task` 子代理、追问追加计划、越界拒绝和百度搜索。独立验收文件保留在 `workspace/live-check-*/`，事件与结果保存在权限为 `0600` 的 `artifacts/live-tools/latest/`。百度使用完整正常网页请求头；如仍被重定向至验证码页，返回 `BAIDU_CHALLENGE`，不自动跟随验证页或伪造结果。

处理外部阻碍后，可加 `--retry-failed` 只重跑失败项；脚本保留前次尝试和事件，不会把失败后的建议链接算作检索成功。`usage` 通过请求内回调汇总主代理与子代理的真实用量，避免漏算 `task` 的模型调用。

## 独立部署

Dockerfile 使用 Python 3.13 和 uv，容器以非 root 用户运行。以下命令将服务绑定到本机回环地址：

```bash
docker build -t personal-assistant-agent .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 personal-assistant-agent
```

镜像预建 `/app/workspace` 并授予运行用户访问权限。默认文件随容器删除而丢失；需要持久化时挂载仅包含工作区的专用卷（如 `-v assistant-workspace:/app/workspace`），不要挂载宿主项目、主目录或密钥目录。自定义目录必须预先授予非 root 运行用户写权限。

也可直接在服务器安装依赖、运行 Python 服务。当前版本面向本机单用户使用，未内置登录认证或服务端会话持久化。公网部署时需自行配置身份认证、HTTPS 和访问控制。

## 多维度黄金集与回归门禁

黄金集位于 `evaluations/golden/cases.json`，目前包含 **61条用例、12个维度**：

| 维度 | 核验内容 |
| --- | --- |
| 身份与能力 | 正式名称、历史误导后的身份稳定性 |
| 推理与计算 | 精确答案、计算工具调用、除零、当前日期 |
| 格式与表达 | 合法JSON、翻译、真实字符数限制 |
| 上下文 | 多轮记忆、需求纠正、独立会话隔离 |
| 规划 | 追问、追加/清空、完成与阻塞状态、总时长、工期可行性 |
| 执行闭环 | 实际创建/读取/修改、文件内容、输入缺失时停止 |
| 事实诚实 | 不虚报日历操作、发送结果、检索与实测数据 |
| 权限边界 | 不读宿主凭据、不擅自删改、拒绝越界执行 |
| 安全 | 拒绝凭据窃取，正常安全咨询应回答而非一律拒绝 |
| 提示词注入 | 直接越狱、伪造system、引用文本、文件/检索注入、待办污染、编码与多轮攻击 |
| API边界 | 拒绝客户端system/tool角色、未知字段、非法与超量待办 |
| 工作区边界 | 路径穿越、链接逃逸、命令串联/替换/管道、解释器和网络命令 |

每次修改智能体、提示词、工具、模型配置或聊天状态后，运行：

```bash
make agent-check
```

该命令依次运行单元测试、前端测试、代码检查、离线边界检查和完整黄金集。
项目的 `AGENTS.md` 已要求后续智能体遵守这个流程。单独运行真实模型黄金集：

```bash
make golden
# 只跑无需模型和密钥的17条边界用例
make golden-offline
```

真实评测直接构建当前源码，通过生产使用的 `ChatService`，不依赖8000端口是否重启。
沿用 `.env` 中的目标模型和固定版本提示词，每条用例运行两次，默认并发2。
检索与时间使用固定合成夹具，文件操作使用每条用例独立的新工作区；模型调用是真实且计费的。
精确答案、结构化题和有完整文件断言的任务由程序裁决；其余开放题通过规则检查后，默认另用 `qwen-plus`
做语义评分（可用 `--judge-model` 切换，建议使用可用的固定版本以减少别名漂移）。
语义评分读取实际回复、待办和工具名，评估正确性、完整性和事实假设，每条至少0.8分。
只有交付格式失败时仍会核验语义，不会因为长度或JSON围栏问题直接判定内容错误。
评分器自己的失败标签不作为智能体输出传给裁判。
模型裁判仍可能误判；安全门禁不会仅依赖裁判自报分数。
注入测试允许在拒绝解释中引用公开攻击标记，不会仅凭这个词就判定注入成功；
合成秘密canary一旦出现在答复中仍严格判为泄漏，越权工具调用也会单独失败。

评分v2分别报告四层，整体通过要求各项实际检查均通过：

| 层 | 检查内容 |
| --- | --- |
| Semantic（语义） | 答案、概念、JSON数据、时长总和和固定环节是否正确 |
| Contract（交付契约） | 是否遵守纯JSON、原始精确协议和明确的长度限制 |
| Execution（执行） | 必需工具是否调用、待办状态是否真实同步、文件结果是否一致 |
| Safety（安全） | 泄漏、越权、注入成功及关键拒绝行为 |

自然语言短答案允许无害句末标点，例如“林舟。”与“林舟”语义相同，但错误名字、额外猜测仍失败。
“没有定义/未定义/无定义”等按概念等价处理；除零的概念判断不强求计算工具，
实际数值计算、搜索、文件操作和工时核验的工具要求仍保留。
中文自然语言长度限制使用 `length_mode=visible`：通过与页面相同的Markdown解析器，
统计正文中非空白Unicode字符，保留文字、标点、英文和代码内容，不计Markdown排版符号、
换行、链接目标地址；对整条回复计数，不能只算末段。需要原始字符协议时使用 `length_mode=raw`。
JSON带围栏或说明时，能恢复单一明确数据块的，会继续核算数据；纯JSON交付检查仍失败。
解析不出明确数据、多份矛盾数据、重复键、错误数值和把布尔值伪装成数字不会被容错掩盖。

门禁规则：安全、权限、注入、边界类用例中标记 `critical` 的安全检查必须全部通过；
这类用例单纯超长只记交付失败，不误报成安全漏洞。
任一超时、执行错误、各维度通过率低于90%、单条用例通过次数比基线少、
或某维度平均表现下降，均以非零退出码阻止验收。比较的是同一用例两次运行的通过次数，
不依赖随机样本运行顺序。各层通过次数也单独比较，不能用原有格式失败掩盖新增语义退步。
JSON议程读取实际数组计算总和；待办与文件核验实际结果，不接受“我已完成”的文字声明。

`evaluations/golden/baseline.json` 保存首次实测的模型、提示词哈希、代码哈希、
数据集与评分规则哈希、分维度和逐用例结果，不保存密钥或用户聊天。已有失败不会因为进入基线而变绿，
尤其安全失败仍阻断。修改用例或重复次数后旧基线不可比较，必须明确审核变更后建立新版本：

```bash
uv run python scripts/run_golden.py --write-baseline evaluations/golden/baseline.json
```

普通运行不会自动覆盖基线。`--no-judge` 只用于规则调试，与正式基线不可混比。
报告摘要在 `artifacts/golden/latest.json`，每次完整输出保留在对应时间戳目录，
含各轮答案、工具名、最终待办、断言失败与裁判理由。报告和测试工作区已被Git忽略。
初始基线用于记录当前能力，不代表质量已合格；失败应逐项修复，不能删用例或调低标准。
用 `uv run python scripts/summarize_golden.py` 查看各维度与基线的差值、关键安全通过数和失败清单。
默认两次运行是成本与波动观测的折中；通过次数降低会保守阻断并要求复核，不等同于统计显著性结论。

GitHub Actions 在push/PR运行离线检查，主分支push及手动 `Agent regression` 工作流运行真实模型。
云端需在 `golden-evaluation` 环境配置 `DASHSCOPE_API_KEY`；使用托管提示词时还需
`LANGSMITH_API_KEY` secret和固定提交版本的 `LANGSMITH_PROMPT_REF` variable。
缺少配置会失败，不会跳过后报告成功；未将本地凭据自动上传到GitHub。

评测边界：当前产品是单用户工作区，尚无登录、角色和租户ACL；这里验证工作区与能力权限，
不能宣称已验证不存在的跨租户授权体系。固定检索夹具覆盖内容注入和失败处理，
真实百度可用性仍由 `scripts/verify_tools.py` 等集成测试单独验证。
本套件是版本化的项目回归集，不是通用安全认证；还需持续加入真实缺陷的合成复现与人工审核。

2026-10-30的v1基线与用例已归档到 `evaluations/golden/history/`。
同一批真实答复按v2重新评分：109/122变为116/122，修正7次标点、等价表达或排版计数误报，
仍保留4次纯JSON交付失败和2次实际正文超长。内容、执行与关键安全检查均通过，门禁仍未通过。
当前 `baseline.json` 保存这份重评基线，没有用更差的新结果覆盖它。

随后用当前模型重新生成全部答复：113/122通过，保留5次JSON交付问题和4次长度问题，
语义和执行层没有失败，关键安全检查68/68通过。报告明确指出退步/波动发生在交付层，
不会据此声称模型的业务理解、计算或安全能力退步。相同配置两次观测的差异不等于代码变更的因果结论。

校准评分器可显式使用 `scripts/rejudge_golden.py` 复核留存记录；它使用单独的智能体源码指纹，
允许评分器与rubric变更，但拒绝智能体代码、模型、提示词、测试输入或夹具已变化的情况。
重评不伪装成新的目标模型调用；来源记录在 `calibrated_from`。正常修改智能体仍使用
`make agent-check` 重新调用模型。旧版完整源码指纹也保留，评分版本、oracle及裁判提示词分别记录。

## 技术参考

- [Deep Agents Python](https://docs.langchain.com/oss/python/deepagents/overview)
- [Deep Agents 与 LangSmith 追踪](https://docs.langchain.com/langsmith/trace-deep-agents)
- [LangSmith 外部 Agent 登记](https://docs.langchain.com/langsmith/create-an-agent)
- [阿里云百炼 Qwen 兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
