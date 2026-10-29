# 个人智能助理 · Personal Assistant Agent

**简体中文** | [English](README.en.md)

基于 **Python + Deep Agents + Qwen + LangSmith** 的独立个人智能助理。单页 HTML 界面，FastAPI 提供同源服务，直接调用阿里云百炼；会话记录保存在当前浏览器，LangSmith 用于追踪调用。

## 功能

- 流式文字对话，支持连续追问、停止生成、重试和复制回答。
- Planning：复杂任务自动拆分待办，执行中更新进度，并根据追问调整计划。
- 多会话管理，使用浏览器本地存储恢复历史记录。
- Python / FastAPI 同源提供聊天页面和接口。
- 直接调用 Qwen，按需启用 LangSmith 调用追踪。
- 支持命令行启动和 Docker 部署。

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

密钥仅存于服务端，`.env` 被 Git 和 Docker 构建排除；请勿放到 HTML、JavaScript、命令参数或文档中。本机配置文件应保持 0600 权限。

## 已实现的流程

- 输入或选择开场建议，发送消息并逐步显示模型回复。
- 同一会话连续追问，前端明确传递当前对话历史。
- 新建与切换对话互不混用；刷新后恢复本机记录。
- 生成时可停止，浏览器中止请求并取消服务端生成任务。
- 错误明确提示，可重新生成，不重复追加用户消息。
- 复制回答、清空本机记录，适配手机和桌面。

MVP 使用单个文字助理，启用规划工具，不启用文件、Shell、联网和子助理工具。每次运行不使用共享 checkpointer；后端不保存跨会话历史。单条消息最多 8000 字符，一次最多 60 条、总计 80000 字符，超限会提示新建对话。超长对话不会静默截断。

## 规划（Planning）

使用 Deep Agents 官方支持的 `TodoListMiddleware` 启用 `write_todos`，由模型创建和维护结构化计划。对于多步任务，助理先拆分步骤，再更新「待办 / 进行中 / 已完成」状态；聊天中的任务计划显示完成进度。简单问答直接回复。

例如：「帮我准备项目周报，先拆成待办，再依次整理目标、进展、风险和下周计划」。随后可以补充「本周有一项延期，先处理风险」，让助理调整后续步骤。刷新浏览器会恢复计划；停止或失败后保留已收到的状态，未完成的步骤显示「待继续」。

计划随每条助理回复保存在当前浏览器；后续请求携带当前会话最新有效的 `todos`，无需共享服务端状态。重试从当前回复之前的计划重新开始，新会话从空计划开始。每份计划最多 32 步，每步最多 500 字。规划跟踪的是助理在对话中完成的分析、整理和撰写工作；提出外部行动不等于已经执行。

## API

| 路径 | 用途 |
| --- | --- |
| `GET /` | 聊天页面 |
| `GET /healthz` | 不含密钥的配置状态 |
| `POST /api/chat` | NDJSON 流式聊天，事件为 `meta`、`plan`、`delta`、`done`、`error` |
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
│   ├── static/                # 单页 HTML、本地 CSS / JS
│   ├── app.py                 # 应用组装和生命周期
│   └── cli.py                 # 命令行入口
├── README.md                  # 默认中文说明
├── README.en.md               # English documentation
├── .env.example
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## 验证

自动化测试不调用付费模型，模型传输替身仅用于测试，不存在于应用运行代码。

```bash
uv run ruff check .
uv run pytest -q
node --test tests/test_stream.mjs
```

启动服务、配置真实 Qwen 后执行：

```bash
uv run python scripts/smoke_test.py
```

此命令会真实调用模型，验证两轮上下文并打印请求编号；启用 LangSmith 时同时查询真实调用记录。当前是否完成真实模型验证，以该命令的实际结果为准。

## 独立部署

Dockerfile 使用 Python 3.13 和 uv，容器以非 root 用户运行。以下命令将服务绑定到本机回环地址：

```bash
docker build -t personal-assistant-agent .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 personal-assistant-agent
```

也可直接在服务器安装依赖、运行 Python 服务。当前版本面向本机单用户使用，未内置登录认证或服务端会话持久化。公网部署时需自行配置身份认证、HTTPS 和访问控制。

## 技术参考

- [Deep Agents Python](https://docs.langchain.com/oss/python/deepagents/overview)
- [Deep Agents 与 LangSmith 追踪](https://docs.langchain.com/langsmith/trace-deep-agents)
- [LangSmith 外部 Agent 登记](https://docs.langchain.com/langsmith/create-an-agent)
- [阿里云百炼 Qwen 兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
