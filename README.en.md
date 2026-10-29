# Personal Assistant Agent · 个人智能助理

[简体中文](README.md) | **English**

A standalone personal assistant built with **Python, Deep Agents, Qwen, and LangSmith**. FastAPI serves a single-page chat interface and its API from the same origin. The backend calls Alibaba Cloud Model Studio directly; conversation history stays in the current browser, with optional LangSmith tracing.

## Features

- Streaming text conversations with follow-up questions, stop, retry, and copy controls.
- Planning for multi-step tasks, with live todo progress and revisions during follow-ups.
- Workspace file tools, restricted command execution, and a `task` subagent alongside Planning.
- Server-side Baidu search, a timezone-aware clock, safe calculations, and tool progress in chat.
- Multiple conversations with history restored from browser local storage.
- A Python / FastAPI backend serving the interface and API together.
- Direct Qwen integration with optional LangSmith tracing.
- Private versioned prompts, synthetic datasets, rule-based evaluation and a Qwen quality judge.
- Measured token/cache usage and estimated costs reconciled against native LangSmith fields.
- Command-line startup and a Dockerfile for deployment.
- LangGraph Agent Server configuration; cloud hosting still requires an eligible paid account.

## Quick start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dezhiguan/personal_assistant_agent.git
cd personal_assistant_agent
uv sync --locked
# Preserve an existing .env; copy the template on first use.
cp -n .env.example .env
chmod 600 .env
```

Edit `.env` and set your own `DASHSCOPE_API_KEY` as described below, then start the server:

```bash
uv run personal-assistant-agent
```

Open <http://127.0.0.1:8000>. For development, use `uv run personal-assistant-agent --reload`; change the port with `--port 8001`. Restart the server after changing model or tracing settings.

The interface and assistant's default responses are in Chinese. The English README documents the same application.

## Model configuration

Set these values in the project-root `.env` file or the server environment:

```dotenv
MODEL_NAME=qwen-plus
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_TIMEOUT_SECONDS=90
```

The API key and endpoint must belong to the same region. The default endpoint is the Beijing OpenAI-compatible endpoint; a Model Studio workspace-specific endpoint can also be configured. `langchain-openai` adapts the compatible protocol, and requests go directly to the configured Qwen service. The application runs independently of Keel, Codex, MCP, and other platforms.

If the key is missing, the interface shows that Qwen needs configuration, and the chat endpoint returns HTTP 503. The application does not substitute canned replies for model responses. The `model_configured` field in `/healthz` indicates that configuration exists; it does not verify authentication. The interface shows a connected state after receiving a real response.

## LangSmith

Tracing is optional and disabled in `.env.example`. To enable it, configure:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=personal-assistant-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

For a LangSmith Agent workspace, register the tracing destination once after setting your API key:

```bash
uv run python scripts/setup_langsmith.py
```

The script reuses or creates an external Agent record named `personal-assistant-agent`, selects its Development environment, and saves the environment's `tracer_session_name` as `LANGSMITH_PROJECT` in `.env` for `LangChainTracer`. This record organizes traces; the Python application still runs on your machine. Do not invent a compatibility name or combine legacy project variables with `LANGSMITH_AGENT_ID`.

Each chat request has a unique request ID. Trace metadata includes `request_id`, `conversation_id`, and `session_id`. When tracing is enabled, model inputs and outputs are sent to your LangSmith workspace. Set `LANGSMITH_TRACING=false` to continue chatting without tracing.

Publish a private system prompt, register the pricing snapshot and run a real dataset experiment:

```bash
uv run python scripts/sync_prompt.py
uv run python scripts/sync_pricing.py
uv run python scripts/evaluate_agent.py --judge
```

Restart the service after changing the pinned `LANGSMITH_PROMPT_REF`. With the service running, `uv run python scripts/verify_usage.py --planning` compares actual provider usage, model spans and root trace totals. Costs use versioned list prices and an FX snapshot; they are estimates, not invoices. Rule checks must all pass and each optional judge score must be at least 0.8.

See [LangSmith integration and verification](docs/langsmith-capabilities.md) for commands, artifacts and deployment constraints. GitHub Actions workflows ship with the code: standard tests run automatically; live evaluation requires repository secrets and explicitly skips automatic runs when they are missing. Cloud deployment is blocked by the current account's Plus upgrade requirement; no paid resources were provisioned. `langgraph.json` includes the graph and existing web app with platform authentication on custom routes.

Keep API keys on the server. Git and Docker builds exclude `.env`; never put keys in HTML, JavaScript, command arguments, or documentation. Keep the local configuration file at permission mode `0600`.

## Conversation behavior and limits

- Send a message or choose a suggested prompt to receive a streamed response.
- Follow-up requests explicitly include the current conversation history.
- New and existing conversations remain separate; refreshing restores local history.
- Stopping generation aborts the browser request and cancels the server task.
- Failed responses can be regenerated without adding a duplicate user message.
- Copy answers or clear local history on desktop and mobile.

The assistant exposes Planning, workspace files, restricted `execute`, `task`, Baidu search, the clock and calculator. There is no shared checkpointer, and the backend does not persist conversation history between requests. Each message is limited to 8,000 characters; a request can contain at most 60 messages and 80,000 characters in total. Oversized conversations are rejected with a prompt to start a new conversation rather than silently truncated.

## Workspace execution and custom tools

Start from the project root. `ASSISTANT_WORKSPACE_DIR=workspace` creates a disk workspace with mode `0700`. Only a non-hidden relative project subdirectory is accepted, with no traversal or symlinked root. Virtual `/notes.txt` maps to `workspace/notes.txt`. Native `ls/read_file/write_file/edit_file/glob/grep/delete/execute/task` coexist with `write_todos`. The general-purpose subagent inherits the same backend, custom tools and execution rules.

`execute` interprets a small command allowlist without starting an OS shell or arbitrary subprocess: `pwd`, `ls [path]`, `cat file...`, `head/tail [-n 1..1000] file`, `wc [-l|-w|-c] file`, `mkdir [-p] path`, `touch file`, `echo text...`. Use file tools to write content. Scripts, interpreters, networking, pipes, redirection, chaining and expansion are rejected. Commands are limited to 4096 characters; reads to 1 MiB per file; output to 16000 characters with a truncation flag. Optional timeout values must be 1–30 seconds; these in-process file operations do not offer a forcibly interruptible process timeout.

This application boundary assumes one local user and a trusted local filesystem. It is not an OS sandbox or multi-tenant isolation. All conversations and subagents share persisted files. Do not store secrets or let untrusted local processes modify this directory concurrently; links and special files are unsupported. Writes/deletions are real and are not rolled back on cancellation, failure or retry. Clearing browser conversations does not remove workspace files. Arbitrary code execution requires a separately isolated backend; setting a local shell's cwd alone does not provide isolation ([official backend guidance](https://reference.langchain.com/python/deepagents/backends/local_shell)).

- `baidu_search(query, max_results=5)` makes server-side HTTPS requests only to Baidu and returns titles, summaries and URLs (possibly Baidu redirect URLs), up to 10 results. It does not fetch result pages. Captchas, errors, empty results and HTML changes are reported without another search engine fallback. No search API key is needed. `BAIDU_TIMEOUT_SECONDS=10` accepts `(0, 60]`; optional `BAIDU_HTTP_PROXY` sets an explicit HTTP(S) proxy, otherwise requests connect directly without system proxies. Redirects are not followed; response size is capped at 2 MiB. Queries go to Baidu/the proxy.
- `get_current_time(timezone="Asia/Shanghai")` uses the real clock and IANA timezone, returning ISO time, date and Monday–Sunday week boundaries. The prompt requires this tool for today/now/this-week questions.
- `calculate(expression)` uses a bounded numeric AST interpreter with no `eval`, imports, attribute access or system commands. It supports arithmetic, powers, parentheses, percentages, `pi/e/tau`, and `sqrt/sin/cos/tan/log/log10/log2/exp/floor/ceil/abs/round/min/max`. Trigonometry uses radians; `200*10%=20`, `200+10%=200.1`, and a 10% increase is `200*(1+10%)`. Limits cover expression length, AST nodes/depth, exponents and finite real results; floating-point math is not arbitrary-precision financial arithmetic.

Keys and optional proxy authentication remain server-side in `.env`; never store them in the workspace or tool arguments. File content/tool results enter the Qwen context and may reach LangSmith if tracing is enabled. Workspace files are excluded from Git and Docker build context. Docker pre-creates `/app/workspace` for its non-root user; use a dedicated named volume for persistence, never mount host secrets or home directories. The local system prompt's current capability and execution rules are appended after a pinned hosted prompt, so an old text-only prompt cannot disable the new capabilities. Restart after code/prompt changes; optionally republish with `scripts/sync_prompt.py`.

## Planning

Deep Agents supports structured planning through the opt-in `TodoListMiddleware`, which provides `write_todos`. For complex tasks, the assistant creates steps, updates their `pending`, `in_progress`, and `completed` states, and revises the plan as requirements change. The chat interface shows progress; simple questions receive a direct answer.

Try asking the assistant to prepare a project report by planning and completing goal analysis, progress review, risk assessment, and a draft. Follow up with new constraints to revise the remaining steps. Plans track real file operations, commands, searches and analysis; proposing an action or delegating it does not mean it has been completed.

Plans are saved with assistant messages in browser storage. Follow-up requests restore the latest valid plan from the same conversation. Interrupted plans remain available and display unfinished work as awaiting continuation. Retry starts with the plan preceding the failed response, and new conversations start empty. A plan supports up to 32 steps with at most 500 characters per step.

## API

| Route | Purpose |
| --- | --- |
| `GET /` | Chat interface |
| `GET /healthz` | Configuration status without credentials |
| `POST /api/chat` | NDJSON stream with `meta`, `plan`, `tool`, `delta`, `usage`, `done`, and `error` events |
| `GET /docs` | Interactive OpenAPI documentation |

Example request body:

```json
{
  "conversation_id": "871f1402-965f-4287-8fe5-f23a72d2bb16",
  "messages": [{"role": "user", "content": "Help me plan today's work"}]
}
```

For follow-up questions, include the current conversation's user and assistant messages. The last message must be from the user. Clients cannot supply system or tool roles. Missing credentials and invalid input use HTTP error codes; model failures after streaming begins are returned as `error` events without exposing upstream requests or keys.

The optional `todos` request field defaults to `[]`. Each item has `content` and a `status` of `pending`, `in_progress`, or `completed`. Successful tool updates emit complete plan snapshots:

```json
{"type":"plan","todos":[{"content":"Review goals","status":"in_progress"},{"content":"Draft report","status":"pending"}]}
```

Replace the previous snapshot on each `plan` event; an empty list clears it. Send the latest snapshot back as `todos` on subsequent requests in the same conversation. Plan events can be interleaved with text deltas; only `done` signals a completed response.

An additive `tool` progress event contains `name`, `call_id` and `status` (`running`, `finished`, `error`), without arguments or raw results. `finished` means the tool returned, including possible business errors; the final response must report the actual outcome. Existing clients may ignore these events. Parent-level `task` progress represents delegated work; the main assistant summarizes its verified result.

## Project structure

```text
personal_assistant_agent/
├── docs/个人助理原型.html       # Original standalone HTML design prototype
├── scripts/                    # One-time setup and live-service verification
├── tests/                      # API, agent graph, stream parsing, cancellation
├── src/personal_assistant_agent/
│   ├── agents/assistant.py     # Deep Agents + Qwen
│   ├── api/                    # Chat and health routes
│   ├── core/config.py          # Server configuration
│   ├── schemas/chat.py         # Input validation
│   ├── services/chat.py        # Streaming execution and LangSmith
│   ├── static/                # HTML, local CSS and JavaScript
│   ├── app.py                 # Application assembly and lifecycle
│   └── cli.py                 # Command-line entry point
├── README.md                  # Default Chinese documentation
├── README.en.md               # English documentation
├── .env.example
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Validation

Unit tests do not call a paid model. Model transport doubles are confined to tests and are absent from application runtime code. Dataset experiments and live-service verification scripts make billable Qwen calls.

```bash
uv run ruff check .
uv run pytest -q
node --test tests/*.mjs
```

After configuring Qwen and starting the server, run:

```bash
uv run python scripts/smoke_test.py
```

This command makes real model calls, checks a two-turn conversation and its context, and prints the request ID. If LangSmith tracing is enabled, it also checks the corresponding live trace. A successful live-model verification depends on the command's actual result.

Run `uv run python scripts/verify_tools.py --url http://127.0.0.1:8000` for paid, real-model checks of Planning, plan continuation, file operations, restricted commands, subagents, time, math, boundary rejection and Baidu search. Synthetic fixture files remain under `workspace/live-check-*/`; private reports and events use mode `0600` under `artifacts/live-tools/latest/`. Baidu requests use complete normal browser headers; captcha redirects produce an explicit `BAIDU_CHALLENGE` without following the challenge or fabricating results.

After resolving an external failure, add `--retry-failed` to rerun only failed checks while preserving earlier attempts. Suggested links following a failed search do not count as retrieved results. Structured tool errors appear as failed tool progress. Usage totals include measured calls from both the main agent and nested `task` agents.

## Deployment

The Dockerfile uses Python 3.13 and uv, and runs the application as a non-root user. These commands bind the service to the host's loopback interface:

```bash
docker build -t personal-assistant-agent .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 personal-assistant-agent
```

You can also install the dependencies and run the Python service directly on a server. The current version targets local, single-user use and includes no built-in login or server-side conversation persistence. Configure authentication, HTTPS, and access controls when deploying for public access.

## References

- [Deep Agents Python](https://docs.langchain.com/oss/python/deepagents/overview)
- [Tracing Deep Agents with LangSmith](https://docs.langchain.com/langsmith/trace-deep-agents)
- [Registering external agents in LangSmith](https://docs.langchain.com/langsmith/create-an-agent)
- [Alibaba Cloud Model Studio OpenAI-compatible API](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
