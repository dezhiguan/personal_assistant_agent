# Personal Assistant Agent · 个人智能助理

[简体中文](README.md) | **English**

A standalone personal assistant built with **Python, Deep Agents, Qwen, and LangSmith**. FastAPI serves a single-page chat interface and its API from the same origin. The backend calls Alibaba Cloud Model Studio directly; conversation history stays in the current browser, with optional LangSmith tracing.

## Features

- Streaming text conversations with follow-up questions, stop, retry, and copy controls.
- Multiple conversations with history restored from browser local storage.
- A Python / FastAPI backend serving the interface and API together.
- Direct Qwen integration with optional LangSmith tracing.
- Command-line startup and a Dockerfile for deployment.

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

Keep API keys on the server. Git and Docker builds exclude `.env`; never put keys in HTML, JavaScript, command arguments, or documentation. Keep the local configuration file at permission mode `0600`.

## Conversation behavior and limits

- Send a message or choose a suggested prompt to receive a streamed response.
- Follow-up requests explicitly include the current conversation history.
- New and existing conversations remain separate; refreshing restores local history.
- Stopping generation aborts the browser request and cancels the server task.
- Failed responses can be regenerated without adding a duplicate user message.
- Copy answers or clear local history on desktop and mobile.

This MVP uses one text assistant. File access, shell execution, web access, and subagent tools are disabled. There is no shared checkpointer, and the backend does not persist conversation history between requests. Each message is limited to 8,000 characters; a request can contain at most 60 messages and 80,000 characters in total. Oversized conversations are rejected with a prompt to start a new conversation rather than silently truncated.

## API

| Route | Purpose |
| --- | --- |
| `GET /` | Chat interface |
| `GET /healthz` | Configuration status without credentials |
| `POST /api/chat` | NDJSON stream with `meta`, `delta`, `done`, and `error` events |
| `GET /docs` | Interactive OpenAPI documentation |

Example request body:

```json
{
  "conversation_id": "871f1402-965f-4287-8fe5-f23a72d2bb16",
  "messages": [{"role": "user", "content": "Help me plan today's work"}]
}
```

For follow-up questions, include the current conversation's user and assistant messages. The last message must be from the user. Clients cannot supply system or tool roles. Missing credentials and invalid input use HTTP error codes; model failures after streaming begins are returned as `error` events without exposing upstream requests or keys.

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

Automated tests do not call a paid model. Model transport doubles are confined to tests and are absent from application runtime code.

```bash
uv run ruff check .
uv run pytest -q
node --test tests/test_stream.mjs
```

After configuring Qwen and starting the server, run:

```bash
uv run python scripts/smoke_test.py
```

This command makes real model calls, checks a two-turn conversation and its context, and prints the request ID. If LangSmith tracing is enabled, it also checks the corresponding live trace. A successful live-model verification depends on the command's actual result.

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
