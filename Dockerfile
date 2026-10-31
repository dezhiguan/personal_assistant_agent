FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev && useradd --create-home assistant \
    && mkdir -p /app/workspace /app/.assistant-private \
    && chown assistant:assistant /app/workspace /app/.assistant-private \
    && chmod 700 /app/workspace /app/.assistant-private
USER assistant
EXPOSE 8000
CMD [".venv/bin/personal-assistant-agent", "--host", "0.0.0.0", "--port", "8000"]
