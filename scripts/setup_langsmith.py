"""Register this external agent once and save its tracing destination securely."""

import argparse
import os
import tempfile
import time
from pathlib import Path

import truststore
from langsmith import Client, agent_address, traceable, tracing_context

from personal_assistant_agent.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", default="personal-assistant-agent")
    parser.add_argument(
        "--environment",
        choices=["local", "development", "staging", "production"],
        default="development",
    )
    args = parser.parse_args()
    truststore.inject_into_ssl()
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise SystemExit("请先在 .env 中配置 LANGSMITH_API_KEY。")
    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    try:
        try:
            agent = client.agents.retrieve(args.agent_id)
        except Exception as error:
            if getattr(error, "status_code", None) != 404:
                raise
            agent = client.agents.create(
                id=args.agent_id, name="个人智能助理", source_type="EXTERNAL"
            )
        env = next(
            (e for e in agent.environments if e.environment.lower() == args.environment), None
        )
        if env is None:
            # Older automatically created records can have Production alone.
            # Addressing a setup trace materializes the requested environment.
            @traceable(
                client=client,
                address=agent_address(agent.id, args.environment),
                name="langsmith-environment-setup",
                run_type="chain",
            )
            def check_environment():
                return {"configured": True, "model_called": False}

            with tracing_context(enabled=True):
                check_environment()
            client.flush()
            for _ in range(5):
                agent = client.agents.retrieve(args.agent_id)
                env = next(
                    (e for e in agent.environments if e.environment.lower() == args.environment),
                    None,
                )
                if env is not None:
                    break
                time.sleep(1)
            if env is None:
                raise RuntimeError("Requested LangSmith environment is not ready")
        path = Path(os.getenv("ASSISTANT_ENV_FILE", ".env"))
        content = path.read_text() if path.exists() else ""
        # LangChainTracer currently accepts project_name; use the environment's
        # compatibility name, not an independently invented project destination.
        lines = [
            line
            for line in content.splitlines()
            if not line.startswith(
                (
                    "LANGSMITH_PROJECT=",
                    "LANGSMITH_AGENT_ID=",
                    "LANGSMITH_AGENT_ENVIRONMENT=",
                    "LANGCHAIN_PROJECT=",
                    "LANGCHAIN_SESSION=",
                )
            )
        ]
        lines.append("LANGSMITH_PROJECT=" + env.tracer_session_name)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            output.write("\n".join(lines) + "\n")
            temporary = Path(output.name)
        temporary.chmod(0o600)
        temporary.replace(path)
        print(f"LangSmith agent ready: {agent.id} / {args.environment}")
        print(f"Tracing environment ID: {env.tracer_session_id}")
    except Exception as error:
        raise SystemExit(f"LangSmith setup failed: {type(error).__name__}") from None
    finally:
        client.close()


if __name__ == "__main__":
    main()
