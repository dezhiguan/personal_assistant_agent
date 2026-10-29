"""Publish a private prompt and pin the returned immutable commit in .env."""

import argparse
from dataclasses import replace

import truststore
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from personal_assistant_agent.core.config import get_settings
from personal_assistant_agent.core.envfile import update_env
from personal_assistant_agent.services.prompts import LOCAL_PROMPT, load_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="personal-assistant-system")
    args = parser.parse_args()
    truststore.inject_into_ssl()
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise SystemExit("LANGSMITH_API_KEY is required")
    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    try:
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=LOCAL_PROMPT.read_text().strip())]
        )
        url = client.push_prompt(
            args.name,
            object=prompt,
            is_public=False,
            description="个人智能助理的版本化系统提示词；不包含模型或密钥。",
            commit_description="Publish local system prompt for the personal assistant",
        )
        pulled = client.pull_prompt(args.name, include_model=False, skip_cache=True)
        commit = (pulled.metadata or {}).get("lc_hub_commit_hash")
        if not commit:
            raise RuntimeError("LangSmith did not return a prompt commit hash")
        reference = args.name + ":" + commit
        version = load_prompt(replace(settings, prompt_ref=reference), client)
        if version.text != LOCAL_PROMPT.read_text().strip():
            raise RuntimeError("Published and loaded prompt content differ")
        update_env({"LANGSMITH_PROMPT_REF": reference})
        print("Private prompt published and read-back verified:", reference)
        print("Prompt URL:", url)
    except Exception as error:
        raise SystemExit(f"Prompt synchronization failed: {type(error).__name__}") from None
    finally:
        client.close()


if __name__ == "__main__":
    main()
