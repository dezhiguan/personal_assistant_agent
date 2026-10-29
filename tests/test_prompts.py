"""Hosted prompt loading is immutable, explicit and contains no model credentials."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from personal_assistant_agent.core.config import Settings
from personal_assistant_agent.core.envfile import update_env
from personal_assistant_agent.services.prompts import load_prompt


def test_local_prompt_retains_identity_and_planning():
    value = load_prompt(Settings())
    assert "个人智能助理" in value.text
    assert "write_todos" in value.text
    assert "不得用新项替换或删除已完成项" in value.text
    assert value.reference.startswith("local:")


def test_hosted_prompt_version_is_explicit_and_read_without_model():
    client = Mock()
    client.pull_prompt.return_value = ChatPromptTemplate.from_messages(
        [SystemMessage(content="test prompt")]
    )
    settings = Settings(prompt_ref="test-prompt:" + "a" * 64, langsmith_api_key="secret")
    loaded = load_prompt(settings, client)
    assert loaded.text == "test prompt"
    assert loaded.reference == settings.prompt_ref
    client.pull_prompt.assert_called_once_with(settings.prompt_ref, include_model=False)
    client.close.assert_not_called()
    with pytest.raises(ValueError):
        load_prompt(replace(settings, prompt_ref="test-prompt:latest"), client)


def test_hosted_prompt_rejects_user_templates():
    client = Mock()
    client.pull_prompt.return_value = ChatPromptTemplate.from_template("User says {message}")
    with pytest.raises(ValueError):
        load_prompt(
            Settings(prompt_ref="test-prompt:" + "a" * 64, langsmith_api_key="secret"), client
        )


def test_env_update_preserves_secrets_and_permissions(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DASHSCOPE_API_KEY=do-not-print\nLANGSMITH_PROMPT_REF=old\n")
    update_env({"LANGSMITH_PROMPT_REF": "test:" + "a" * 64}, path)
    assert "DASHSCOPE_API_KEY=do-not-print" in path.read_text()
    assert path.read_text().count("LANGSMITH_PROMPT_REF=") == 1
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        update_env({"NEW": "invalid\nvalue"}, path)
