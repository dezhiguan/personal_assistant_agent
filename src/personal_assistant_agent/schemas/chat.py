"""Bounded conversation input; clients cannot supply system or tool messages."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("消息不能为空")
        return value


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=500, pattern=r"\S")
    status: Literal["pending", "in_progress", "completed"]


TodoList = Annotated[list[TodoItem], Field(max_length=32)]


class ToolProgress(BaseModel):
    """Additive NDJSON event. Contains no arguments, tool output or credentials."""

    type: Literal["tool"] = "tool"
    name: str
    call_id: str
    status: Literal["running", "finished", "error"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    messages: list[ChatMessage] = Field(min_length=1, max_length=60)
    todos: TodoList = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conversation(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("最后一条消息必须是用户消息")
        if sum(len(m.content) for m in self.messages) > 80000:
            raise ValueError("对话过长，请开启新对话")
        return self
