"""Submission and event subscription have independent lifetimes."""

import asyncio
import json
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from personal_assistant_agent.interaction.store import Conflict

router = APIRouter(prefix="/api", tags=["interaction"])


class MessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    client_message_id: UUID
    content: str = Field(min_length=1, max_length=8000)
    intent: Literal["CHAT", "NEW_TASK", "STEER", "STATUS", "PAUSE", "RESUME", "CANCEL"] | None = (
        None
    )
    target_task_id: UUID | None = None


class ControlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    control_id: UUID
    type: Literal["STEER", "PAUSE", "RESUME", "CANCEL"]
    content: str = Field(default="", max_length=8000)
    expected_revision: int | None = Field(default=None, ge=0)


def runtime(request):
    service = getattr(request.app.state, "interaction", None)
    if service is None:
        raise HTTPException(503, "任务服务尚未配置。")
    return service


def translate(error):
    if isinstance(error, KeyError):
        return HTTPException(404, "任务不属于当前会话或不存在。")
    if isinstance(error, Conflict):
        return HTTPException(409, str(error))
    return HTTPException(503, "消息未被接收，请稍后重试。")


@router.post("/conversations/{conversation}/messages", status_code=202)
async def submit(conversation: UUID, payload: MessageInput, request: Request):
    try:
        return await runtime(request).submit(str(conversation), payload.model_dump(mode="json"))
    except Exception as error:
        raise translate(error) from None


@router.get("/conversations/{conversation}/tasks")
async def snapshot(conversation: UUID, request: Request):
    service = runtime(request)
    return {
        **service.store.snapshot(str(conversation)),
        "runtime": {
            "persistence_fault": service.persistence_fault,
            "task_concurrency": service.settings.task_concurrency,
            "active_tasks": len(service.running),
        },
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: UUID, conversation_id: UUID, request: Request):
    try:
        service = runtime(request)
        task = service.store.task(str(task_id), str(conversation_id))
        return {**task, "controls": service.store.controls(str(task_id))}
    except KeyError as error:
        raise translate(error) from None


@router.post("/tasks/{task_id}/controls", status_code=202)
async def control_task(task_id: UUID, payload: ControlInput, request: Request):
    service = runtime(request)
    try:
        if service.persistence_fault:
            raise RuntimeError("任务持久化不可用。")
        changes = {}
        if payload.type == "STEER":
            # Same natural-language merge as the message endpoint.
            task = service.store.task(str(task_id), str(payload.conversation_id))
            decision = await service.router.route(
                payload.content, [task], [], "STEER", str(task_id)
            )
            if decision["intent"] != "STEER" or not decision.get("changes"):
                raise Conflict("修正条件不明确，请澄清后重试。")
            changes = decision["changes"]
        result = service.store.control(
            str(task_id),
            str(payload.conversation_id),
            str(payload.control_id),
            payload.type,
            payload.content,
            changes,
            payload.expected_revision,
        )
        service.wakeup.set()
        return result
    except Exception as error:
        raise translate(error) from None


@router.post("/conversations/{conversation}/chats/{message_id}/stop", status_code=202)
async def stop_chat(conversation: UUID, message_id: UUID, request: Request):
    try:
        runtime(request).stop_chat(str(conversation), str(message_id))
        return {"status": "RECEIVED"}
    except Conflict as error:
        raise translate(error) from None


@router.get("/conversations/{conversation}/events")
async def events(conversation: UUID, request: Request, after: int = 0):
    service = runtime(request)
    conversation = str(conversation)
    cursor = request.headers.get("last-event-id", "")
    if cursor:
        try:
            owner, number = cursor.rsplit(":", 1)
            if owner != conversation:
                raise ValueError
            after = max(after, int(number))
        except ValueError:
            raise HTTPException(400, "事件游标无效。") from None
    if after < 0:
        raise HTTPException(400, "事件游标无效。")

    async def stream():
        position = after
        ticks = 0
        while not service.stopping and not await request.is_disconnected():
            batch = service.store.events(conversation, position)
            for event in batch:
                position = event["sequence"]
                yield f"id: {event['event_id']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if not batch:
                ticks += 1
                if ticks % 40 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
