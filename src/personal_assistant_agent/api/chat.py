"""POST streaming chat endpoint."""

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from personal_assistant_agent.schemas.chat import ChatRequest

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    service = request.app.state.chat_service
    if service is None:
        raise HTTPException(status_code=503, detail="Qwen 尚未配置，请检查服务端 .env。")
    request_id = str(uuid4())

    async def events():
        async for event in service.stream(payload, request_id):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )
