"""Resolve a reply's private LangSmith URL without exposing service credentials."""

import asyncio
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from langsmith.utils import LangSmithNotFoundError

router = APIRouter(prefix="/api", tags=["tracing"])


@router.get("/traces/{request_id}")
async def open_trace(request_id: UUID, request: Request) -> RedirectResponse:
    service = request.app.state.chat_service
    client = service.trace_client if service else None
    if client is None:
        raise HTTPException(status_code=503, detail="当前服务未开启 LangSmith 追踪。")
    try:
        async with asyncio.timeout(15):
            # Resolve the run's own project, including replies saved before a
            # configuration change. Only fetch a URL when the link is opened.
            run = await asyncio.to_thread(client.read_run, request_id)
            runs = await asyncio.to_thread(lambda: client.runs)
            result = await runs.get_url(
                str(run.id), project_id=str(run.session_id), trace_id=str(run.trace_id)
            )
            url = result.url
            parsed = urlsplit(url or "")
            if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                raise ValueError("Missing LangSmith run URL")
    except LangSmithNotFoundError:
        raise HTTPException(
            status_code=404, detail="尚未找到这条回复的追踪记录，请稍后重试。"
        ) from None
    except Exception as error:
        if getattr(error, "status_code", None) == 404:
            raise HTTPException(
                status_code=404, detail="尚未找到这条回复的追踪记录，请稍后重试。"
            ) from None
        raise HTTPException(
            status_code=503, detail="暂时无法打开 LangSmith 追踪，请稍后重试。"
        ) from None
    return RedirectResponse(
        url=url,
        status_code=303,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
