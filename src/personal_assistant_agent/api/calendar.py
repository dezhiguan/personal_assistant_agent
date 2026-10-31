"""Download generated calendar files without exposing arbitrary workspace paths."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from personal_assistant_agent.agents.workspace import WorkspaceBackend

router = APIRouter()


@router.get("/api/calendar/{event_id}.ics")
def download_calendar(event_id: str, request: Request) -> Response:
    import re

    if not re.fullmatch(r"[a-f0-9]{32}", event_id):
        raise HTTPException(404, "日历文件不存在。")
    try:
        backend = WorkspaceBackend(request.app.state.settings.workspace_dir)
        content = backend._text(f"/calendar/{event_id}.ics")
    except (OSError, ValueError):
        raise HTTPException(404, "日历文件不存在或无法读取。") from None
    return Response(content, media_type="text/calendar; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="{event_id}.ics"',
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    })
