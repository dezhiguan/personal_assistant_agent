"""Safe runtime status without credentials."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["status"])


@router.get("/healthz")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "model_configured": settings.model_configured,
        "model": settings.model_name,
        "provider": "Qwen",
        "framework": "Deep Agents",
        "tracing_enabled": settings.tracing_configured,
        "tracing_requested": settings.tracing_enabled,
        "langsmith_project": settings.langsmith_project if settings.tracing_configured else None,
    }
