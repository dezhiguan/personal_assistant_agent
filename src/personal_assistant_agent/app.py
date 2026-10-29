"""Application assembly; routes and agent execution live in separate modules."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from personal_assistant_agent.agents.assistant import build_agent
from personal_assistant_agent.api.chat import router as chat_router
from personal_assistant_agent.api.health import router as health_router
from personal_assistant_agent.core.config import Settings, get_settings
from personal_assistant_agent.services.chat import ChatService

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        def initialize():
            import truststore

            truststore.inject_into_ssl()
            return ChatService(build_agent(settings), settings)

        service = await asyncio.to_thread(initialize) if settings.model_configured else None
        app.state.settings = settings
        app.state.chat_service = service
        yield
        if service:
            await service.close()

    app = FastAPI(title="个人智能助理", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(chat_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
