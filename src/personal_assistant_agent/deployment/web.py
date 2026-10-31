"""Expose chat routes without shadowing Agent Server APIs.

A Starlette adapter also avoids FastAPI's lazy included-router objects, which
Agent Server 0.16 cannot yet wrap with custom-route authentication middleware.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from personal_assistant_agent.app import STATIC_DIR
from personal_assistant_agent.app import app as chat_app


def create_deployment_app(chat: FastAPI) -> Starlette:
    @asynccontextmanager
    async def lifespan(app):
        async with chat.router.lifespan_context(chat):
            yield

    return Starlette(
        routes=[
            Route("/", endpoint=chat, methods=["GET"]),
            Route("/api/chat", endpoint=chat, methods=["POST"]),
            Route("/api/calendar/{event_id}.ics", endpoint=chat, methods=["GET"]),
            Route("/healthz", endpoint=chat, methods=["GET"]),
            Mount("/static", app=StaticFiles(directory=STATIC_DIR)),
        ],
        lifespan=lifespan,
    )


app = create_deployment_app(chat_app)
