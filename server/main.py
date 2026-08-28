"""
ArthaSaathi API.

A thin HTTP layer over `ml/src`. Routes validate, delegate and serialise -- none
of them contain agent logic, because all of it already exists and a second copy
in a route handler would drift from the tested one.

Run:
    uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.src.common import config, llm
from ml.src.memory import embeddings
from ml.src.memory.store import get_store
from ml.src.workflows.base import AGENT_ORDER
from ml.src.workflows.catalogue import WORKFLOWS

from .deps import init_store, store_kind
from .routes import cards, chat, memory, profile, voice, workflows
from .schemas import HealthResponse

log = logging.getLogger("arthasaathi.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = init_store()
    log.info("memory store: %s", type(store).__name__)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArthaSaathi API",
        version="1.0.0",
        description=(
            "Multi-agent financial intelligence: 5 councils, 18 agents, a "
            "decision layer and an embedded memory over Neon Postgres."
        ),
        lifespan=lifespan,
    )

    # The frontend is served from a different origin during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """
        Last-resort guard.

        Individual routes already degrade on dependency failures; this exists so
        that a genuinely unexpected error still returns structured JSON a client
        can render, rather than an HTML stack trace.
        """
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal error",
                "path": request.url.path,
                "errors": [repr(exc)],
            },
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> dict[str, Any]:
        """
        Liveness plus what is actually wired up.

        Reports only whether each credential is present -- never a value.
        """
        database: dict[str, Any] = {
            "configured": bool(config.DATABASE_URL),
            "store": store_kind(),
        }
        if config.DATABASE_URL:
            try:
                get_store().recent("__healthcheck__", limit=1)
                database["reachable"] = True
            except Exception as exc:
                database["reachable"] = False
                database["error"] = repr(exc)

        return {
            "status": "ok",
            "database": database,
            "providers": llm.available_providers(),
            "embedding_backend": embeddings.active_backend(),
            "workflows": len(WORKFLOWS),
            "agents": len(AGENT_ORDER),
        }

    @app.get("/", tags=["meta"])
    def index() -> dict[str, Any]:
        # Read the endpoint list off the OpenAPI schema rather than `app.routes`:
        # included routers appear there as opaque wrapper objects with no `.path`,
        # so walking `app.routes` silently reports only the built-in endpoints.
        return {
            "name": "ArthaSaathi API",
            "docs": "/docs",
            "health": "/health",
            "endpoints": sorted(app.openapi().get("paths", {})),
        }

    app.include_router(chat.router)
    app.include_router(profile.router)
    app.include_router(workflows.router)
    app.include_router(memory.router)
    app.include_router(cards.router)
    app.include_router(voice.router)

    return app


app = create_app()
