"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from ctxf.api.routes import playbook, evolve, retrieve, feedback


def create_app() -> FastAPI:
    app = FastAPI(
        title="context-forge",
        description="Self-hostable Agentic Context Engineering API",
        version="0.1.0",
    )

    app.include_router(playbook.router, prefix="/v1", tags=["playbook"])
    app.include_router(evolve.router, prefix="/v1", tags=["evolve"])
    app.include_router(retrieve.router, prefix="/v1", tags=["retrieve"])
    app.include_router(feedback.router, prefix="/v1", tags=["feedback"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
