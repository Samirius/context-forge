"""FastAPI application factory with web UI and WebSocket support."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ctxf.api.routes import playbook, evolve, retrieve, feedback
from ctxf.config import get_settings


# Connected WebSocket clients
_ws_clients: list[WebSocket] = []


def create_app() -> FastAPI:
    app = FastAPI(
        title="context-forge",
        description="Self-hostable Agentic Context Engineering API",
        version="0.1.0",
    )

    # REST API routes
    app.include_router(playbook.router)
    app.include_router(evolve.router)
    app.include_router(retrieve.router)
    app.include_router(feedback.router)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # WebSocket for real-time updates
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        _ws_clients.append(ws)
        try:
            while True:
                # Keep alive, receive client messages
                data = await ws.receive_text()
                # Handle ping
                if data == "ping":
                    await ws.send_text("pong")
        except WebSocketDisconnect:
            _ws_clients.remove(ws)

    # Broadcast to all connected clients
    @app.post("/_broadcast")
    async def broadcast_update():
        """Internal: notify all WS clients of playbook update."""
        msg = json.dumps({"type": "playbook_updated"})
        disconnected = []
        for ws in _ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            _ws_clients.remove(ws)
        return {"notified": len(_ws_clients)}

    # Static files
    static_dir = Path(__file__).parent.parent / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Serve index.html for all non-API routes (SPA)
    @app.get("/")
    async def serve_index():
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "context-forge API running. Web UI not found."}

    return app
