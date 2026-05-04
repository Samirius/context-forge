"""Playbook routes — GET /v1/playbook, GET /v1/playbook/{id}."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ctxf.config import get_settings
from ctxf.store.sqlite_store import SqliteStore

router = APIRouter()


def _get_store() -> SqliteStore:
    settings = get_settings()
    return SqliteStore(settings.db_path)


class PlaybookResponse(BaseModel):
    id: str
    name: str
    version: int
    sections: list[str]
    bullets: list[dict]
    active_count: int
    total_count: int


class PlaybookListResponse(BaseModel):
    playbooks: list[dict]


@router.get("/playbook", response_model=PlaybookListResponse)
async def list_playbooks():
    """List all playbooks."""
    store = _get_store()
    try:
        playbooks = store.list_playbooks()
        return PlaybookListResponse(playbooks=playbooks)
    finally:
        store.close()


@router.get("/playbook/{playbook_id}", response_model=PlaybookResponse)
async def get_playbook(playbook_id: str):
    """Get a specific playbook with its bullets."""
    store = _get_store()
    try:
        pb = store.get_playbook(playbook_id)
        if not pb:
            raise HTTPException(status_code=404, detail="Playbook not found")
        d = pb.to_dict()
        return PlaybookResponse(**d)
    finally:
        store.close()


@router.post("/playbook", response_model=PlaybookResponse)
async def create_playbook(name: str = "default"):
    """Create a new empty playbook."""
    store = _get_store()
    try:
        pb = store.create_playbook(name=name)
        d = pb.to_dict()
        return PlaybookResponse(**d)
    finally:
        store.close()


@router.get("/playbook/{playbook_id}/stats")
async def get_stats(playbook_id: str):
    """Get playbook statistics."""
    store = _get_store()
    try:
        return store.get_stats(playbook_id)
    finally:
        store.close()


@router.get("/playbook/{playbook_id}/history")
async def get_history(playbook_id: str, limit: int = 20):
    """Get version history for a playbook."""
    store = _get_store()
    try:
        return {"history": store.get_version_history(playbook_id, limit)}
    finally:
        store.close()


@router.get("/playbook/{playbook_id}/deltas")
async def get_deltas(playbook_id: str, limit: int = 50):
    """Get delta log for a playbook."""
    store = _get_store()
    try:
        return {"deltas": store.get_delta_log(playbook_id, limit)}
    finally:
        store.close()
