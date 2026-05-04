"""Feedback route — POST /v1/feedback."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ctxf.config import get_settings
from ctxf.models.delta import Delta
from ctxf.store.sqlite_store import SqliteStore

router = APIRouter()


class FeedbackRequest(BaseModel):
    playbook_id: Optional[str] = None
    helpful_bullet_ids: list[str] = []
    harmful_bullet_ids: list[str] = []
    new_insights: list[dict] = []  # [{"section": "...", "content": "..."}]
    deprecate_ids: list[str] = []
    reason: str = ""


class FeedbackResponse(BaseModel):
    applied: int
    deltas: list[dict]


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """Apply feedback as delta operations."""
    settings = get_settings()
    store = SqliteStore(settings.db_path)

    try:
        pb = store.get_playbook(req.playbook_id)
        if not pb:
            raise HTTPException(status_code=404, detail="Playbook not found")

        deltas: list[Delta] = []

        # Helpful
        for bid in req.helpful_bullet_ids:
            deltas.append(Delta.incr(bid, "helpful", reason=req.reason))

        # Harmful
        for bid in req.harmful_bullet_ids:
            deltas.append(Delta.incr(bid, "harmful", reason=req.reason))

        # New insights
        for insight in req.new_insights:
            section = insight.get("section", "general")
            content = insight.get("content", "")
            if content:
                deltas.append(Delta.add(section, content, reason=req.reason))

        # Deprecate
        for bid in req.deprecate_ids:
            deltas.append(Delta.deprecate(bid, reason=req.reason))

        # Apply
        store.apply_deltas(pb.id, deltas)

        return FeedbackResponse(
            applied=len(deltas),
            deltas=[d.to_dict() for d in deltas],
        )
    finally:
        store.close()
