"""Retrieve route — POST /v1/retrieve."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ctxf.config import get_settings
from ctxf.retrieval.embedder import Embedder
from ctxf.retrieval.hybrid import HybridRetriever
from ctxf.store.sqlite_store import SqliteStore

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    playbook_id: Optional[str] = None
    top_k: Optional[int] = None


class ScoredBullet(BaseModel):
    id: str
    section: str
    content: str
    helpful: int
    harmful: int
    score: float


class RetrieveResponse(BaseModel):
    results: list[ScoredBullet]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Search playbook for relevant bullets."""
    settings = get_settings()
    store = SqliteStore(settings.db_path)

    try:
        pb = store.get_playbook(req.playbook_id)
        if not pb:
            raise HTTPException(status_code=404, detail="Playbook not found")

        embedder = None
        if settings.embeddings_enabled:
            embedder = Embedder(
                base_url=settings.effective_embed_base_url,
                api_key=settings.effective_embed_api_key,
                model=settings.embed_model,
                dim=settings.embed_dim,
            )

        retriever = HybridRetriever(
            embedder=embedder,
            top_k=req.top_k or settings.retrieval_top_k,
            alpha=settings.hybrid_alpha,
        )

        results = await retriever.retrieve(req.query, pb.active_bullets())

        scored = []
        for bullet, score in results:
            scored.append(ScoredBullet(
                id=bullet.id,
                section=bullet.section,
                content=bullet.content,
                helpful=bullet.helpful,
                harmful=bullet.harmful,
                score=round(score, 4),
            ))

        return RetrieveResponse(results=scored)
    finally:
        store.close()
