"""Evolve route — POST /v1/evolve (full ACE loop)."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ctxf.config import get_settings
from ctxf.llm.factory import create_llm
from ctxf.retrieval.embedder import Embedder
from ctxf.retrieval.hybrid import HybridRetriever
from ctxf.engine.generator import Generator
from ctxf.engine.reflector import Reflector
from ctxf.engine.curator import Curator
from ctxf.store.sqlite_store import SqliteStore

router = APIRouter()


class EvolveRequest(BaseModel):
    task: str
    playbook_id: Optional[str] = None
    outcome: Optional[str] = None
    auto_apply: bool = True


class EvolveResponse(BaseModel):
    response: str
    reflection: Optional[dict] = None
    deltas: list[dict] = []
    applied: bool = False
    error: Optional[str] = None


@router.post("/evolve", response_model=EvolveResponse)
async def evolve(req: EvolveRequest):
    """Run the full ACE loop: retrieve → generate → reflect → curate → apply."""
    settings = get_settings()
    store = SqliteStore(settings.db_path)

    try:
        # 1. Get playbook
        pb = store.get_playbook(req.playbook_id)
        if not pb:
            raise HTTPException(status_code=404, detail="Playbook not found")

        # 2. Retrieve relevant bullets
        embedder = Embedder(
            base_url=settings.effective_embed_base_url,
            api_key=settings.effective_embed_api_key,
            model=settings.embed_model,
            dim=settings.embed_dim,
        ) if settings.embeddings_enabled else None

        retriever = HybridRetriever(
            embedder=embedder,
            top_k=settings.retrieval_top_k,
            alpha=settings.hybrid_alpha,
        )
        results = await retriever.retrieve(req.task, pb.active_bullets())
        retrieved_bullets = [b for b, _ in results]

        # 3. Generate response
        gen_llm = create_llm(settings, "generator")
        generator = Generator(gen_llm)
        gen_result = await generator.execute(req.task, retrieved_bullets)

        # 4. Reflect
        refl_llm = create_llm(settings, "reflector")
        reflector = Reflector(refl_llm)
        reflection = await reflector.reflect(
            task=req.task,
            response=gen_result.response,
            outcome=req.outcome,
            bullets_used=[b.to_dict() for b in retrieved_bullets],
        )

        # 5. Curate
        cur_llm = create_llm(settings, "curator")
        curator = Curator(cur_llm, sections=pb.sections)
        deltas = await curator.curate(reflection)

        # 6. Apply deltas
        applied = False
        if req.auto_apply and deltas:
            store.apply_deltas(pb.id, deltas)
            applied = True

        return EvolveResponse(
            response=gen_result.response,
            reflection=reflection.to_dict(),
            deltas=[d.to_dict() for d in deltas],
            applied=applied,
        )
    except HTTPException:
        raise
    except Exception as e:
        return EvolveResponse(
            response="",
            error=str(e),
        )
    finally:
        store.close()
