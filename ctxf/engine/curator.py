"""Curator — converts reflections into delta operations for playbook evolution."""

from __future__ import annotations

from ctxf.llm.base import LLMProvider, Message
from ctxf.engine.reflector import Reflection
from ctxf.models.delta import Delta


CURATOR_SYSTEM_PROMPT = """You are a context engineering curator. Your job is to take a structured \
reflection about a task execution and produce concrete delta operations to evolve the playbook.

The playbook has these sections: {sections}

You can produce these delta operations:
- ADD: Add a new bullet to a section. Fields: op="ADD", section, content
- PATCH: Update an existing bullet's content. Fields: op="PATCH", bullet_id, new_content
- INCR: Increment a bullet's helpful or harmful counter. Fields: op="INCR", bullet_id, field="helpful"|"harmful", amount=1
- DEPRECATE: Soft-delete a bullet. Fields: op="DEPRECATE", bullet_id
- MERGE: Combine multiple similar bullets. Fields: op="MERGE", bullet_ids=["id1","id2"], content="merged content"

Rules:
1. Only ADD bullets that represent genuine, reusable knowledge.
2. PATCH bullets when the reflection suggests a specific correction.
3. INCR helpful/harmful based on explicit feedback.
4. DEPRECATE bullets that are confirmed outdated or wrong.
5. MERGE bullets that are near-duplicates saying the same thing differently.
6. Each delta should have a "reason" field explaining why.

Respond with a JSON array of delta operations:
[
    {{"op": "ADD", "section": "section_name", "content": "bullet content", "reason": "why"}},
    ...
]"""


class Curator:
    """Converts reflections into delta operations."""

    def __init__(self, llm: LLMProvider, sections: list[str] | None = None) -> None:
        self.llm = llm
        self.sections = sections or ["governance", "technical", "style", "domain", "workflow"]

    async def curate(self, reflection: Reflection) -> list[Delta]:
        """Produce delta operations from a reflection."""
        system = CURATOR_SYSTEM_PROMPT.format(sections=", ".join(self.sections))

        reflection_json = reflection.to_dict()
        import json
        user_content = f"## Reflection\n{json.dumps(reflection_json, indent=2)}"

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user_content),
        ]

        try:
            result = await self.llm.complete_json(messages)
            if isinstance(result, dict):
                # Might be wrapped in a key
                deltas_raw = result.get("deltas", result.get("operations", [result]))
            elif isinstance(result, list):
                deltas_raw = result
            else:
                deltas_raw = []
        except Exception:
            deltas_raw = []

        deltas = []
        for d in deltas_raw:
            if isinstance(d, dict) and "op" in d:
                try:
                    deltas.append(Delta.from_dict(d))
                except Exception:
                    continue

        return deltas

    async def curate_simple(
        self,
        task: str,
        response: str,
        outcome: str = "",
        helpful_ids: list[str] | None = None,
        harmful_ids: list[str] | None = None,
    ) -> list[Delta]:
        """Convenience: create deltas directly from task/response/outcome without a separate reflection step."""
        deltas = []

        # Increment helpful/harmful counters
        for bid in (helpful_ids or []):
            deltas.append(Delta.incr(bid, "helpful", reason=f"helpful for task: {task[:100]}"))
        for bid in (harmful_ids or []):
            deltas.append(Delta.incr(bid, "harmful", reason=f"harmful for task: {task[:100]}"))

        # If outcome is provided, also run the full curation
        if outcome and self.llm:
            from ctxf.engine.reflector import Reflection
            reflection = Reflection(
                summary=f"Task: {task[:200]}\nOutcome: {outcome[:200]}",
                what_failed=[outcome] if "fail" in outcome.lower() or "bad" in outcome.lower() else [],
                what_worked=[outcome] if "success" in outcome.lower() or "good" in outcome.lower() else [],
            )
            deltas.extend(await self.curate(reflection))

        return deltas
