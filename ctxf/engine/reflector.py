"""Reflector — analyzes task outcome and extracts structured insights."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ctxf.llm.base import LLMProvider, Message


REFLECTOR_SYSTEM_PROMPT = """You are a context engineering reflector. Your job is to analyze a completed \
AI task execution and extract insights about what worked, what didn't, and what should be added to \
or changed in the playbook (context document).

The playbook contains "bullets" — concise statements of knowledge, preferences, rules, or patterns \
that guide future task executions.

Analyze the task, the response, and the outcome. Produce a structured reflection with:
1. What went well (which existing bullets helped)
2. What went wrong (which bullets were harmful or missing)
3. New insights (potential new bullets to add)
4. Outdated info (bullets that should be updated or removed)

Respond in JSON format:
{
    "summary": "brief summary of the reflection",
    "what_worked": ["description of what worked"],
    "what_failed": ["description of what went wrong"],
    "new_insights": [
        {"section": "section_name", "content": "insight content", "confidence": 0.8}
    ],
    "outdated": [
        {"bullet_id": "ID", "reason": "why it's outdated", "suggested_fix": "new content or null"}
    ],
    "helpful_bullet_ids": ["bullet ids that helped"],
    "harmful_bullet_ids": ["bullet ids that hurt"]
}"""


@dataclass
class Reflection:
    summary: str
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    new_insights: list[dict] = field(default_factory=list)
    outdated: list[dict] = field(default_factory=list)
    helpful_bullet_ids: list[str] = field(default_factory=list)
    harmful_bullet_ids: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "new_insights": self.new_insights,
            "outdated": self.outdated,
            "helpful_bullet_ids": self.helpful_bullet_ids,
            "harmful_bullet_ids": self.harmful_bullet_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Reflection":
        return cls(
            summary=data.get("summary", ""),
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            new_insights=data.get("new_insights", []),
            outdated=data.get("outdated", []),
            helpful_bullet_ids=data.get("helpful_bullet_ids", []),
            harmful_bullet_ids=data.get("harmful_bullet_ids", []),
            raw=data,
        )


class Reflector:
    """Analyzes task outcomes and produces structured reflections."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def reflect(
        self,
        task: str,
        response: str,
        outcome: Optional[str] = None,
        bullets_used: Optional[list[dict]] = None,
    ) -> Reflection:
        """Reflect on a task execution and extract insights.

        Args:
            task: The original task description.
            response: The AI's response/output.
            outcome: Optional human-provided outcome feedback (success/failure details).
            bullets_used: List of bullets that were injected, with their content.
        """
        bullets_text = ""
        if bullets_used:
            lines = []
            for b in bullets_used:
                lines.append(f"[{b.get('id', '?')}] ({b.get('section', '?')}) {b.get('content', '')}")
            bullets_text = "\n".join(lines)

        user_parts = [
            f"## Task\n{task}",
            f"\n## Response\n{response}",
        ]
        if outcome:
            user_parts.append(f"\n## Outcome Feedback\n{outcome}")
        if bullets_text:
            user_parts.append(f"\n## Playbook Bullets Used\n{bullets_text}")

        user_content = "\n".join(user_parts)

        messages = [
            Message(role="system", content=REFLECTOR_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        try:
            result = await self.llm.complete_json(messages)
            return Reflection.from_dict(result)
        except Exception:
            # Fallback: try to parse from plain text
            resp = await self.llm.complete(messages)
            return Reflection(
                summary=resp.content[:500],
                what_worked=["reflection produced in fallback mode"],
                raw={"raw_response": resp.content},
            )
