"""Generator — task execution wrapper that injects playbook context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ctxf.llm.base import LLMProvider, Message
from ctxf.models.bullet import Bullet


SYSTEM_PROMPT = """You are an expert AI assistant. You have been given a playbook of context bullets \
that represent accumulated knowledge and best practices. Follow these bullets as guidelines \
when executing the task.

Playbook Context:
{playbook_context}

Execute the task below, applying relevant playbook guidance where appropriate. \
If a playbook bullet is not relevant to this task, ignore it. \
After completing the task, note which playbook bullets were helpful and which were not, if any."""


@dataclass
class GeneratorResult:
    response: str
    helpful_bullets: list[str] = field(default_factory=list)
    harmful_bullets: list[str] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)


class Generator:
    """Executes tasks with injected playbook context."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def execute(
        self,
        task: str,
        bullets: list[Bullet],
        system_override: Optional[str] = None,
    ) -> GeneratorResult:
        """Execute a task with playbook context injected."""
        # Format playbook bullets
        if bullets:
            context_lines = []
            for b in bullets:
                context_lines.append(f"[{b.id}] ({b.section}) {b.content}")
            playbook_context = "\n".join(context_lines)
        else:
            playbook_context = "(empty playbook)"

        system = system_override or SYSTEM_PROMPT.format(playbook_context=playbook_context)

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=task),
        ]

        resp = await self.llm.complete(messages)

        # Parse response for bullet references (simple heuristic)
        helpful = []
        harmful = []
        response_lower = resp.content.lower()
        for b in bullets:
            if b.id in resp.content:
                if any(w in response_lower for w in ["helpful", "useful", "correct", "worked"]):
                    helpful.append(b.id)
                elif any(w in response_lower for w in ["harmful", "wrong", "incorrect", "outdated"]):
                    harmful.append(b.id)

        return GeneratorResult(
            response=resp.content,
            helpful_bullets=helpful,
            harmful_bullets=harmful,
            model=resp.model,
            usage=resp.usage,
        )
