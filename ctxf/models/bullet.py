"""Bullet — a single playbook entry."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


def _make_bullet_id(section: str) -> str:
    """Generate a bullet id like GOV-00001, TECH-00012, etc."""
    prefix = section[:3].upper()
    # strip non-alpha
    prefix = re.sub(r"[^A-Z]", "", prefix) or "GEN"
    num = uuid.uuid4().int % 100_000
    return f"{prefix}-{num:05d}"


@dataclass
class Bullet:
    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    deprecated: bool = False
    embedding: Optional[list[float]] = field(default=None, repr=False)

    @classmethod
    def create(cls, section: str, content: str) -> "Bullet":
        return cls(
            id=_make_bullet_id(section),
            section=section,
            content=content,
        )

    @property
    def score(self) -> float:
        total = self.helpful + self.harmful
        if total == 0:
            return 0.0
        return self.helpful / total

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "deprecated": self.deprecated,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Bullet":
        return cls(
            id=data["id"],
            section=data["section"],
            content=data["content"],
            helpful=data.get("helpful", 0),
            harmful=data.get("harmful", 0),
            deprecated=data.get("deprecated", False),
        )
