"""Playbook — the evolving context document."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from ctxf.models.bullet import Bullet


DEFAULT_SECTIONS = [
    "governance",
    "technical",
    "style",
    "domain",
    "workflow",
]


@dataclass
class Playbook:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    name: str = "default"
    sections: list[str] = field(default_factory=lambda: list(DEFAULT_SECTIONS))
    bullets: list[Bullet] = field(default_factory=list)

    def active_bullets(self) -> list[Bullet]:
        return [b for b in self.bullets if not b.deprecated]

    def bullets_for_section(self, section: str) -> list[Bullet]:
        return [b for b in self.active_bullets() if b.section == section]

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        for b in self.bullets:
            if b.id == bullet_id:
                return b
        return None

    def add_bullet(self, bullet: Bullet) -> None:
        if bullet.section not in self.sections:
            self.sections.append(bullet.section)
        self.bullets.append(bullet)

    def remove_bullet(self, bullet_id: str) -> bool:
        b = self.get_bullet(bullet_id)
        if b:
            b.deprecated = True
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "sections": self.sections,
            "bullets": [b.to_dict() for b in self.bullets],
            "active_count": len(self.active_bullets()),
            "total_count": len(self.bullets),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Playbook":
        pb = cls(
            id=data.get("id", str(uuid.uuid4())),
            version=data.get("version", 1),
            name=data.get("name", "default"),
            sections=data.get("sections", list(DEFAULT_SECTIONS)),
        )
        for bd in data.get("bullets", []):
            pb.bullets.append(Bullet.from_dict(bd))
        return pb

    @classmethod
    def create(cls, name: str = "default") -> "Playbook":
        return cls(name=name, sections=list(DEFAULT_SECTIONS))
