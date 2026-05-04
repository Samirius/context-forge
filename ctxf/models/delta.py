"""Delta operations — the mutation primitives for playbook evolution."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class DeltaOp(enum.Enum):
    ADD = "ADD"
    PATCH = "PATCH"
    INCR = "INCR"
    DEPRECATE = "DEPRECATE"
    MERGE = "MERGE"


@dataclass
class Delta:
    op: DeltaOp
    # ADD: section, content
    section: Optional[str] = None
    content: Optional[str] = None
    # PATCH: bullet_id, new_content
    bullet_id: Optional[str] = None
    new_content: Optional[str] = None
    # INCR: bullet_id, field ("helpful"|"harmful"), amount
    field: Optional[str] = None
    amount: int = 1
    # MERGE: bullet_ids, merged_content
    bullet_ids: Optional[list[str]] = None
    # Reason for traceability
    reason: str = ""

    def to_dict(self) -> dict:
        d: dict = {"op": self.op.value, "reason": self.reason}
        if self.section is not None:
            d["section"] = self.section
        if self.content is not None:
            d["content"] = self.content
        if self.bullet_id is not None:
            d["bullet_id"] = self.bullet_id
        if self.new_content is not None:
            d["new_content"] = self.new_content
        if self.field is not None:
            d["field"] = self.field
        if self.amount != 1:
            d["amount"] = self.amount
        if self.bullet_ids is not None:
            d["bullet_ids"] = self.bullet_ids
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Delta":
        return cls(
            op=DeltaOp(data["op"]),
            section=data.get("section"),
            content=data.get("content"),
            bullet_id=data.get("bullet_id"),
            new_content=data.get("new_content"),
            field=data.get("field"),
            amount=data.get("amount", 1),
            bullet_ids=data.get("bullet_ids"),
            reason=data.get("reason", ""),
        )

    @classmethod
    def add(cls, section: str, content: str, reason: str = "") -> "Delta":
        return cls(op=DeltaOp.ADD, section=section, content=content, reason=reason)

    @classmethod
    def patch(cls, bullet_id: str, new_content: str, reason: str = "") -> "Delta":
        return cls(op=DeltaOp.PATCH, bullet_id=bullet_id, new_content=new_content, reason=reason)

    @classmethod
    def incr(cls, bullet_id: str, field: str, amount: int = 1, reason: str = "") -> "Delta":
        return cls(op=DeltaOp.INCR, bullet_id=bullet_id, field=field, amount=amount, reason=reason)

    @classmethod
    def deprecate(cls, bullet_id: str, reason: str = "") -> "Delta":
        return cls(op=DeltaOp.DEPRECATE, bullet_id=bullet_id, reason=reason)

    @classmethod
    def merge(cls, bullet_ids: list[str], merged_content: str, reason: str = "") -> "Delta":
        return cls(op=DeltaOp.MERGE, bullet_ids=bullet_ids, content=merged_content, reason=reason)
