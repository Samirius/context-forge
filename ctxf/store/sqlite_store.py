"""SQLite-backed playbook store with version history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from ctxf.models.bullet import Bullet
from ctxf.models.delta import Delta, DeltaOp
from ctxf.models.playbook import Playbook


class SqliteStore:
    """Persistent store for playbooks using SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_tables(self) -> None:
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS playbooks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT 'default',
                version INTEGER NOT NULL DEFAULT 1,
                sections TEXT NOT NULL DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS bullets (
                id TEXT PRIMARY KEY,
                playbook_id TEXT NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                helpful INTEGER NOT NULL DEFAULT 0,
                harmful INTEGER NOT NULL DEFAULT 0,
                deprecated INTEGER NOT NULL DEFAULT 0,
                embedding TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (playbook_id) REFERENCES playbooks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_bullets_playbook
                ON bullets(playbook_id);

            CREATE INDEX IF NOT EXISTS idx_bullets_section
                ON bullets(section);

            CREATE TABLE IF NOT EXISTS version_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                playbook_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (playbook_id) REFERENCES playbooks(id)
            );

            CREATE TABLE IF NOT EXISTS delta_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                playbook_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                delta TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (playbook_id) REFERENCES playbooks(id)
            );
        """)
        self.conn.commit()

    # --- Playbook CRUD ---

    def create_playbook(self, name: str = "default") -> Playbook:
        pb = Playbook.create(name=name)
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO playbooks (id, name, version, sections) VALUES (?, ?, ?, ?)",
            (pb.id, pb.name, pb.version, json.dumps(pb.sections)),
        )
        self.conn.commit()
        self._save_version(pb)
        return pb

    def get_playbook(self, playbook_id: Optional[str] = None, name: Optional[str] = None) -> Optional[Playbook]:
        c = self.conn.cursor()
        if playbook_id:
            row = c.execute("SELECT * FROM playbooks WHERE id = ?", (playbook_id,)).fetchone()
        elif name:
            row = c.execute("SELECT * FROM playbooks WHERE name = ? ORDER BY updated_at DESC LIMIT 1", (name,)).fetchone()
        else:
            row = c.execute("SELECT * FROM playbooks ORDER BY updated_at DESC LIMIT 1").fetchone()

        if not row:
            return None

        pb = Playbook(
            id=row["id"],
            version=row["version"],
            name=row["name"],
            sections=json.loads(row["sections"]),
        )
        # Load bullets
        brows = c.execute(
            "SELECT * FROM bullets WHERE playbook_id = ? AND deprecated = 0", (pb.id,)
        ).fetchall()
        for br in brows:
            emb = None
            if br["embedding"]:
                emb = json.loads(br["embedding"])
            pb.bullets.append(Bullet(
                id=br["id"],
                section=br["section"],
                content=br["content"],
                helpful=br["helpful"],
                harmful=br["harmful"],
                deprecated=bool(br["deprecated"]),
                embedding=emb,
            ))
        return pb

    def list_playbooks(self) -> list[dict]:
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT p.id, p.name, p.version, p.updated_at, "
            "  (SELECT COUNT(*) FROM bullets b WHERE b.playbook_id = p.id AND b.deprecated = 0) as active_bullets "
            "FROM playbooks p ORDER BY p.updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def save_playbook(self, pb: Playbook) -> None:
        """Persist playbook metadata (sections, version bump)."""
        c = self.conn.cursor()
        pb.version += 1
        c.execute(
            "UPDATE playbooks SET version = ?, sections = ?, updated_at = datetime('now') WHERE id = ?",
            (pb.version, json.dumps(pb.sections), pb.id),
        )
        self.conn.commit()
        self._save_version(pb)

    # --- Bullets ---

    def save_bullet(self, playbook_id: str, bullet: Bullet) -> None:
        c = self.conn.cursor()
        emb_json = json.dumps(bullet.embedding) if bullet.embedding else None
        c.execute(
            "INSERT OR REPLACE INTO bullets (id, playbook_id, section, content, helpful, harmful, deprecated, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (bullet.id, playbook_id, bullet.section, bullet.content,
             bullet.helpful, bullet.harmful, int(bullet.deprecated), emb_json),
        )
        self.conn.commit()

    def update_bullet(self, bullet: Bullet) -> None:
        c = self.conn.cursor()
        emb_json = json.dumps(bullet.embedding) if bullet.embedding else None
        c.execute(
            "UPDATE bullets SET content = ?, helpful = ?, harmful = ?, deprecated = ?, embedding = ?, section = ? "
            "WHERE id = ?",
            (bullet.content, bullet.helpful, bullet.harmful, int(bullet.deprecated), emb_json, bullet.section, bullet.id),
        )
        self.conn.commit()

    def delete_bullet(self, bullet_id: str) -> bool:
        c = self.conn.cursor()
        c.execute("UPDATE bullets SET deprecated = 1 WHERE id = ?", (bullet_id,))
        self.conn.commit()
        return c.rowcount > 0

    def get_all_active_bullets(self, playbook_id: str) -> list[Bullet]:
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT * FROM bullets WHERE playbook_id = ? AND deprecated = 0", (playbook_id,)
        ).fetchall()
        bullets = []
        for r in rows:
            emb = json.loads(r["embedding"]) if r["embedding"] else None
            bullets.append(Bullet(
                id=r["id"], section=r["section"], content=r["content"],
                helpful=r["helpful"], harmful=r["harmful"],
                deprecated=bool(r["deprecated"]), embedding=emb,
            ))
        return bullets

    def update_embeddings(self, bullets: list[Bullet]) -> None:
        c = self.conn.cursor()
        for b in bullets:
            if b.embedding is not None:
                c.execute(
                    "UPDATE bullets SET embedding = ? WHERE id = ?",
                    (json.dumps(b.embedding), b.id),
                )
        self.conn.commit()

    # --- Delta Application ---

    def apply_delta(self, playbook_id: str, delta: Delta) -> Optional[Bullet]:
        """Apply a single delta and return the affected/new bullet."""
        pb = self.get_playbook(playbook_id)
        if pb is None:
            raise ValueError(f"Playbook {playbook_id} not found")

        result: Optional[Bullet] = None

        if delta.op == DeltaOp.ADD:
            bullet = Bullet.create(section=delta.section or "general", content=delta.content or "")
            self.save_bullet(playbook_id, bullet)
            pb.bullets.append(bullet)
            result = bullet

        elif delta.op == DeltaOp.PATCH:
            bullet = pb.get_bullet(delta.bullet_id or "")
            if bullet:
                bullet.content = delta.new_content or bullet.content
                self.update_bullet(bullet)
                result = bullet

        elif delta.op == DeltaOp.INCR:
            bullet = pb.get_bullet(delta.bullet_id or "")
            if bullet:
                field_name = delta.field or "helpful"
                current = getattr(bullet, field_name, 0)
                setattr(bullet, field_name, current + delta.amount)
                self.update_bullet(bullet)
                result = bullet

        elif delta.op == DeltaOp.DEPRECATE:
            bullet = pb.get_bullet(delta.bullet_id or "")
            if bullet:
                bullet.deprecated = True
                self.update_bullet(bullet)
                result = bullet

        elif delta.op == DeltaOp.MERGE:
            ids = delta.bullet_ids or []
            merged = []
            for bid in ids:
                b = pb.get_bullet(bid)
                if b:
                    merged.append(b)
                    b.deprecated = True
                    self.update_bullet(b)
            if merged and delta.content:
                section = merged[0].section
                new_b = Bullet.create(section=section, content=delta.content)
                self.save_bullet(playbook_id, new_b)
                result = new_b

        # Log the delta
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO delta_log (playbook_id, version, delta) VALUES (?, ?, ?)",
            (playbook_id, pb.version, json.dumps(delta.to_dict())),
        )
        self.conn.commit()

        # Bump version
        self.save_playbook(pb)

        return result

    def apply_deltas(self, playbook_id: str, deltas: list[Delta]) -> list[Optional[Bullet]]:
        results = []
        for d in deltas:
            results.append(self.apply_delta(playbook_id, d))
        return results

    # --- Version History ---

    def _save_version(self, pb: Playbook) -> None:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO version_history (playbook_id, version, snapshot) VALUES (?, ?, ?)",
            (pb.id, pb.version, json.dumps(pb.to_dict())),
        )
        self.conn.commit()

    def get_version_history(self, playbook_id: str, limit: int = 20) -> list[dict]:
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT version, created_at FROM version_history WHERE playbook_id = ? ORDER BY version DESC LIMIT ?",
            (playbook_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_delta_log(self, playbook_id: str, limit: int = 50) -> list[dict]:
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT version, delta, created_at FROM delta_log WHERE playbook_id = ? ORDER BY id DESC LIMIT ?",
            (playbook_id, limit),
        ).fetchall()
        return [{"version": r["version"], "delta": json.loads(r["delta"]), "created_at": r["created_at"]} for r in rows]

    # --- Stats ---

    def get_stats(self, playbook_id: Optional[str] = None) -> dict:
        pb = self.get_playbook(playbook_id)
        if not pb:
            return {"error": "no playbook found"}

        c = self.conn.cursor()
        total = c.execute(
            "SELECT COUNT(*) as cnt FROM bullets WHERE playbook_id = ?", (pb.id,)
        ).fetchone()["cnt"]
        active = c.execute(
            "SELECT COUNT(*) as cnt FROM bullets WHERE playbook_id = ? AND deprecated = 0", (pb.id,)
        ).fetchone()["cnt"]
        deprecated = total - active

        section_rows = c.execute(
            "SELECT section, COUNT(*) as cnt FROM bullets WHERE playbook_id = ? AND deprecated = 0 GROUP BY section",
            (pb.id,),
        ).fetchall()
        by_section = {r["section"]: r["cnt"] for r in section_rows}

        helpful_total = c.execute(
            "SELECT COALESCE(SUM(helpful),0) as s FROM bullets WHERE playbook_id = ? AND deprecated = 0",
            (pb.id,),
        ).fetchone()["s"]
        harmful_total = c.execute(
            "SELECT COALESCE(SUM(harmful),0) as s FROM bullets WHERE playbook_id = ? AND deprecated = 0",
            (pb.id,),
        ).fetchone()["s"]

        return {
            "playbook_id": pb.id,
            "name": pb.name,
            "version": pb.version,
            "total_bullets": total,
            "active_bullets": active,
            "deprecated_bullets": deprecated,
            "by_section": by_section,
            "total_helpful": helpful_total,
            "total_harmful": harmful_total,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
