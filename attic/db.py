"""SQLite storage: documents, FTS5 index, extraction facts, drafts, audit log."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  doc_type TEXT, vendor TEXT, product TEXT, model_no TEXT, serial_no TEXT,
  purchase_date TEXT, amount REAL, currency TEXT,
  warranty_months INTEGER, expires_on TEXT,
  file_path TEXT, ocr_text TEXT, confidence REAL,
  notes TEXT DEFAULT '',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS facts(
  doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  key TEXT, value TEXT, source_span TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
  USING fts5(title, vendor, product, notes, ocr_text);
CREATE TABLE IF NOT EXISTS drafts(
  id TEXT PRIMARY KEY, file_path TEXT, ocr_text TEXT, fields_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, tool TEXT, summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_expires ON documents(expires_on);
"""

DOC_COLS = (
    "title", "doc_type", "vendor", "product", "model_no", "serial_no",
    "purchase_date", "amount", "currency", "warranty_months", "expires_on",
)

_STOP = {
    "a", "an", "the", "my", "me", "of", "for", "to", "is", "are", "does", "do",
    "when", "what", "which", "where", "how", "on", "in", "at", "and", "or",
    "warranty", "expire", "expires", "expiry", "expiring", "receipt", "document",
    "number", "model", "about", "any", "i", "it", "this", "that",
}


def days_left(expires_on: str | None, today: date | None = None) -> int | None:
    if not expires_on:
        return None
    return (date.fromisoformat(expires_on) - (today or date.today())).days


def _span(d: int) -> str:
    if d < 14:
        return f"{d} day{'s' if d != 1 else ''}"
    if d < 60:
        return f"about {round(d / 7)} weeks"
    if d < 700:
        return f"about {round(d / 30)} months"
    return f"about {round(d / 365, 1):g} years"


def spoken_span(d: int) -> str:
    """Short, speakable phrase for Alexa to read aloud."""
    if d < 0:
        return f"expired {_span(-d)} ago"
    if d == 0:
        return "expires today"
    return f"expires in {_span(d)}"


# Everyday words people say aloud vs. what receipts print.
_SYNONYMS = {
    "fridge": ["refrigerator"], "refrigerator": ["fridge"],
    "tv": ["television"], "television": ["tv"],
    "ac": ["air", "conditioner"], "washer": ["washing"], "mixie": ["mixer", "grinder"],
    "laptop": ["notebook"], "phone": ["mobile", "smartphone"], "mobile": ["phone"],
    "cooker": ["pressure", "induction"], "geyser": ["water", "heater"],
}


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query with prefix matching and light synonyms."""
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    keep = [t for t in toks if t not in _STOP] or toks
    keep = keep[:8]
    for t in list(keep):
        keep.extend(_SYNONYMS.get(t, []))
    return " OR ".join(f'"{t}"*' for t in dict.fromkeys(keep))


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- documents -------------------------------------------------------
    def add_document(
        self,
        fields: dict[str, Any],
        ocr_text: str = "",
        file_path: str | None = None,
        confidence: float | None = None,
        spans: dict[str, str] | None = None,
    ) -> int:
        vals = {c: fields.get(c) for c in DOC_COLS}
        if not vals["title"]:
            vals["title"] = " - ".join(
                x for x in (vals.get("product") or vals.get("doc_type"), vals.get("vendor")) if x
            ) or "Untitled document"
        with self._lock:
            cur = self.conn.execute(
                f"INSERT INTO documents({','.join(DOC_COLS)}, file_path, ocr_text, confidence) "
                f"VALUES ({','.join('?' * len(DOC_COLS))}, ?, ?, ?)",
                [vals[c] for c in DOC_COLS] + [file_path, ocr_text, confidence],
            )
            doc_id = cur.lastrowid
            for key, span in (spans or {}).items():
                self.conn.execute(
                    "INSERT INTO facts(doc_id,key,value,source_span) VALUES (?,?,?,?)",
                    (doc_id, key, str(vals.get(key, "")), span),
                )
            self.conn.execute(
                "INSERT INTO documents_fts(rowid,title,vendor,product,notes,ocr_text) VALUES (?,?,?,?,?,?)",
                (doc_id, vals["title"], vals["vendor"] or "", vals["product"] or "", "", ocr_text or ""),
            )
            self.conn.commit()
        return int(doc_id)

    def get(self, doc_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            doc = dict(row)
            doc["spans"] = {
                r["key"]: r["source_span"]
                for r in self.conn.execute("SELECT key, source_span FROM facts WHERE doc_id=?", (doc_id,))
            }
        return doc

    def search(self, query: str, doc_type: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        q = fts_query(query)
        with self._lock:
            rows: list[sqlite3.Row] = []
            if q:
                sql = (
                    "SELECT d.* FROM documents_fts f JOIN documents d ON d.id=f.rowid "
                    "WHERE documents_fts MATCH ?"
                )
                args: list[Any] = [q]
                if doc_type:
                    sql += " AND d.doc_type=?"
                    args.append(doc_type)
                sql += " ORDER BY bm25(documents_fts, 5, 3, 5, 1, 1) LIMIT ?"
                args.append(limit)
                rows = self.conn.execute(sql, args).fetchall()
            else:
                sql, args = "SELECT * FROM documents", []
                if doc_type:
                    sql += " WHERE doc_type=?"
                    args.append(doc_type)
                rows = self.conn.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
        return [dict(r) for r in rows]

    def expiring(self, days: int, today: date | None = None, grace_days: int = 30) -> list[dict[str, Any]]:
        today = today or date.today()
        lo = (today - timedelta(days=grace_days)).isoformat()
        hi = (today + timedelta(days=days)).isoformat()
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM documents WHERE expires_on IS NOT NULL AND expires_on BETWEEN ? AND ? "
                "ORDER BY expires_on",
                (lo, hi),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_note(self, doc_id: int, text: str) -> bool:
        with self._lock:
            row = self.conn.execute("SELECT notes FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return False
            notes = (row["notes"] + "\n" + text).strip()
            self.conn.execute("UPDATE documents SET notes=? WHERE id=?", (notes, doc_id))
            d = self.conn.execute(
                "SELECT title,vendor,product,ocr_text FROM documents WHERE id=?", (doc_id,)
            ).fetchone()
            self.conn.execute("DELETE FROM documents_fts WHERE rowid=?", (doc_id,))
            self.conn.execute(
                "INSERT INTO documents_fts(rowid,title,vendor,product,notes,ocr_text) VALUES (?,?,?,?,?,?)",
                (doc_id, d["title"], d["vendor"] or "", d["product"] or "", notes, d["ocr_text"] or ""),
            )
            self.conn.commit()
        return True

    def all_brief(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id,title,doc_type,vendor,product,purchase_date,expires_on FROM documents ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- drafts (ingest, then human confirm) -------------------------------
    def create_draft(self, file_path: str | None, ocr_text: str, fields: dict[str, Any]) -> str:
        draft_id = uuid.uuid4().hex
        with self._lock:
            self.conn.execute(
                "INSERT INTO drafts(id,file_path,ocr_text,fields_json) VALUES (?,?,?,?)",
                (draft_id, file_path, ocr_text, json.dumps(fields)),
            )
            self.conn.commit()
        return draft_id

    def pop_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
            if not row:
                return None
            self.conn.execute("DELETE FROM drafts WHERE id=?", (draft_id,))
            self.conn.commit()
        return {"file_path": row["file_path"], "ocr_text": row["ocr_text"], "fields": json.loads(row["fields_json"])}

    # ---- audit -------------------------------------------------------------
    def audit(self, tool: str, summary: str = "") -> None:
        with self._lock:
            self.conn.execute("INSERT INTO audit(tool,summary) VALUES (?,?)", (tool, summary[:200]))
            self.conn.commit()
