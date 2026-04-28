from __future__ import annotations

import json
import logging
from dataclasses import asdict
import sqlite3
from pathlib import Path
from typing import Any

from hmr.domain import ReaderKnowledge, ReaderNode, RetrievalResult, utc_now_iso

logger = logging.getLogger(__name__)


class SQLiteKnowledgeStore:
    """SQLite-backed implementation of the structured storage port."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        logger.info("SQLite store opened at %s", self.db_path)

    def init_schema(self) -> None:
        logger.debug("Initializing SQLite schema")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS readers (
                reader_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                parent_id TEXT,
                depth INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                knowledge_json TEXT NOT NULL,
                child_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_readers_document
            ON readers(document_id);

            CREATE INDEX IF NOT EXISTS idx_readers_parent
            ON readers(parent_id);

            CREATE TABLE IF NOT EXISTS query_logs (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def upsert_reader(self, reader: ReaderNode) -> None:
        logger.debug("Upserting reader id=%s title=%s", reader.reader_id, reader.title)
        self.connection.execute(
            """
            INSERT INTO readers (
                reader_id, document_id, title, parent_id, depth, ordinal,
                text, knowledge_json, child_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reader_id) DO UPDATE SET
                document_id=excluded.document_id,
                title=excluded.title,
                parent_id=excluded.parent_id,
                depth=excluded.depth,
                ordinal=excluded.ordinal,
                text=excluded.text,
                knowledge_json=excluded.knowledge_json,
                child_ids_json=excluded.child_ids_json,
                created_at=excluded.created_at
            """,
            (
                reader.reader_id,
                reader.document_id,
                reader.title,
                reader.parent_id,
                reader.depth,
                reader.ordinal,
                reader.text,
                json.dumps(reader.knowledge.to_dict(), ensure_ascii=False),
                json.dumps(reader.child_ids, ensure_ascii=False),
                reader.created_at,
            ),
        )
        self.connection.commit()

    def get_reader(self, reader_id: str) -> ReaderNode | None:
        row = self.connection.execute(
            "SELECT * FROM readers WHERE reader_id = ?",
            (reader_id,),
        ).fetchone()
        return self._row_to_reader(row) if row else None

    def list_children(self, parent_id: str) -> list[ReaderNode]:
        rows = self.connection.execute(
            "SELECT * FROM readers WHERE parent_id = ? ORDER BY ordinal",
            (parent_id,),
        ).fetchall()
        return [self._row_to_reader(row) for row in rows]

    def list_document_readers(self, document_id: str) -> list[ReaderNode]:
        rows = self.connection.execute(
            "SELECT * FROM readers WHERE document_id = ? ORDER BY depth, ordinal",
            (document_id,),
        ).fetchall()
        return [self._row_to_reader(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        logger.info("Deleting existing readers for document_id=%s", document_id)
        self.connection.execute("DELETE FROM readers WHERE document_id = ?", (document_id,))
        self.connection.commit()

    def save_query_result(self, result: RetrievalResult) -> None:
        trace = {
            "candidates": [asdict(candidate) for candidate in result.candidates],
            "activated_answers": [asdict(answer) for answer in result.activated_answers],
        }
        self.connection.execute(
            """
            INSERT INTO query_logs (question, answer, trace_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                result.question,
                result.answer,
                json.dumps(trace, ensure_ascii=False),
                utc_now_iso(),
            ),
        )
        self.connection.commit()
        logger.debug("Saved query log for question=%s", result.question)

    def close(self) -> None:
        logger.info("Closing SQLite store")
        self.connection.close()

    def _row_to_reader(self, row: sqlite3.Row) -> ReaderNode:
        payload: dict[str, Any] = json.loads(row["knowledge_json"])
        return ReaderNode(
            reader_id=row["reader_id"],
            document_id=row["document_id"],
            title=row["title"],
            parent_id=row["parent_id"],
            depth=int(row["depth"]),
            ordinal=int(row["ordinal"]),
            text=row["text"],
            knowledge=ReaderKnowledge.from_dict(payload),
            child_ids=list(json.loads(row["child_ids_json"])),
            created_at=row["created_at"],
        )
