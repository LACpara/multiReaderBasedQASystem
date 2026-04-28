from __future__ import annotations

from typing import Protocol

from hmr.domain import ReaderNode, RetrievalResult


class KnowledgeStore(Protocol):
    """Structured persistence boundary for Reader metadata and query traces."""

    def init_schema(self) -> None:
        """Create required tables if they do not exist."""

    def upsert_reader(self, reader: ReaderNode) -> None:
        """Insert or update one Reader node."""

    def get_reader(self, reader_id: str) -> ReaderNode | None:
        """Return one Reader by id."""

    def list_children(self, parent_id: str) -> list[ReaderNode]:
        """Return children of a Reader ordered by ordinal."""

    def list_document_readers(self, document_id: str) -> list[ReaderNode]:
        """Return all Readers for one document."""

    def delete_document(self, document_id: str) -> None:
        """Delete all structured data for one document."""

    def save_query_result(self, result: RetrievalResult) -> None:
        """Persist a retrieval trace for observability."""

    def close(self) -> None:
        """Release database resources."""
