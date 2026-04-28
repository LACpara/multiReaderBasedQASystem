from __future__ import annotations

from typing import Protocol

from hmr.domain import ReaderNode, VectorCandidate


class VectorIndex(Protocol):
    """Vector database boundary for Reader capability recall."""

    def upsert_reader(self, reader: ReaderNode) -> None:
        """Index one Reader's capability representation."""

    def query(self, question: str, *, top_k: int) -> list[VectorCandidate]:
        """Return coarse recall candidates for a user question."""

    def delete_document(self, document_id: str) -> None:
        """Delete indexed records for one document."""

    def close(self) -> None:
        """Release vector database resources if needed."""
