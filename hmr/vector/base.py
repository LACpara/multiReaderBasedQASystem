from __future__ import annotations

from typing_extensions import overload
from abc import abstractmethod, ABC

from hmr.domain import ReaderNode, VectorCandidate


class VectorIndex(ABC):
    """Vector database boundary for Reader capability recall."""

    @abstractmethod
    def upsert_reader(self, reader: ReaderNode) -> None:
        """Index one Reader's capability representation."""

    @abstractmethod
    def query(self, question: str, *, top_k: int) -> list[VectorCandidate]:
        """Return coarse recall candidates for a user question."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete indexed records for one document."""

    @abstractmethod
    def close(self) -> None:
        """Release vector database resources if needed."""
