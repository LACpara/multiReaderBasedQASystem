from __future__ import annotations

from abc import abstractmethod, ABC

from hmr.domain import ReaderNode, RetrievalResult


class KnowledgeStore(ABC):
    """Structured persistence boundary for Reader metadata and query traces."""

    @abstractmethod
    def init_schema(self) -> None:
        """Create required tables if they do not exist."""

    @abstractmethod
    def upsert_reader(self, reader: ReaderNode) -> None:
        """Insert or update one Reader node."""

    @abstractmethod
    def get_reader(self, reader_id: str) -> ReaderNode | None:
        """Return one Reader by id."""

    @abstractmethod
    def list_children(self, parent_id: str) -> list[ReaderNode]:
        """Return children of a Reader ordered by ordinal."""

    @abstractmethod
    def list_document_readers(self, document_id: str) -> list[ReaderNode]:
        """Return all Readers for one document."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all structured data for one document."""

    @abstractmethod
    def save_query_result(self, result: RetrievalResult) -> None:
        """Persist a retrieval trace for observability."""

    @abstractmethod
    def close(self) -> None:
        """Release database resources."""
