from __future__ import annotations

import logging
from pathlib import Path

from hmr.config import AppConfig
from hmr.domain import RetrievalResult
from hmr.llm.base import ReaderLLMService
from hmr.llm.heuristic_service import HeuristicReaderLLMService
from hmr.reader_builder import ReaderTreeBuilder
from hmr.retrieval_engine import RetrievalEngine
from hmr.storage.sqlite_store import SQLiteKnowledgeStore
from hmr.vector.chroma_index import ChromaVectorIndex

logger = logging.getLogger(__name__)


class ReaderRetrievalApp:
    """Pre-assembled application facade for ingestion and retrieval."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: ReaderLLMService | None = None,
    ) -> None:
        self.config = config
        self.llm_service = llm_service or HeuristicReaderLLMService()
        self.store = SQLiteKnowledgeStore(config.storage.sqlite_path)
        self.store.init_schema()
        self.vector_index = ChromaVectorIndex(
            config.storage.chroma_path,
            config.storage.chroma_collection,
        )
        self.builder = ReaderTreeBuilder(
            config.ingestion,
            self.llm_service,
            self.store,
            self.vector_index,
        )
        self.engine = RetrievalEngine(
            config.retrieval,
            self.llm_service,
            self.store,
            self.vector_index,
        )
        logger.info("ReaderRetrievalApp initialized")

    def ingest_file(self, path: Path, *, document_id: str | None = None) -> str:
        text = path.read_text(encoding="utf-8")
        doc_id = document_id or path.stem
        root = self.builder.ingest_document(document_id=doc_id, title=path.stem, text=text)
        return root.reader_id

    def ask(self, question: str) -> RetrievalResult:
        return self.engine.ask(question)

    def close(self) -> None:
        self.vector_index.close()
        self.store.close()
