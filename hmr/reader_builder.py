from __future__ import annotations

import logging
import uuid

from hmr.complexity import ComplexityEstimator
from hmr.config import IngestionConfig
from hmr.domain import ReaderNode
from hmr.llm.base import ReaderLLMService
from hmr.storage.base import KnowledgeStore
from hmr.text_splitter import SemanticTextSplitter
from hmr.vector.base import VectorIndex

logger = logging.getLogger(__name__)


class ReaderTreeBuilder:
    """Builds a recursive Tree of Readers from an input document."""

    def __init__(
        self,
        config: IngestionConfig,
        llm_service: ReaderLLMService,
        store: KnowledgeStore,
        vector_index: VectorIndex,
    ) -> None:
        self.config = config
        self.llm_service = llm_service
        self.store = store
        self.vector_index = vector_index
        self.complexity = ComplexityEstimator()
        self.splitter = SemanticTextSplitter(config.max_leaf_chars)

    def ingest_document(self, *, document_id: str, title: str, text: str) -> ReaderNode:
        logger.info("Starting ingestion document_id=%s title=%s", document_id, title)
        self.store.delete_document(document_id)
        self.vector_index.delete_document(document_id)
        root = self._build_node(
            document_id=document_id,
            title=title,
            text=text,
            parent_id=None,
            depth=0,
            ordinal=0,
        )
        logger.info("Finished ingestion root_reader_id=%s", root.reader_id)
        return root

    def _build_node(
        self,
        *,
        document_id: str,
        title: str,
        text: str,
        parent_id: str | None,
        depth: int,
        ordinal: int,
    ) -> ReaderNode:
        reader_id = self._new_reader_id(document_id, depth, ordinal)
        logger.debug("Building reader id=%s depth=%s ordinal=%s", reader_id, depth, ordinal)
        child_ids = self._build_children_if_needed(document_id, title, text, reader_id, depth)
        node = self._make_reader(reader_id, document_id, title, parent_id, depth, ordinal, text, child_ids)
        self.store.upsert_reader(node)
        self.vector_index.upsert_reader(node)
        return node

    def _build_children_if_needed(
        self,
        document_id: str,
        title: str,
        text: str,
        parent_id: str,
        depth: int,
    ) -> list[str]:
        if not self._should_split(text, depth):
            return []
        chunks = self.splitter.split(text)
        if len(chunks) <= 1:
            return []
        logger.info("Reader depth=%s split into %s sub-readers", depth, len(chunks))
        return [
            self._build_node(
                document_id=document_id,
                title=f"{title} / part-{index + 1}",
                text=chunk,
                parent_id=parent_id,
                depth=depth + 1,
                ordinal=index,
            ).reader_id
            for index, chunk in enumerate(chunks)
        ]

    def _should_split(self, text: str, depth: int) -> bool:
        if depth >= self.config.max_depth:
            return False
        score = self.complexity.score(text)
        should_split = score >= self.config.complexity_threshold and len(text) > self.config.max_leaf_chars
        logger.debug("Split decision depth=%s score=%.2f should_split=%s", depth, score, should_split)
        return should_split

    def _make_reader(
        self,
        reader_id: str,
        document_id: str,
        title: str,
        parent_id: str | None,
        depth: int,
        ordinal: int,
        text: str,
        child_ids: list[str],
    ) -> ReaderNode:
        knowledge = self.llm_service.extract_knowledge(text, title=title)
        knowledge.capability_questions = self.llm_service.build_capability_questions(knowledge, title=title)
        return ReaderNode(
            reader_id=reader_id,
            document_id=document_id,
            title=title,
            parent_id=parent_id,
            depth=depth,
            ordinal=ordinal,
            text=text,
            knowledge=knowledge,
            child_ids=child_ids,
        )

    def _new_reader_id(self, document_id: str, depth: int, ordinal: int) -> str:
        random_part = uuid.uuid4().hex[:10]
        return f"reader::{document_id}::{depth}::{ordinal}::{random_part}"
