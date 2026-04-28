from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hmr.domain import ReaderNode, VectorCandidate
from hmr.vector.embedding import HashEmbeddingModel

logger = logging.getLogger(__name__)


class ChromaVectorIndex:
    """ChromaDB implementation of the vector recall boundary."""

    def __init__(
        self,
        persist_path: Path,
        collection_name: str,
        embedding_model: HashEmbeddingModel | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for ChromaVectorIndex. Run: pip install -r requirements.txt"
            ) from exc

        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info("Chroma index opened at %s collection=%s", self.persist_path, collection_name)

    def upsert_reader(self, reader: ReaderNode) -> None:
        document = self._capability_document(reader)
        metadata = self._metadata(reader)
        embedding = self.embedding_model.embed(document)
        logger.debug("Upserting vector for reader_id=%s", reader.reader_id)
        self.collection.upsert(
            ids=[reader.reader_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def query(self, question: str, *, top_k: int) -> list[VectorCandidate]:
        logger.info("Running Chroma coarse recall top_k=%s", top_k)
        result = self.collection.query(
            query_embeddings=[self.embedding_model.embed(question)],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        return self._to_candidates(result)

    def delete_document(self, document_id: str) -> None:
        logger.info("Deleting Chroma vectors for document_id=%s", document_id)
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:  # Chroma raises if where matches nothing in some versions.
            logger.debug("Chroma delete skipped: %s", exc)

    def close(self) -> None:
        logger.debug("Chroma PersistentClient does not require explicit close")

    def _capability_document(self, reader: ReaderNode) -> str:
        knowledge = reader.knowledge
        questions = "\n".join(knowledge.capability_questions)
        return "\n".join(
            part
            for part in [reader.title, questions, knowledge.summary, *knowledge.entities, *knowledge.relations]
            if part
        )

    def _metadata(self, reader: ReaderNode) -> dict[str, str | int | bool | None]:
        return {
            "reader_id": reader.reader_id,
            "document_id": reader.document_id,
            "title": reader.title,
            "parent_id": reader.parent_id or "",
            "depth": reader.depth,
            "is_leaf": reader.is_leaf,
        }

    def _to_candidates(self, result: dict[str, Any]) -> list[VectorCandidate]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        candidates: list[VectorCandidate] = []
        for reader_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            candidates.append(
                VectorCandidate(
                    reader_id=reader_id,
                    score=self._distance_to_score(float(distance)),
                    document=document or "",
                    metadata=dict(metadata or {}),
                )
            )
        return candidates

    def _distance_to_score(self, distance: float) -> float:
        return max(0.0, 1.0 - min(distance, 2.0) / 2.0)
