from __future__ import annotations

import logging

from hmr.config import RetrievalConfig
from hmr.domain import ReaderAnswer, RetrievalResult, VectorCandidate
from hmr.llm.base import ReaderLLMService
from hmr.storage.base import KnowledgeStore
from hmr.vector.base import VectorIndex

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Runs two-stage self-activated retrieval over indexed Readers."""

    def __init__(
        self,
        config: RetrievalConfig,
        llm_service: ReaderLLMService,
        store: KnowledgeStore,
        vector_index: VectorIndex,
    ) -> None:
        self.config = config
        self.llm_service = llm_service
        self.store = store
        self.vector_index = vector_index

    def ask(self, question: str) -> RetrievalResult:
        logger.info("Received query: %s", question)
        candidates = self.vector_index.query(question, top_k=self.config.top_k)
        answers = self._activate_and_answer(question, candidates)
        answer = self.llm_service.merge_answers(question, answers)
        result = RetrievalResult(question=question, answer=answer, candidates=candidates, activated_answers=answers)
        self.store.save_query_result(result)
        logger.info("Query completed with %s activated readers", len(answers))
        return result

    def _activate_and_answer(
        self,
        question: str,
        candidates: list[VectorCandidate],
    ) -> list[ReaderAnswer]:
        answers: list[ReaderAnswer] = []
        for candidate in candidates:
            reader = self.store.get_reader(candidate.reader_id)
            if reader is None:
                logger.warning("Candidate reader not found reader_id=%s", candidate.reader_id)
                continue
            decision = self.llm_service.evaluate_activation(reader.knowledge, question)
            logger.info(
                "Activation reader=%s vector=%.3f self=%.3f pass=%s",
                reader.title,
                candidate.score,
                decision.score,
                decision.should_answer,
            )
            if not self._passes_activation(decision.score, decision.should_answer):
                continue
            answers.append(
                self.llm_service.answer_question(
                    reader.knowledge,
                    decision.sub_question,
                    reader_id=reader.reader_id,
                    title=reader.title,
                )
            )
            if len(answers) >= self.config.max_answers:
                break
        return answers

    def _passes_activation(self, score: float, should_answer: bool) -> bool:
        return should_answer and score >= self.config.activation_threshold
