from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any
from typing_extensions import override

from hmr.utils import retry
from hmr.domain import ActivationDecision, ReaderAnswer, ReaderKnowledge
from hmr.prompt_loader import PromptLoader
from hmr.llm.base import LLMClient, ReaderLLMService

logger = logging.getLogger(__name__)


class PromptedReaderLLMService(ReaderLLMService):
    """LLM-backed implementation of high-level Reader operations.

    It depends only on the low-level LLMClient protocol, not on a specific vendor.
    The demo quick-start uses HeuristicReaderLLMService, while production can swap
    this service in without changing storage, vector search, or core orchestration.
    """

    def __init__(self, client: LLMClient, prompt_loader: PromptLoader = None) -> None:
        self.client = client
        self.prompt_loader = prompt_loader or PromptLoader(Path(__file__).resolve().parent.parent / "promptTemplates")

    @override
    def extract_knowledge(self, text: str, *, title: str) -> ReaderKnowledge:
        prompt = self.prompt_loader.get_prompt("knowledge_extract", text=text, title=title)
        payload = self._json_call(prompt)
        return ReaderKnowledge.from_dict({**payload, "source_excerpt": text[:500]})

    @override
    def build_capability_questions(self, knowledge: ReaderKnowledge, *, title: str) -> list[str]:
        prompt = self.prompt_loader.get_prompt("question_set_build", knowledge=knowledge, title=title)
        payload = self._json_call(prompt)
        questions = payload.get("capability_questions", [])
        return [str(question) for question in questions][:10]

    @override
    def evaluate_activation(self, knowledge: ReaderKnowledge, question: str) -> ActivationDecision:
        prompt = self.prompt_loader.get_prompt("evluate_activation", knowledge=knowledge, question=question)
        payload = self._json_call(prompt)
        return ActivationDecision(
            should_answer=bool(payload.get("should_answer", False)),
            score=float(payload.get("score", 0.0)),
            sub_question=str(payload.get("sub_question", question)),
            reason=str(payload.get("reason", "")),
        )

    @override
    def answer_question(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str,
    ) -> ReaderAnswer:
        prompt = self.prompt_loader.get_prompt("retrival_answer", knowledge=knowledge, question=question)
        payload = self._json_call(prompt)
        return ReaderAnswer(
            reader_id=reader_id,
            title=title,
            answer=str(payload.get("answer", "")),
            confidence=float(payload.get("confidence", 0.5)),
            source_excerpt=knowledge.source_excerpt,
        )

    @override
    def merge_answers(self, question: str, answers: list[ReaderAnswer]) -> str:
        prompt = self.prompt_loader.get_prompt("retrival_merge", question=question, answers=answers)
        result = self.client.complete(prompt, temperature=0.0, max_tokens=900).strip()
        return result
    
    @retry(retries=5)
    def _json_call(self, prompt: str) -> dict[str, Any]:
        raw = self.client.complete(prompt, temperature=0.0, max_tokens=1100, json_require=True)
        try:
            return json.loads(self._strip_fence(raw))
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", raw)
            raise ValueError("LLM service returned invalid JSON") from exc

    def _strip_fence(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1]
            stripped = stripped.rsplit("```", 1)[0]
        return stripped.strip()
