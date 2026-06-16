from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Any
from typing_extensions import override

from hmr.utils import retry
from hmr.domain import ActivationDecision, ReaderAnswer, ReaderKnowledge, CompleteAnswer
from hmr.llm.base import LLMClient, ReaderLLMService

from prompt_manager.prompt_loader import PromptLoader
from prompt_manager.domain import PromptDefinition

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
        prompt = self.prompt_loader.get_prompt("read.knowledge.extract", text=text, title=title)
        payload = self._json_call(prompt)
        return ReaderKnowledge.from_dict({**payload, "source_excerpt": text[:500]})

    @override
    def build_capability_questions(self, knowledge: ReaderKnowledge, *, title: str) -> list[str]:
        prompt = self.prompt_loader.get_prompt("read.question.build", knowledge=knowledge, title=title)
        payload = self._json_call(prompt)
        questions = payload.get("capability_questions", [])
        return [str(question) for question in questions][:10]

    @override
    def evaluate_activation(self, knowledge: ReaderKnowledge, question: str) -> ActivationDecision:
        prompt = self.prompt_loader.get_prompt("read.activate.evaluate", knowledge=knowledge, question=question)
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
        prompt = self.prompt_loader.get_prompt("retrival.answer.build", knowledge=knowledge, question=question)
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
        prompt = self.prompt_loader.get_prompt("retrival.answer.merge", question=question, answers=answers)
        result = self.client.complete(prompt, temperature=0.0, max_tokens=900).strip()
        return result
    
    @override
    def aggregate_children_knowledge(
        self,
        children_knowledge: list[ReaderKnowledge],
        *,
        title: str
    ) -> ReaderKnowledge:
        children_json = json.dumps([k.to_dict() for k in children_knowledge], ensure_ascii=False)
        prompt = self.prompt_loader.get_prompt("read.knowledge.aggregate", title=title, children_knowledge_json=children_json)
        payload = self._json_call(prompt)
        return ReaderKnowledge.from_dict(payload)
    
    @override
    def estimate_capability_from_children(
        self,
        children_capabilities: list[list[str]],
        *,
        title: str,
        limits: int = -1
    ) -> list[str]:
        caps_json = json.dumps(children_capabilities, ensure_ascii=False)
        prompt = self.prompt_loader.get_prompt("read.capability.estimate", title=title, children_capabilities_json=caps_json)
        payload = self._json_call(prompt)
        questions = payload.get("capability_questions", [])
        return [str(q) for q in questions][:limits]
    
    @override
    def detect_information_gaps(
        self,
        text: str,
        knowledge: ReaderKnowledge,
        *,
        title: str
    ) -> list[str]:
        knowledge_json = json.dumps(knowledge.to_dict(), ensure_ascii=False)
        prompt = self.prompt_loader.get_prompt("read.gap.detect", title=title, text=text, knowledge_json=knowledge_json)
        payload = self._json_call(prompt)
        gaps = payload.get("gaps", [])
        return [str(g) for g in gaps]
    
    @override
    def integrate_knowledge(
        self,
        original_knowledge: ReaderKnowledge,
        complete_answers: list[CompleteAnswer],
        *,
        title: str
    ) -> ReaderKnowledge:
        orig_json = json.dumps(original_knowledge.to_dict(), ensure_ascii=False)
        answers_json = json.dumps([a.to_dict() for a in complete_answers], ensure_ascii=False)
        prompt = self.prompt_loader.get_prompt("read.knowledge.integrate", title=title, original_knowledge_json=orig_json, answers_json=answers_json)
        payload = self._json_call(prompt)
        result = ReaderKnowledge.from_dict(payload)
        result.capability_questions = original_knowledge.capability_questions
        result.source_excerpt = original_knowledge.source_excerpt
        return result
    
    @override
    def answer_backward_inquiry(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str
    ) -> tuple[str, str | None, float]:
        knowledge_json = json.dumps(knowledge.to_dict(), ensure_ascii=False)
        prompt = self.prompt_loader.get_prompt("read.answer.backward", title=title, knowledge_json=knowledge_json, question=question)
        payload = self._json_call(prompt)
        answered = str(payload.get("answered_content", ""))
        remaining = payload.get("remaining_question")
        remaining = str(remaining) if remaining is not None else None
        confidence = float(payload.get("confidence", 0.0))
        return answered, remaining, confidence
    
    @override
    def detect_information_gaps_from_knowledge(
        self,
        knowledge: ReaderKnowledge,
        *,
        title: str
    ) -> list[str]:
        """从知识中检测信息缺口（父节点专用），返回需要向上游询问的问题列表"""
        knowledge_json = json.dumps(knowledge.to_dict(), ensure_ascii=False)
        prompt = self.prompt_loader.get_prompt("read.gap.detect_from_knowledge", title=title, knowledge_json=knowledge_json)
        payload = self._json_call(prompt)
        gaps = payload.get("gaps", [])
        return [str(g) for g in gaps]
    
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
