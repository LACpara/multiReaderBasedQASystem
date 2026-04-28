from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from hmr.domain import ActivationDecision, ReaderAnswer, ReaderKnowledge
from hmr.llm.base import LLMClient

logger = logging.getLogger(__name__)


class PromptedReaderLLMService:
    """LLM-backed implementation of high-level Reader operations.

    It depends only on the low-level LLMClient protocol, not on a specific vendor.
    The demo quick-start uses HeuristicReaderLLMService, while production can swap
    this service in without changing storage, vector search, or core orchestration.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract_knowledge(self, text: str, *, title: str) -> ReaderKnowledge:
        prompt = self._knowledge_prompt(text, title)
        payload = self._json_call(prompt)
        return ReaderKnowledge.from_dict({**payload, "source_excerpt": text[:500]})

    def build_capability_questions(self, knowledge: ReaderKnowledge, *, title: str) -> list[str]:
        prompt = self._questions_prompt(knowledge, title)
        payload = self._json_call(prompt)
        questions = payload.get("capability_questions", [])
        return [str(question) for question in questions][:10]

    def evaluate_activation(self, knowledge: ReaderKnowledge, question: str) -> ActivationDecision:
        prompt = self._activation_prompt(knowledge, question)
        payload = self._json_call(prompt)
        return ActivationDecision(
            should_answer=bool(payload.get("should_answer", False)),
            score=float(payload.get("score", 0.0)),
            sub_question=str(payload.get("sub_question", question)),
            reason=str(payload.get("reason", "")),
        )

    def answer_question(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str,
    ) -> ReaderAnswer:
        prompt = self._answer_prompt(knowledge, question)
        payload = self._json_call(prompt)
        return ReaderAnswer(
            reader_id=reader_id,
            title=title,
            answer=str(payload.get("answer", "")),
            confidence=float(payload.get("confidence", 0.5)),
            source_excerpt=knowledge.source_excerpt,
        )

    def merge_answers(self, question: str, answers: list[ReaderAnswer]) -> str:
        prompt = self._merge_prompt(question, answers)
        return self.client.complete(prompt, temperature=0.0, max_tokens=900).strip()

    def _json_call(self, prompt: str) -> dict[str, Any]:
        raw = self.client.complete(prompt, temperature=0.0, max_tokens=1100)
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

    def _knowledge_prompt(self, text: str, title: str) -> str:
        return f"""
你是分层 Reader 系统中的知识提炼器。请只基于输入文本提炼结构化知识。
返回严格 JSON，字段为 summary, entities, relations, exceptions。

标题：{title}
文本：
{text}
""".strip()

    def _questions_prompt(self, knowledge: ReaderKnowledge, title: str) -> str:
        return f"""
请根据 Reader 知识生成它能够回答的问题边界。返回严格 JSON：
{{"capability_questions": ["..."]}}

标题：{title}
知识：{json.dumps(knowledge.to_dict(), ensure_ascii=False)}
""".strip()

    def _activation_prompt(self, knowledge: ReaderKnowledge, question: str) -> str:
        return f"""
判断该 Reader 是否应回答用户问题。只基于 Reader 知识，不要外推。
返回严格 JSON：should_answer(boolean), score(0-1), sub_question, reason。

问题：{question}
知识：{json.dumps(knowledge.to_dict(), ensure_ascii=False)}
""".strip()

    def _answer_prompt(self, knowledge: ReaderKnowledge, question: str) -> str:
        return f"""
请只基于 Reader 知识回答问题，可部分回答。返回严格 JSON：
{{"answer": "...", "confidence": 0.0}}

问题：{question}
知识：{json.dumps(knowledge.to_dict(), ensure_ascii=False)}
""".strip()

    def _merge_prompt(self, question: str, answers: list[ReaderAnswer]) -> str:
        serialized = [asdict(answer) for answer in answers]
        return f"""
请整合多个 Reader 的部分回答，去除重复、保留高置信度信息。
只能使用给定回答中的信息，无法确定则说明不足。

问题：{question}
Reader 回答：{json.dumps(serialized, ensure_ascii=False)}
""".strip()
