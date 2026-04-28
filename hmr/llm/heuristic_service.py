from __future__ import annotations

import logging
import re
from collections import Counter

from hmr.domain import ActivationDecision, ReaderAnswer, ReaderKnowledge

logger = logging.getLogger(__name__)


class HeuristicReaderLLMService:
    """Deterministic stand-in for an LLM-backed Reader service.

    This keeps the demo runnable without API keys. Production code can replace this
    object with PromptedReaderLLMService or any class matching ReaderLLMService.
    """

    def extract_knowledge(self, text: str, *, title: str) -> ReaderKnowledge:
        logger.debug("Extracting heuristic knowledge for title=%s", title)
        sentences = self._sentences(text)
        return ReaderKnowledge(
            summary=self._summary(sentences),
            entities=self._entities(text),
            relations=self._relation_sentences(sentences),
            exceptions=self._exception_sentences(sentences),
            source_excerpt=self._excerpt(text),
        )

    def build_capability_questions(self, knowledge: ReaderKnowledge, *, title: str) -> list[str]:
        logger.debug("Building capability questions for title=%s", title)
        seeds = self._question_seeds(knowledge, title)
        questions = [f"{seed} 是什么？" for seed in seeds]
        questions.extend(f"{seed} 如何工作？" for seed in seeds[:3])
        questions.append(f"{title} 的核心内容是什么？")
        questions.append(f"{title} 中有哪些限制、风险或例外？")
        return self._deduplicate(questions)[:8]

    def evaluate_activation(self, knowledge: ReaderKnowledge, question: str) -> ActivationDecision:
        score = self._overlap_score(question, knowledge.searchable_text())
        should_answer = score > 0.0
        reason = "matched local knowledge tokens" if should_answer else "no meaningful overlap"
        logger.debug("Activation score=%.3f reason=%s", score, reason)
        return ActivationDecision(
            should_answer=should_answer,
            score=score,
            sub_question=question,
            reason=reason,
        )

    def answer_question(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str,
    ) -> ReaderAnswer:
        sentences = self._sentences(knowledge.searchable_text())
        selected = self._select_sentences(question, sentences)
        answer_text = self._format_answer(selected, title)
        confidence = min(1.0, 0.35 + 0.15 * len(selected))
        return ReaderAnswer(
            reader_id=reader_id,
            title=title,
            answer=answer_text,
            confidence=confidence,
            source_excerpt=knowledge.source_excerpt,
        )

    def merge_answers(self, question: str, answers: list[ReaderAnswer]) -> str:
        if not answers:
            return "没有 Reader 通过自评估激活，因此无法基于当前知识库回答该问题。"

        lines = [f"问题：{question}", "", "综合回答："]
        seen: set[str] = set()
        for answer in sorted(answers, key=lambda item: item.confidence, reverse=True):
            for bullet in self._answer_bullets(answer.answer):
                normalized = self._normalize(bullet)
                if normalized in seen:
                    continue
                seen.add(normalized)
                lines.append(f"- {bullet}")
        lines.append("")
        lines.append("参考 Reader：" + ", ".join(answer.title for answer in answers))
        return "\n".join(lines)

    def _summary(self, sentences: list[str]) -> str:
        return " ".join(sentences[:3]) if sentences else ""

    def _entities(self, text: str) -> list[str]:
        english = re.findall(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b", text)
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,8}(?:系统|阶段|机制|策略|数据库|智能体|检索|索引|能力)", text)
        return self._deduplicate([*english, *chinese_terms])[:12]

    def _relation_sentences(self, sentences: list[str]) -> list[str]:
        markers = ("是", "采用", "通过", "基于", "由", "负责", "用于", "contains", "uses", "stores")
        return [sentence for sentence in sentences if any(marker in sentence for marker in markers)][:8]

    def _exception_sentences(self, sentences: list[str]) -> list[str]:
        markers = ("不", "但", "除非", "限制", "风险", "问题", "难以", "不能", "avoid", "except")
        return [sentence for sentence in sentences if any(marker in sentence for marker in markers)][:6]

    def _question_seeds(self, knowledge: ReaderKnowledge, title: str) -> list[str]:
        candidates = [title, *knowledge.entities, *self._keywords(knowledge.searchable_text())]
        return self._deduplicate([candidate.strip("：:,.，。") for candidate in candidates if candidate])[:6]

    def _keywords(self, text: str) -> list[str]:
        tokens = [token for token in self._tokens(text) if len(token) > 3]
        return [item for item, _ in Counter(tokens).most_common(8)]

    def _select_sentences(self, question: str, sentences: list[str]) -> list[str]:
        scored = [(self._overlap_score(question, sentence), sentence) for sentence in sentences]
        selected = [sentence for score, sentence in sorted(scored, reverse=True) if score > 0.0]
        return self._deduplicate(selected)[:4]

    def _format_answer(self, sentences: list[str], title: str) -> str:
        if not sentences:
            return f"{title}：当前 Reader 只有弱相关信息，无法形成可靠细节。"
        return "\n".join(f"{sentence}" for sentence in sentences)

    def _answer_bullets(self, answer: str) -> list[str]:
        return [line.lstrip("- ").strip() for line in answer.splitlines() if line.strip()]

    def _overlap_score(self, left: str, right: str) -> float:
        left_tokens = set(self._tokens(left))
        right_tokens = set(self._tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        return len(overlap) / max(len(left_tokens), 1)

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,2}", text.lower())

    def _sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[。！？.!?])\s+|\n+", text.strip())
        return [sentence.strip(" -\t") for sentence in raw if sentence.strip()]

    def _excerpt(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:500]

    def _deduplicate(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            normalized = self._normalize(item)
            if normalized in seen:
                continue
            seen.add(normalized)
            output.append(item)
        return output

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.lower())
