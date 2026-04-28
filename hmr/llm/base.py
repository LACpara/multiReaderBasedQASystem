from __future__ import annotations

from typing import Protocol

from hmr.domain import ActivationDecision, ReaderAnswer, ReaderKnowledge


class LLMClient(Protocol):
    """Low-level provider boundary for actual remote LLM calls."""

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Return a raw completion string from the configured model provider."""


class ReaderLLMService(Protocol):
    """High-level semantic operations required by the core Reader system."""

    def extract_knowledge(self, text: str, *, title: str) -> ReaderKnowledge:
        """Turn a text span into structured local Reader knowledge."""

    def build_capability_questions(self, knowledge: ReaderKnowledge, *, title: str) -> list[str]:
        """Generate questions that describe what this Reader can answer."""

    def evaluate_activation(self, knowledge: ReaderKnowledge, question: str) -> ActivationDecision:
        """Decide whether the Reader should answer the external question."""

    def answer_question(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str,
    ) -> ReaderAnswer:
        """Produce one grounded partial answer from local Reader knowledge."""

    def merge_answers(self, question: str, answers: list[ReaderAnswer]) -> str:
        """Merge activated Readers' answers into a compact final response."""
