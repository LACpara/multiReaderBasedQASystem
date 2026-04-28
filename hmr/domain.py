from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ReaderKnowledge:
    """Structured local knowledge owned by one Reader."""

    summary: str
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    capability_questions: list[str] = field(default_factory=list)
    source_excerpt: str = ""

    def searchable_text(self) -> str:
        parts = [self.summary, *self.entities, *self.relations, *self.exceptions]
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReaderKnowledge":
        return cls(
            summary=payload.get("summary", ""),
            entities=list(payload.get("entities", [])),
            relations=list(payload.get("relations", [])),
            exceptions=list(payload.get("exceptions", [])),
            capability_questions=list(payload.get("capability_questions", [])),
            source_excerpt=payload.get("source_excerpt", ""),
        )


@dataclass(slots=True)
class ReaderNode:
    """Persistent representation of a Reader in the recursive tree."""

    reader_id: str
    document_id: str
    title: str
    parent_id: str | None
    depth: int
    ordinal: int
    text: str
    knowledge: ReaderKnowledge
    child_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids


@dataclass(slots=True)
class VectorCandidate:
    """Candidate returned by the vector database coarse recall stage."""

    reader_id: str
    score: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActivationDecision:
    """Reader-side self-assessment for whether it should answer."""

    should_answer: bool
    score: float
    sub_question: str
    reason: str


@dataclass(slots=True)
class ReaderAnswer:
    """Partial answer produced by an activated Reader."""

    reader_id: str
    title: str
    answer: str
    confidence: float
    source_excerpt: str


@dataclass(slots=True)
class RetrievalResult:
    """Final response and trace data for one query."""

    question: str
    answer: str
    candidates: list[VectorCandidate]
    activated_answers: list[ReaderAnswer]
