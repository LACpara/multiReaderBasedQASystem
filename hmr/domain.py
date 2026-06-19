from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fileinput
from typing import Any, Literal


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
class BackwardInquiry:
    """逆向求知请求包"""
    inquiry_id: str
    source_reader_id: str  # 当前来源 reader
    origin_source_reader_id: str  # 原始发起 reader
    target_reader_id: str
    question: str
    depth: int  # 逆向传递深度
    source_type: str  # sibling / child / parent
    answered_content_chain: list[str] = field(default_factory=list)  # 累积的回答链
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "answered_content_chain": self.answered_content_chain}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackwardInquiry":
        return cls(
            inquiry_id=payload["inquiry_id"],
            source_reader_id=payload["source_reader_id"],
            origin_source_reader_id=payload.get("origin_source_reader_id", payload["source_reader_id"]),
            target_reader_id=payload["target_reader_id"],
            question=payload["question"],
            depth=payload["depth"],
            source_type=payload.get("source_type", "sibling"),
            answered_content_chain=list(payload.get("answered_content_chain", [])),
            created_at=payload.get("created_at", utc_now_iso()),
        )


@dataclass(slots=True)
class PartialAnswer:
    """部分回答包"""
    answer_id: str
    inquiry_id: str
    answering_reader_id: str
    answered_content: str  # 已回答的部分
    remaining_question: str | None  # 剩余未回答的问题
    confidence: float
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PartialAnswer":
        return cls(
            answer_id=payload["answer_id"],
            inquiry_id=payload["inquiry_id"],
            answering_reader_id=payload["answering_reader_id"],
            answered_content=payload["answered_content"],
            remaining_question=payload.get("remaining_question"),
            confidence=payload["confidence"],
            created_at=payload.get("created_at", utc_now_iso()),
        )


@dataclass(slots=True)
class CompleteAnswer:
    """完整回答包（最终回传）"""
    answer_id: str
    inquiry_id: str
    full_answer: str
    answering_chain: list[str]  # 参与回答的 Reader ID 链
    is_full: bool = True  # 是否完整回答
    remaining_question: str = ""  # 剩余未回答的问题
    confidence: float = 0.0  # 回答置信度
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "answering_chain": self.answering_chain}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompleteAnswer":
        return cls(
            answer_id=payload["answer_id"],
            inquiry_id=payload["inquiry_id"],
            full_answer=payload["full_answer"],
            answering_chain=list(payload.get("answering_chain", [])),
            is_full=payload.get("is_full", True),
            remaining_question=payload.get("remaining_question", ""),
            confidence=payload.get("confidence", 0.0),
            created_at=payload.get("created_at", utc_now_iso()),
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

    channels: dict[Literal["prev", "next"], "channel"] = field(default_factory=dict)

    # 逆向求知相关字段
    emitted_inquiries: list[BackwardInquiry] = field(default_factory=list)  # 发出的逆向请求
    received_inquiries: list[BackwardInquiry] = field(default_factory=list)  # 收到的逆向请求
    emitted_partial_answers: list[PartialAnswer] = field(default_factory=list)  # 发出的部分回答
    received_complete_answers: list[CompleteAnswer] = field(default_factory=list)  # 收到的完整回答

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
    analysis: str = "..."


@dataclass(slots=True)
class RetrievalResult:
    """Final response and trace data for one query."""

    question: str
    answer: str
    candidates: list[VectorCandidate]
    activated_answers: list[ReaderAnswer]
