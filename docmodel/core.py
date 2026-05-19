"""
核心数据模型：Source, Span, DocView, Granularity

实现设计文档中定义的核心抽象。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from docmodel.splitters import Splitter


class Granularity(Enum):
    """迭代粒度枚举。"""
    CHAR = "char"
    WORD = "word"
    SENTENCE = "sentence"
    LINE = "line"
    PARAGRAPH = "paragraph"
    SOURCE = "source"


@dataclass(frozen=True)
class Source:
    """
    底层数据源，一份原始字符流。
    '书本'的一个 .txt 文件对应一个 Source。
    """
    source_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class Span:
    """
    原子坐标，使用半开区间 [start, end)。
    """
    source_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Span start cannot be negative: {self.start}")
        if self.end < self.start:
            raise ValueError(f"Span end ({self.end}) cannot be less than start ({self.start})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, offset: int) -> bool:
        return self.start <= offset < self.end

    def overlaps(self, other: Span) -> bool:
        if self.source_id != other.source_id:
            return False
        return self.start < other.end and other.start < self.end


class DocView:
    """
    文档视图，唯一的文档抽象。
    '整本书'、'第3章'、'某段落'、'某句话' 全部是 DocView。
    视图本身不持有文本，只持有对 Source 的引用 + 一组有序 Span。
    """

    def __init__(
        self,
        spans: Sequence[Span],
        sources: Mapping[str, Source],
        parent: Optional[DocView] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._spans = tuple(spans)
        self._sources = sources
        self._parent = parent
        self._tags = MappingProxyType(dict(tags)) if tags else MappingProxyType({})

        self._validate()

    def _validate(self) -> None:
        for span in self._spans:
            if span.source_id not in self._sources:
                raise ValueError(f"Source not found: {span.source_id}")
            source = self._sources[span.source_id]
            if span.end > source.length:
                raise ValueError(
                    f"Span end ({span.end}) exceeds source length ({source.length}) "
                    f"for source {span.source_id}"
                )

    @property
    def spans(self) -> Tuple[Span, ...]:
        return self._spans

    @property
    def sources(self) -> Mapping[str, Source]:
        return self._sources

    @property
    def parent(self) -> Optional[DocView]:
        return self._parent

    @property
    def tags(self) -> Mapping[str, Any]:
        return self._tags

    @property
    def length(self) -> int:
        return sum(span.length for span in self._spans)

    def text(self) -> str:
        parts = []
        for span in self._spans:
            source = self._sources[span.source_id]
            parts.append(source.text[span.start : span.end])
        return "".join(parts)

    def slice(self, start: int, end: int) -> DocView:
        if start < 0:
            raise ValueError(f"Slice start cannot be negative: {start}")
        if end < start:
            raise ValueError(f"Slice end ({end}) cannot be less than start ({start})")
        if end > self.length:
            raise ValueError(f"Slice end ({end}) exceeds view length ({self.length})")

        if start == end:
            return DocView([], self._sources, parent=self)

        new_spans = []
        current_offset = 0

        for span in self._spans:
            span_start = current_offset
            span_end = current_offset + span.length

            if span_end <= start:
                current_offset = span_end
                continue
            if span_start >= end:
                break

            local_start = max(0, start - span_start)
            local_end = min(span.length, end - span_start)

            new_span = Span(
                source_id=span.source_id,
                start=span.start + local_start,
                end=span.start + local_end,
            )
            new_spans.append(new_span)
            current_offset = span_end

        return DocView(new_spans, self._sources, parent=self, tags=self._tags)

    def slice_by_span(self, span: Span) -> DocView:
        if span.source_id not in self._sources:
            raise ValueError(f"Source not found: {span.source_id}")
        return DocView([span], self._sources, parent=self, tags=self._tags)

    def split(self, splitter: Splitter) -> list[DocView]:
        text = self.text()
        regions = splitter(text)

        result = []
        for local_start, local_end in regions:
            child = self.slice(local_start, local_end)
            result.append(child)

        return result

    def search(
        self,
        query: Union[str, re.Pattern, Callable[[str], Iterator[Tuple[int, int]]]],
        *,
        overlapping: bool = False,
        limit: Optional[int] = None,
    ) -> list[DocView]:
        text = self.text()
        matches: list[Tuple[int, int]] = []

        if isinstance(query, str):
            pattern = re.compile(re.escape(query))
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end()))
        elif isinstance(query, re.Pattern):
            for m in query.finditer(text):
                matches.append((m.start(), m.end()))
        elif callable(query):
            matches = list(query(text))
        else:
            raise TypeError(f"query must be str, re.Pattern, or callable, got {type(query)}")

        if not overlapping and matches:
            matches = self._remove_overlaps(matches)

        if limit is not None:
            matches = matches[:limit]

        return [self.slice(start, end) for start, end in matches]

    def _remove_overlaps(self, matches: list[Tuple[int, int]]) -> list[Tuple[int, int]]:
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda x: x[0])
        result = [sorted_matches[0]]

        for start, end in sorted_matches[1:]:
            last_start, last_end = result[-1]
            if start >= last_end:
                result.append((start, end))

        return result

    def iter(self, granularity: Granularity) -> Iterator[DocView]:
        if granularity == Granularity.CHAR:
            yield from self._iter_chars()
        elif granularity == Granularity.WORD:
            yield from self._iter_words()
        elif granularity == Granularity.SENTENCE:
            yield from self._iter_sentences()
        elif granularity == Granularity.LINE:
            yield from self._iter_lines()
        elif granularity == Granularity.PARAGRAPH:
            yield from self._iter_paragraphs()
        elif granularity == Granularity.SOURCE:
            yield from self._iter_sources()
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    def _iter_chars(self) -> Iterator[DocView]:
        for i in range(self.length):
            yield self.slice(i, i + 1)

    def _iter_words(self) -> Iterator[DocView]:
        text = self.text()
        pattern = re.compile(r"\S+")
        for m in pattern.finditer(text):
            yield self.slice(m.start(), m.end())

    def _iter_sentences(self) -> Iterator[DocView]:
        text = self.text()
        pattern = re.compile(r"[^。！？.!?]+[。！？.!?]?\s*")
        for m in pattern.finditer(text):
            content = m.group().strip()
            if content:
                yield self.slice(m.start(), m.start() + len(content))

    def _iter_lines(self) -> Iterator[DocView]:
        text = self.text()
        lines = text.split("\n")
        offset = 0
        for line in lines:
            if line:
                yield self.slice(offset, offset + len(line))
            offset += len(line) + 1

    def _iter_paragraphs(self) -> Iterator[DocView]:
        text = self.text()
        pattern = re.compile(r"[^\n]+(?:\n[^\n]+)*")
        for m in pattern.finditer(text):
            content = m.group().strip()
            if content:
                yield self.slice(m.start(), m.start() + len(content))

    def _iter_sources(self) -> Iterator[DocView]:
        for span in self._spans:
            yield self.slice_by_span(span)

    def fold(
        self,
        granularity: Granularity,
        init: Any,
        step: Callable[[Any, DocView], Any],
        *,
        progress: bool = False,
    ) -> Any:
        state = init
        for view in self.iter(granularity):
            state = step(state, view)
        return state

    def project(self, child: DocView) -> list[Tuple[int, int]]:
        result = []
        child_spans = list(child.spans)

        for child_span in child_spans:
            local_start = self._find_local_offset(child_span.source_id, child_span.start)
            local_end = self._find_local_offset(child_span.source_id, child_span.end)

            if local_start is not None and local_end is not None:
                result.append((local_start, local_end))

        return result

    def _find_local_offset(self, source_id: str, global_offset: int) -> Optional[int]:
        local_offset = 0

        for span in self._spans:
            if span.source_id == source_id:
                if span.start <= global_offset <= span.end:
                    return local_offset + (global_offset - span.start)
            local_offset += span.length

        return None

    def to_book_spans(self) -> list[Span]:
        return list(self._spans)

    def excerpt_with_context(self, context_chars: int = 200) -> DocView:
        if len(self._spans) == 0:
            return self

        if self._parent is not None:
            local_positions = self._parent.project(self)
            
            if local_positions:
                parent_start = max(0, local_positions[0][0] - context_chars)
                parent_end = min(self._parent.length, local_positions[-1][1] + context_chars)
                return self._parent.slice(parent_start, parent_end)

        return self

    def __repr__(self) -> str:
        preview = self.text()[:50]
        if len(preview) == 50:
            preview += "..."
        return f"DocView(length={self.length}, text={preview!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DocView):
            return NotImplemented
        return (
            self._spans == other._spans
            and self._sources is other._sources
            and self._tags == other._tags
        )

    def __hash__(self) -> int:
        return hash((self._spans, id(self._sources), tuple(sorted(self._tags.items()))))
