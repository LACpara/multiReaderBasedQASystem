"""
内置 Splitter 实现。

提供 RegexSplitter, RecursiveSplitter, WindowSplitter, SentenceSplitter, TagSplitter。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Protocol, Tuple

if TYPE_CHECKING:
    pass


class Splitter(Protocol):
    """Splitter 协议：输入文本，返回本地坐标区间列表。"""
    def __call__(self, text: str) -> List[Tuple[int, int]]: ...


@dataclass
class RegexSplitter:
    """
    按正则表达式切分。
    用于按章节标题、空行段落等模式切分。
    """
    pattern: str
    include_match: bool = True

    def __call__(self, text: str) -> List[Tuple[int, int]]:
        compiled = re.compile(self.pattern, re.MULTILINE)
        matches = list(compiled.finditer(text))

        if not matches:
            return [(0, len(text))] if text else []

        regions: List[Tuple[int, int]] = []
        last_end = 0

        for m in matches:
            if self.include_match:
                if m.start() > last_end:
                    regions.append((last_end, m.start()))
                last_end = m.end()
            else:
                if m.start() > last_end:
                    regions.append((last_end, m.start()))
                last_end = m.end()

        if last_end < len(text):
            regions.append((last_end, len(text)))

        return [r for r in regions if r[1] > r[0]]

    def __repr__(self) -> str:
        return f"RegexSplitter(pattern={self.pattern!r})"


@dataclass
class RecursiveSplitter:
    """
    递归字符切分器，类似 LangChain 的 RecursiveCharacterTextSplitter。
    按优先级依次尝试分隔符，直到满足大小约束。
    """
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", ".", " "])
    max_size: int = 500
    overlap: int = 50
    keep_separator: bool = True

    def __call__(self, text: str) -> List[Tuple[int, int]]:
        if not text:
            return []

        if len(text) <= self.max_size:
            return [(0, len(text))]

        return self._split_recursive(text, 0, len(text), 0)

    def _split_recursive(
        self, text: str, start: int, end: int, sep_idx: int
    ) -> List[Tuple[int, int]]:
        chunk = text[start:end]

        if len(chunk) <= self.max_size:
            return [(start, end)]

        if sep_idx >= len(self.separators):
            return self._split_by_size(text, start, end)

        sep = self.separators[sep_idx]
        parts = chunk.split(sep)

        if len(parts) == 1:
            return self._split_recursive(text, start, end, sep_idx + 1)

        regions: List[Tuple[int, int]] = []
        current_start = start

        for i, part in enumerate(parts):
            part_start = current_start
            part_end = part_start + len(part)

            # 根据 keep_separator 决定是否添加分隔符
            if self.keep_separator and i < len(parts) - 1:
                part_end += len(sep)

            if len(part) > self.max_size:
                sub_regions = self._split_recursive(text, part_start, part_end, sep_idx + 1)
                regions.extend(sub_regions)
            else:
                if part.strip():
                    regions.append((part_start, part_end))

            # 更新 current_start 时也要考虑 keep_separator
            current_start = part_end
            if not self.keep_separator and i < len(parts) - 1:
                current_start += len(sep)

        return self._apply_overlap(regions, len(text))

    def _split_by_size(self, text: str, start: int, end: int) -> List[Tuple[int, int]]:
        regions: List[Tuple[int, int]] = []
        pos = start

        while pos < end:
            chunk_end = min(pos + self.max_size, end)
            regions.append((pos, chunk_end))
            pos = chunk_end

        return self._apply_overlap(regions, end)

    def _apply_overlap(self, regions: List[Tuple[int, int]], text_length: int = None) -> List[Tuple[int, int]]:
        if self.overlap <= 0 or len(regions) <= 1:
            return regions

        result: List[Tuple[int, int]] = []
        for i, (start, end) in enumerate(regions):
            if i > 0:
                chunk_size = regions[i][1] - regions[i][0]
                new_start = result[-1][1] - self.overlap
                
                # 确保起始位置不为负
                if new_start < 0:
                    new_start = 0
                
                new_end = new_start + chunk_size
                
                # 确保结束位置不超出原始文本范围
                if text_length is not None and new_end > text_length:
                    new_end = text_length
                
                start, end = new_start, new_end
                
            result.append((start, end))

        return result

    def __repr__(self) -> str:
        return f"RecursiveSplitter(max_size={self.max_size}, overlap={self.overlap})"


@dataclass
class WindowSplitter:
    """
    字符级滑窗切分器。
    配合 fold 做迭代统计。
    """
    size: int = 100
    stride: int = 50

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Window size must be positive: {self.size}")
        if self.stride <= 0:
            raise ValueError(f"Stride must be positive: {self.stride}")

    def __call__(self, text: str) -> List[Tuple[int, int]]:
        if not text:
            return []

        regions: List[Tuple[int, int]] = []
        pos = 0

        while pos < len(text):
            end = min(pos + self.size, len(text))
            regions.append((pos, end))
            pos += self.stride

            if end == len(text):
                break

        return regions

    def __repr__(self) -> str:
        return f"WindowSplitter(size={self.size}, stride={self.stride})"


@dataclass
class SentenceSplitter:
    """
    句子切分器。
    支持中英文句子边界检测。
    """
    lang: str = "zh"

    def __call__(self, text: str) -> List[Tuple[int, int]]:
        if not text:
            return []

        if self.lang == "zh":
            pattern = re.compile(r"[^。！？\n]+[。！？]?\s*")
        else:
            pattern = re.compile(r"[^.!?\n]+[.!?]?\s*")

        regions: List[Tuple[int, int]] = []
        for m in pattern.finditer(text):
            content = m.group().strip()
            if content:
                regions.append((m.start(), m.start() + len(content)))

        return regions

    def __repr__(self) -> str:
        return f"SentenceSplitter(lang={self.lang!r})"


@dataclass
class TagSplitter:
    """
    按已有 tag 边界切分。
    例如按目录预设章节切。
    """
    tag_key: str

    def __call__(self, text: str) -> List[Tuple[int, int]]:
        raise NotImplementedError(
            "TagSplitter requires access to DocView.tags. "
            "Use DocView.split_with_tags() instead."
        )

    def split_with_tags(
        self, view: "DocView", tag_boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        return tag_boundaries

    def __repr__(self) -> str:
        return f"TagSplitter(tag_key={self.tag_key!r})"


def _merge_regions(regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并重叠或相邻的区域。"""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = [sorted_regions[0]]

    for start, end in sorted_regions[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged
