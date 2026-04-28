from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class SemanticTextSplitter:
    """Small semantic-ish splitter that preserves paragraph continuity."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def split(self, text: str) -> list[str]:
        paragraphs = self._paragraphs(text)
        chunks = self._pack(paragraphs)
        logger.debug("Split text into %s chunks", len(chunks))
        return chunks

    def _paragraphs(self, text: str) -> list[str]:
        raw_parts = re.split(r"\n\s*\n", text.strip())
        parts: list[str] = []
        for part in raw_parts:
            if len(part) <= self.max_chars:
                parts.append(part.strip())
                continue
            parts.extend(self._split_long_paragraph(part))
        return [part for part in parts if part]

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[。！？.!?])\s+", paragraph.strip())
        return self._pack([sentence.strip() for sentence in sentences if sentence.strip()])

    def _pack(self, units: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for unit in units:
            size = len(unit)
            if current and current_size + size > self.max_chars:
                chunks.append("\n\n".join(current))
                current = [unit]
                current_size = size
            else:
                current.append(unit)
                current_size += size

        if current:
            chunks.append("\n\n".join(current))
        return chunks
