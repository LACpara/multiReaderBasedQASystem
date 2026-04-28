from __future__ import annotations

import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)


class ComplexityEstimator:
    """Approximates information complexity for demo-time dynamic splitting."""

    def score(self, text: str) -> float:
        tokens = self._tokens(text)
        if not tokens:
            return 0.0
        length_score = len(text)
        entropy_score = self._entropy(tokens) * 80.0
        density_score = self._concept_density(tokens) * 260.0
        total = length_score + entropy_score + density_score
        logger.debug(
            "Complexity score %.2f = length %.2f + entropy %.2f + density %.2f",
            total,
            length_score,
            entropy_score,
            density_score,
        )
        return total

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text.lower())

    def _entropy(self, tokens: list[str]) -> float:
        counts = Counter(tokens)
        total = len(tokens)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def _concept_density(self, tokens: list[str]) -> float:
        concept_tokens = [token for token in tokens if len(token) > 3 or "\u4e00" <= token <= "\u9fff"]
        return len(set(concept_tokens)) / max(len(tokens), 1)
