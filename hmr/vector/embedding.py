from __future__ import annotations

import hashlib
import math
import re


class HashEmbeddingModel:
    """Small local embedding model for deterministic demo indexing.

    It is not a semantic model, but it keeps the vector database path realistic and
    removes external API dependencies from quick start.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return self._normalize(vector)

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,2}", text.lower())

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
