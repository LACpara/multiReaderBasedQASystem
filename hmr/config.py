from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class IngestionConfig:
    """Controls how documents are converted into a Reader tree."""

    max_leaf_chars: int = 900
    max_depth: int = 4
    complexity_threshold: float = 1050.0


@dataclass(slots=True)
class RetrievalConfig:
    """Controls two-stage activation during retrieval."""

    top_k: int = 6
    activation_threshold: float = 0.08
    max_answers: int = 4


@dataclass(slots=True)
class StorageConfig:
    """Persistence locations for structured and vector data."""

    sqlite_path: Path = Path("runtime/hmr_demo.sqlite3")
    chroma_path: Path = Path("runtime/chroma")
    chroma_collection: str = "reader_capabilities"


@dataclass(slots=True)
class AppConfig:
    """Top-level configuration used by the application factory."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
