"""
Book 协议实现。

处理 manifest.json 和书本目录的加载/保存。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from docmodel.core import DocView, Source, Span


@dataclass
class SourceInfo:
    """manifest.json 中的 source 元信息。"""
    source_id: str
    path: str
    order: int
    encoding: str = "utf-8"
    sha256: Optional[str] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Book:
    """
    书本抽象，管理一组有序的 Source。
    提供 load/save 静态方法处理 manifest.json。
    """
    book_id: str
    sources: Dict[str, Source]
    source_order: List[str]
    title: Optional[str] = None
    language: Optional[str] = None
    tags: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> Book:
        """从目录加载书本。"""
        book_dir = Path(path)

        if not book_dir.exists():
            raise FileNotFoundError(f"Book directory not found: {book_dir}")

        manifest_path = book_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {book_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        schema_version = manifest.get("schema_version", "1.0")
        if not schema_version.startswith("1."):
            raise ValueError(f"Unsupported manifest version: {schema_version}")

        book_id = manifest.get("book_id", book_dir.name)
        title = manifest.get("title")
        language = manifest.get("language")
        tags = manifest.get("tags", {})

        sources: Dict[str, Source] = {}
        source_order: List[str] = []

        for source_info in manifest.get("sources", []):
            info = SourceInfo(
                source_id=source_info["source_id"],
                path=source_info["path"],
                order=source_info["order"],
                encoding=source_info.get("encoding", "utf-8"),
                sha256=source_info.get("sha256"),
                meta=source_info.get("meta", {}),
            )

            source_file = book_dir / info.path
            if not source_file.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")

            with open(source_file, "r", encoding=info.encoding) as f:
                text = f.read()

            if info.sha256:
                actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                if actual_sha256 != info.sha256:
                    raise ValueError(
                        f"SHA256 mismatch for {info.source_id}: "
                        f"expected {info.sha256}, got {actual_sha256}"
                    )

            source = Source(
                source_id=info.source_id,
                text=text,
                meta={
                    "filename": info.path,
                    "char_count": len(text),
                    "sha256": info.sha256,
                    "order_index": info.order,
                    **info.meta,
                },
            )

            sources[info.source_id] = source
            source_order.append(info.source_id)

        source_order.sort(key=lambda sid: sources[sid].meta.get("order_index", 0))

        return cls(
            book_id=book_id,
            sources=sources,
            source_order=source_order,
            title=title,
            language=language,
            tags=tags,
        )

    def save(self, path: Path | str) -> None:
        """保存书本到目录。"""
        book_dir = Path(path)
        book_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "schema_version": "1.0",
            "book_id": self.book_id,
            "title": self.title,
            "language": self.language,
            "sources": [],
            "tags": dict(self.tags),
        }

        for i, source_id in enumerate(self.source_order):
            source = self.sources[source_id]

            source_file = book_dir / f"{i:03d}_{source_id}.txt"
            with open(source_file, "w", encoding="utf-8") as f:
                f.write(source.text)

            sha256 = hashlib.sha256(source.text.encode("utf-8")).hexdigest()[:16]

            manifest["sources"].append({
                "source_id": source_id,
                "path": source_file.name,
                "order": i,
                "encoding": "utf-8",
                "sha256": sha256,
                "meta": dict(source.meta),
            })

        manifest_path = book_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def root_view(self) -> DocView:
        """创建根视图，包含所有 Source。"""
        spans: List[Span] = []
        for source_id in self.source_order:
            source = self.sources[source_id]
            spans.append(Span(
                source_id=source_id,
                start=0,
                end=source.length,
            ))

        return DocView(
            spans=spans,
            sources=self.sources,
            parent=None,
            tags=self.tags,
        )

    def source_view(self, source_id: str) -> DocView:
        """获取单个 Source 的视图。"""
        if source_id not in self.sources:
            raise KeyError(f"Source not found: {source_id}")

        source = self.sources[source_id]
        span = Span(source_id=source_id, start=0, end=source.length)

        return DocView(
            spans=[span],
            sources=self.sources,
            parent=None,
            tags=source.meta,
        )

    def __repr__(self) -> str:
        return f"Book(book_id={self.book_id!r}, sources={len(self.sources)})"

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self):
        return iter(self.source_order)

    def __getitem__(self, source_id: str) -> Source:
        return self.sources[source_id]
