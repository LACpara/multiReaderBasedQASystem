"""
Markdown 摄入器实现。

按 H1 标题切分 Markdown 文件。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List


class MarkdownIngester:
    """
    Markdown 摄入器。
    按 # H1 标题切分。
    """

    def __init__(self, book_id: Optional[str] = None) -> None:
        self.book_id = book_id

    def ingest(self, input_path: Path | str, output_dir: Path | str) -> Path:
        input_file = Path(input_path)
        output = Path(output_dir)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not input_file.suffix.lower() == ".md":
            raise ValueError(f"Expected .md file, got {input_file.suffix}")

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        sections = self._split_by_h1(content)

        output.mkdir(parents=True, exist_ok=True)

        book_id = self.book_id or input_file.stem
        manifest = self._create_manifest(book_id, sections, input_file.name)

        for i, (title, text) in enumerate(sections):
            source_file = output / f"{i:03d}_{self._sanitize_title(title)}.txt"
            with open(source_file, "w", encoding="utf-8") as f:
                f.write(text)

        manifest_path = output / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return output

    def _split_by_h1(self, content: str) -> List[tuple[str, str]]:
        pattern = re.compile(r"^#\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(content))

        if not matches:
            return [("main", content)]

        sections: List[tuple[str, str]] = []

        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()

            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(content)

            text = content[start:end].strip()
            sections.append((title, text))

        return sections

    def _sanitize_title(self, title: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", title)
        sanitized = re.sub(r"\s+", "_", sanitized)
        return sanitized[:50] or "section"

    def _create_manifest(
        self, book_id: str, sections: List[tuple[str, str]], original_name: str
    ) -> Dict[str, Any]:
        sources: List[Dict[str, Any]] = []

        for i, (title, text) in enumerate(sections):
            sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            filename = f"{i:03d}_{self._sanitize_title(title)}.txt"

            sources.append({
                "source_id": f"section_{i}",
                "path": filename,
                "order": i,
                "encoding": "utf-8",
                "sha256": sha256,
                "meta": {
                    "title": title,
                    "char_count": len(text),
                    "origin": {"format": "markdown", "original_file": original_name},
                },
            })

        return {
            "schema_version": "1.0",
            "book_id": book_id,
            "title": book_id,
            "language": "unknown",
            "sources": sources,
            "tags": {"origin_format": "markdown"},
        }
