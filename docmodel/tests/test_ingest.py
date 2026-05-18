"""
摄入器模块单元测试。

测试 MarkdownIngester 和其他摄入器。
"""

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from docmodel.ingest.markdown import MarkdownIngester


class TestMarkdownIngester(unittest.TestCase):
    """MarkdownIngester 测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ingest_simple_markdown(self) -> None:
        md_content = """# Chapter 1

This is the first chapter.

# Chapter 2

This is the second chapter.
"""
        input_file = Path(self.temp_dir) / "test.md"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        output_dir = Path(self.temp_dir) / "output"
        ingester = MarkdownIngester(book_id="test-book")
        result = ingester.ingest(input_file, output_dir)

        assert result == output_dir
        assert (output_dir / "manifest.json").exists()

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["book_id"] == "test-book"
        assert len(manifest["sources"]) == 2

    def test_ingest_no_h1_headings(self) -> None:
        md_content = """This is content without H1 headings.

Just plain text.
"""
        input_file = Path(self.temp_dir) / "no_h1.md"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        output_dir = Path(self.temp_dir) / "output_no_h1"
        ingester = MarkdownIngester()
        result = ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest["sources"]) == 1

    def test_ingest_nonexistent_file_raises(self) -> None:
        ingester = MarkdownIngester()
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            ingester.ingest("/nonexistent/file.md", self.temp_dir)

    def test_ingest_non_markdown_raises(self) -> None:
        txt_file = Path(self.temp_dir) / "test.txt"
        txt_file.write_text("content")

        ingester = MarkdownIngester()
        with pytest.raises(ValueError, match="Expected .md file"):
            ingester.ingest(txt_file, self.temp_dir)

    def test_ingest_unicode_content(self) -> None:
        md_content = """# 第一章

这是中文内容。

# 第二章

更多中文内容。
"""
        input_file = Path(self.temp_dir) / "chinese.md"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        output_dir = Path(self.temp_dir) / "unicode_output"
        ingester = MarkdownIngester(book_id="chinese-book")
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["book_id"] == "chinese-book"
        assert len(manifest["sources"]) == 2

    def test_sanitize_title(self) -> None:
        ingester = MarkdownIngester()

        assert ingester._sanitize_title("Simple Title") == "Simple_Title"
        assert ingester._sanitize_title("Title: With/Special\\Chars") == "Title__With_Special_Chars"
        assert ingester._sanitize_title("") == "section"

    def test_split_by_h1(self) -> None:
        ingester = MarkdownIngester()

        content = """# First

Content 1

# Second

Content 2"""

        sections = ingester._split_by_h1(content)

        assert len(sections) == 2
        assert sections[0][0] == "First"
        assert sections[1][0] == "Second"

    def test_split_by_h1_empty_content(self) -> None:
        ingester = MarkdownIngester()
        sections = ingester._split_by_h1("")

        assert len(sections) == 1
        assert sections[0][0] == "main"
        assert sections[0][1] == ""

    def test_create_manifest(self) -> None:
        ingester = MarkdownIngester()
        sections = [
            ("Chapter 1", "Content 1"),
            ("Chapter 2", "Content 2"),
        ]

        manifest = ingester._create_manifest("test-book", sections, "test.md")

        assert manifest["book_id"] == "test-book"
        assert manifest["schema_version"] == "1.0"
        assert len(manifest["sources"]) == 2
        assert manifest["sources"][0]["meta"]["title"] == "Chapter 1"

    def test_ingest_with_string_paths(self) -> None:
        md_content = "# Test\n\nContent"
        input_file = Path(self.temp_dir) / "string_path.md"
        input_file.write_text(md_content)

        output_dir = str(Path(self.temp_dir) / "string_output")
        ingester = MarkdownIngester()
        result = ingester.ingest(str(input_file), output_dir)

        assert Path(output_dir).exists()

    def test_ingest_creates_output_directory(self) -> None:
        md_content = "# Test\n\nContent"
        input_file = Path(self.temp_dir) / "new_dir_test.md"
        input_file.write_text(md_content)

        output_dir = Path(self.temp_dir) / "new" / "nested" / "output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        assert output_dir.exists()

    def test_ingest_preserves_content(self) -> None:
        md_content = """# Chapter 1

Paragraph 1.

Paragraph 2.

# Chapter 2

Another paragraph.
"""
        input_file = Path(self.temp_dir) / "preserve.md"
        input_file.write_text(md_content)

        output_dir = Path(self.temp_dir) / "preserve_output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "000_Chapter_1.txt", "r", encoding="utf-8") as f:
            content = f.read()

        assert "Paragraph 1" in content
        assert "Paragraph 2" in content

    def test_ingest_sha256_in_manifest(self) -> None:
        md_content = "# Test\n\nContent"
        input_file = Path(self.temp_dir) / "sha256.md"
        input_file.write_text(md_content)

        output_dir = Path(self.temp_dir) / "sha256_output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "sha256" in manifest["sources"][0]
        assert len(manifest["sources"][0]["sha256"]) == 16


class TestMarkdownIngesterEdgeCases(unittest.TestCase):
    """MarkdownIngester 边界情况测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ingest_multiple_h1_same_line(self) -> None:
        md_content = "# Title1\n# Title2\nContent"
        input_file = Path(self.temp_dir) / "multi_h1.md"
        input_file.write_text(md_content)

        output_dir = Path(self.temp_dir) / "multi_output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest["sources"]) == 2

    def test_ingest_h2_not_split(self) -> None:
        md_content = """## H2 Title

This should not be split by H2.

# H1 Title

This should be split.
"""
        input_file = Path(self.temp_dir) / "h2_test.md"
        input_file.write_text(md_content)

        output_dir = Path(self.temp_dir) / "h2_output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest["sources"]) == 1

    def test_ingest_empty_file(self) -> None:
        input_file = Path(self.temp_dir) / "empty.md"
        input_file.write_text("")

        output_dir = Path(self.temp_dir) / "empty_output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest["sources"]) == 1

    def test_ingest_only_h1_no_content(self) -> None:
        md_content = "# Title\n\n# Another Title"
        input_file = Path(self.temp_dir) / "only_h1.md"
        input_file.write_text(md_content)

        output_dir = Path(self.temp_dir) / "only_h1_output"
        ingester = MarkdownIngester()
        ingester.ingest(input_file, output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest["sources"]) == 2

    def test_sanitize_long_title(self) -> None:
        ingester = MarkdownIngester()
        long_title = "A" * 100

        sanitized = ingester._sanitize_title(long_title)

        assert len(sanitized) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
