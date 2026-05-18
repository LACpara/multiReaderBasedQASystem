"""
Book 模块单元测试。

测试 Book 类的加载、保存和视图创建功能。
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

from docmodel.book import Book, SourceInfo
from docmodel.core import DocView, Source, Span


class TestSourceInfo(unittest.TestCase):
    """SourceInfo 数据类测试。"""

    def test_source_info_creation(self) -> None:
        info = SourceInfo(
            source_id="ch1",
            path="chapter1.txt",
            order=0,
            encoding="utf-8",
            sha256="abc123",
            meta={"title": "Chapter 1"},
        )
        assert info.source_id == "ch1"
        assert info.path == "chapter1.txt"
        assert info.order == 0
        assert info.encoding == "utf-8"
        assert info.sha256 == "abc123"
        assert info.meta["title"] == "Chapter 1"

    def test_source_info_defaults(self) -> None:
        info = SourceInfo(
            source_id="ch1",
            path="chapter1.txt",
            order=0,
        )
        assert info.encoding == "utf-8"
        assert info.sha256 is None
        assert info.meta == {}


class TestBookCreation(unittest.TestCase):
    """Book 创建测试。"""

    def test_book_creation(self) -> None:
        source = Source(source_id="ch1", text="Hello World")
        book = Book(
            book_id="test-book",
            sources={"ch1": source},
            source_order=["ch1"],
        )
        assert book.book_id == "test-book"
        assert len(book) == 1
        assert "ch1" in book

    def test_book_with_metadata(self) -> None:
        source = Source(source_id="ch1", text="Hello")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
            title="Test Book",
            language="zh",
            tags={"author": "Test Author"},
        )
        assert book.title == "Test Book"
        assert book.language == "zh"
        assert book.tags["author"] == "Test Author"

    def test_book_iteration(self) -> None:
        source1 = Source(source_id="ch1", text="First")
        source2 = Source(source_id="ch2", text="Second")
        book = Book(
            book_id="test",
            sources={"ch1": source1, "ch2": source2},
            source_order=["ch1", "ch2"],
        )
        source_ids = list(book)
        assert source_ids == ["ch1", "ch2"]

    def test_book_getitem(self) -> None:
        source = Source(source_id="ch1", text="Hello")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
        )
        assert book["ch1"] == source

    def test_book_repr(self) -> None:
        source = Source(source_id="ch1", text="Hello")
        book = Book(
            book_id="test-book",
            sources={"ch1": source},
            source_order=["ch1"],
        )
        repr_str = repr(book)
        assert "Book" in repr_str
        assert "test-book" in repr_str


class TestBookLoad(unittest.TestCase):
    """Book.load 方法测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_simple_book(self) -> None:
        book_dir = Path(self.temp_dir) / "test_book"
        book_dir.mkdir()

        manifest = {
            "schema_version": "1.0",
            "book_id": "test-book",
            "title": "Test Book",
            "sources": [
                {
                    "source_id": "ch1",
                    "path": "chapter1.txt",
                    "order": 0,
                    "encoding": "utf-8",
                }
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with open(book_dir / "chapter1.txt", "w", encoding="utf-8") as f:
            f.write("Hello World")

        book = Book.load(book_dir)

        assert book.book_id == "test-book"
        assert book.title == "Test Book"
        assert len(book) == 1
        assert book["ch1"].text == "Hello World"

    def test_load_multiple_sources(self) -> None:
        book_dir = Path(self.temp_dir) / "multi_book"
        book_dir.mkdir()

        manifest = {
            "schema_version": "1.0",
            "book_id": "multi",
            "sources": [
                {"source_id": "ch1", "path": "ch1.txt", "order": 0},
                {"source_id": "ch2", "path": "ch2.txt", "order": 1},
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with open(book_dir / "ch1.txt", "w", encoding="utf-8") as f:
            f.write("Chapter 1")

        with open(book_dir / "ch2.txt", "w", encoding="utf-8") as f:
            f.write("Chapter 2")

        book = Book.load(book_dir)

        assert len(book) == 2
        assert book.source_order == ["ch1", "ch2"]

    def test_load_nonexistent_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Book directory not found"):
            Book.load("/nonexistent/path")

    def test_load_missing_manifest_raises(self) -> None:
        book_dir = Path(self.temp_dir) / "no_manifest"
        book_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            Book.load(book_dir)

    def test_load_missing_source_file_raises(self) -> None:
        book_dir = Path(self.temp_dir) / "missing_source"
        book_dir.mkdir()

        manifest = {
            "schema_version": "1.0",
            "book_id": "test",
            "sources": [
                {"source_id": "ch1", "path": "nonexistent.txt", "order": 0}
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            Book.load(book_dir)

    def test_load_with_sha256_validation(self) -> None:
        book_dir = Path(self.temp_dir) / "sha256_book"
        book_dir.mkdir()

        text = "Hello World"
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        manifest = {
            "schema_version": "1.0",
            "book_id": "test",
            "sources": [
                {
                    "source_id": "ch1",
                    "path": "ch1.txt",
                    "order": 0,
                    "sha256": sha256,
                }
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with open(book_dir / "ch1.txt", "w", encoding="utf-8") as f:
            f.write(text)

        book = Book.load(book_dir)
        assert book["ch1"].text == text

    def test_load_sha256_mismatch_raises(self) -> None:
        book_dir = Path(self.temp_dir) / "mismatch_book"
        book_dir.mkdir()

        manifest = {
            "schema_version": "1.0",
            "book_id": "test",
            "sources": [
                {
                    "source_id": "ch1",
                    "path": "ch1.txt",
                    "order": 0,
                    "sha256": "wronghash123456",
                }
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with open(book_dir / "ch1.txt", "w", encoding="utf-8") as f:
            f.write("Hello World")

        with pytest.raises(ValueError, match="SHA256 mismatch"):
            Book.load(book_dir)

    def test_load_unsupported_version_raises(self) -> None:
        book_dir = Path(self.temp_dir) / "version_book"
        book_dir.mkdir()

        manifest = {
            "schema_version": "2.0",
            "book_id": "test",
            "sources": [],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="Unsupported manifest version"):
            Book.load(book_dir)

    def test_load_with_tags(self) -> None:
        book_dir = Path(self.temp_dir) / "tags_book"
        book_dir.mkdir()

        manifest = {
            "schema_version": "1.0",
            "book_id": "test",
            "tags": {"author": "Test", "year": 2024},
            "sources": [
                {"source_id": "ch1", "path": "ch1.txt", "order": 0}
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with open(book_dir / "ch1.txt", "w", encoding="utf-8") as f:
            f.write("Content")

        book = Book.load(book_dir)
        assert book.tags["author"] == "Test"
        assert book.tags["year"] == 2024


class TestBookSave(unittest.TestCase):
    """Book.save 方法测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_simple_book(self) -> None:
        source = Source(source_id="ch1", text="Hello World")
        book = Book(
            book_id="test-book",
            sources={"ch1": source},
            source_order=["ch1"],
            title="Test Book",
        )

        output_dir = Path(self.temp_dir) / "output"
        book.save(output_dir)

        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "000_ch1.txt").exists()

        with open(output_dir / "000_ch1.txt", "r", encoding="utf-8") as f:
            assert f.read() == "Hello World"

    def test_save_multiple_sources(self) -> None:
        source1 = Source(source_id="ch1", text="Chapter 1")
        source2 = Source(source_id="ch2", text="Chapter 2")
        book = Book(
            book_id="multi",
            sources={"ch1": source1, "ch2": source2},
            source_order=["ch1", "ch2"],
        )

        output_dir = Path(self.temp_dir) / "multi_output"
        book.save(output_dir)

        assert (output_dir / "000_ch1.txt").exists()
        assert (output_dir / "001_ch2.txt").exists()

    def test_save_creates_directory(self) -> None:
        source = Source(source_id="ch1", text="Content")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
        )

        output_dir = Path(self.temp_dir) / "new_dir" / "nested"
        book.save(output_dir)

        assert output_dir.exists()

    def test_save_manifest_content(self) -> None:
        source = Source(source_id="ch1", text="Content")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
            title="Test Title",
            language="zh",
        )

        output_dir = Path(self.temp_dir) / "manifest_test"
        book.save(output_dir)

        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["book_id"] == "test"
        assert manifest["title"] == "Test Title"
        assert manifest["language"] == "zh"
        assert manifest["schema_version"] == "1.0"

    def test_save_and_load_roundtrip(self) -> None:
        source1 = Source(source_id="ch1", text="First chapter content")
        source2 = Source(source_id="ch2", text="Second chapter content")
        original = Book(
            book_id="roundtrip",
            sources={"ch1": source1, "ch2": source2},
            source_order=["ch1", "ch2"],
            title="Roundtrip Test",
            language="zh",
            tags={"author": "Test"},
        )

        output_dir = Path(self.temp_dir) / "roundtrip"
        original.save(output_dir)
        loaded = Book.load(output_dir)

        assert loaded.book_id == original.book_id
        assert loaded.title == original.title
        assert loaded.language == original.language
        assert len(loaded) == len(original)
        assert loaded["ch1"].text == source1.text
        assert loaded["ch2"].text == source2.text


class TestBookViews(unittest.TestCase):
    """Book 视图创建测试。"""

    def test_root_view(self) -> None:
        source1 = Source(source_id="ch1", text="Hello")
        source2 = Source(source_id="ch2", text="World")
        book = Book(
            book_id="test",
            sources={"ch1": source1, "ch2": source2},
            source_order=["ch1", "ch2"],
        )

        root = book.root_view()

        assert isinstance(root, DocView)
        assert root.length == 10
        assert root.text() == "HelloWorld"

    def test_source_view(self) -> None:
        source = Source(source_id="ch1", text="Hello World")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
        )

        view = book.source_view("ch1")

        assert isinstance(view, DocView)
        assert view.text() == "Hello World"

    def test_source_view_nonexistent_raises(self) -> None:
        source = Source(source_id="ch1", text="Hello")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
        )

        with pytest.raises(KeyError, match="Source not found"):
            book.source_view("nonexistent")


class TestBookEdgeCases(unittest.TestCase):
    """Book 边界情况测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_book(self) -> None:
        book = Book(
            book_id="empty",
            sources={},
            source_order=[],
        )

        assert len(book) == 0
        assert book.root_view().text() == ""

    def test_single_char_source(self) -> None:
        source = Source(source_id="single", text="A")
        book = Book(
            book_id="test",
            sources={"single": source},
            source_order=["single"],
        )

        assert book.root_view().text() == "A"

    def test_unicode_content(self) -> None:
        source = Source(source_id="unicode", text="你好世界 🌍")
        book = Book(
            book_id="unicode",
            sources={"unicode": source},
            source_order=["unicode"],
        )

        output_dir = Path(self.temp_dir) / "unicode"
        book.save(output_dir)
        loaded = Book.load(output_dir)

        assert loaded["unicode"].text == "你好世界 🌍"

    def test_large_source(self) -> None:
        large_text = "a" * 100000
        source = Source(source_id="large", text=large_text)
        book = Book(
            book_id="large",
            sources={"large": source},
            source_order=["large"],
        )

        output_dir = Path(self.temp_dir) / "large"
        book.save(output_dir)
        loaded = Book.load(output_dir)

        assert len(loaded["large"].text) == 100000

    def test_load_with_string_path(self) -> None:
        book_dir = Path(self.temp_dir) / "string_path"
        book_dir.mkdir()

        manifest = {
            "schema_version": "1.0",
            "book_id": "test",
            "sources": [
                {"source_id": "ch1", "path": "ch1.txt", "order": 0}
            ],
        }

        with open(book_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with open(book_dir / "ch1.txt", "w", encoding="utf-8") as f:
            f.write("Content")

        book = Book.load(str(book_dir))
        assert book.book_id == "test"

    def test_save_with_string_path(self) -> None:
        source = Source(source_id="ch1", text="Content")
        book = Book(
            book_id="test",
            sources={"ch1": source},
            source_order=["ch1"],
        )

        output_dir = str(Path(self.temp_dir) / "string_output")
        book.save(output_dir)

        assert Path(output_dir).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
