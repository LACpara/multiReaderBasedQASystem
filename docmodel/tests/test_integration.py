"""
集成测试。

测试完整工作流：摄入 → 加载 → 切分 → 检索 → 计算。
"""

import math
import tempfile
import unittest
from pathlib import Path

import pytest

from docmodel import Book, DocView, Granularity, Source, Span
from docmodel.ingest.markdown import MarkdownIngester
from docmodel.splitters import RecursiveSplitter, RegexSplitter, SentenceSplitter, WindowSplitter


class TestEndToEndWorkflow(unittest.TestCase):
    """端到端工作流测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self) -> None:
        md_content = """# 第一章 引言

这是引言部分的内容。介绍了本书的主题和目标。

# 第二章 基础概念

本章介绍基础概念。包括核心定义和基本原理。

# 第三章 高级主题

高级主题的详细讨论。深入探讨复杂问题。
"""
        input_file = Path(self.temp_dir) / "book.md"
        input_file.write_text(md_content, encoding="utf-8")

        output_dir = Path(self.temp_dir) / "book_output"
        ingester = MarkdownIngester(book_id="test-book")
        ingester.ingest(input_file, output_dir)

        book = Book.load(output_dir)
        root = book.root_view()

        assert root.length > 0
        assert "引言" in root.text()

        splitter = RecursiveSplitter(separators=["\n\n", "\n", "。"], max_size=50)
        chunks = root.split(splitter)

        assert len(chunks) >= 1

        hits = root.search("基础")
        assert len(hits) >= 1

        for hit in hits:
            assert "基础" in hit.text()


class TestMarkdownToBookToView(unittest.TestCase):
    """Markdown → Book → DocView 完整流程测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_markdown_to_view(self) -> None:
        md_content = """# 标题一

第一段内容。

# 标题二

第二段内容。
"""
        input_file = Path(self.temp_dir) / "test.md"
        input_file.write_text(md_content, encoding="utf-8")

        output_dir = Path(self.temp_dir) / "output"
        MarkdownIngester().ingest(input_file, output_dir)

        book = Book.load(output_dir)
        view = book.root_view()

        assert "标题一" in view.text() or "第一段内容" in view.text()


class TestSplitAndSearch(unittest.TestCase):
    """切分与检索集成测试。"""

    def test_split_then_search(self) -> None:
        source = Source(
            source_id="test",
            text="第一章 引言。这是引言内容。第二章 方法。这是方法内容。第三章 结论。这是结论内容。",
        )
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=source.length)
        view = DocView([span], sources)

        splitter = SentenceSplitter(lang="zh")
        sentences = view.split(splitter)

        assert len(sentences) >= 1

        for sentence in sentences:
            text = sentence.text()
            if "引言" in text:
                hits = sentence.search("引言")
                assert len(hits) >= 1


class TestFoldWithSplit(unittest.TestCase):
    """Fold 与 Split 集成测试。"""

    def test_fold_after_split(self) -> None:
        source = Source(source_id="test", text="Hello World Python")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=source.length)
        view = DocView([span], sources)

        splitter = WindowSplitter(size=10, stride=5)
        windows = view.split(splitter)

        total_length = sum(w.length for w in windows)

        def count_words(acc: int, w: DocView) -> int:
            return acc + 1

        word_count = view.fold(Granularity.WORD, 0, count_words)
        assert word_count == 3


class TestNGramEntropy(unittest.TestCase):
    """N-gram 熵计算测试（设计文档示例）。"""

    def test_ngram_entropy(self) -> None:
        class NGramState:
            def __init__(self, n: int) -> None:
                self.n = n
                self.counts: dict[str, int] = {}
                self.buf = ""

            def update(self, ch: str) -> None:
                self.buf = (self.buf + ch)[-self.n :]
                if len(self.buf) == self.n:
                    self.counts[self.buf] = self.counts.get(self.buf, 0) + 1

            def entropy(self) -> float:
                if not self.counts:
                    return 0.0
                total = sum(self.counts.values())
                return -sum(
                    (c / total) * math.log2(c / total) for c in self.counts.values()
                )

        source = Source(source_id="test", text="aabbcc")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=source.length)
        view = DocView([span], sources)

        def step(state: NGramState, char_view: DocView) -> NGramState:
            state.update(char_view.text())
            return state

        state = view.fold(Granularity.CHAR, NGramState(n=2), step)
        entropy = state.entropy()

        assert entropy >= 0
        assert entropy <= math.log2(len(state.counts)) if state.counts else True


class TestRecursiveChunking(unittest.TestCase):
    """递归分块测试。"""

    def test_recursive_chunking(self) -> None:
        source = Source(
            source_id="test",
            text="第一段内容。\n\n第二段内容。\n\n第三段内容。",
        )
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=source.length)
        view = DocView([span], sources)

        splitter = RecursiveSplitter(separators=["\n\n", "\n", "。"], max_size=20)
        chunks = view.split(splitter)

        assert len(chunks) >= 1

        for chunk in chunks:
            assert chunk.length <= 25


class TestMultiSourceBook(unittest.TestCase):
    """多 Source 书本测试。"""

    def test_multi_source_operations(self) -> None:
        source1 = Source(source_id="ch1", text="第一章内容。")
        source2 = Source(source_id="ch2", text="第二章内容。")
        sources = {"ch1": source1, "ch2": source2}

        book = Book(
            book_id="multi",
            sources=sources,
            source_order=["ch1", "ch2"],
        )

        root = book.root_view()

        assert root.length == source1.length + source2.length
        assert "第一章" in root.text()
        assert "第二章" in root.text()

        hits = root.search("章")
        assert len(hits) >= 2


class TestExcerptWithContext(unittest.TestCase):
    """上下文摘录测试。"""

    def test_excerpt_context(self) -> None:
        source = Source(
            source_id="test",
            text="前面的内容。目标内容。后面的内容。",
        )
        sources = {"test": source}

        span = Span(source_id="test", start=7, end=11)
        view = DocView([span], sources)

        excerpt = view.excerpt_with_context(5)

        assert excerpt.length >= view.length


class TestViewHierarchy(unittest.TestCase):
    """视图层级测试。"""

    def test_parent_child_relationship(self) -> None:
        source = Source(source_id="test", text="Hello World Python")
        sources = {"test": source}

        span = Span(source_id="test", start=0, end=source.length)
        parent = DocView([span], sources)

        child1 = parent.slice(0, 5)
        child2 = parent.slice(6, 11)

        assert child1.parent == parent
        assert child2.parent == parent
        assert child1.text() == "Hello"
        assert child2.text() == "World"

        grandchild = child1.slice(0, 3)
        assert grandchild.parent == child1
        assert grandchild.text() == "Hel"


class TestProjectCoordinates(unittest.TestCase):
    """坐标投影测试。"""

    def test_project_coordinates(self) -> None:
        source = Source(source_id="test", text="Hello World Python")
        sources = {"test": source}

        span = Span(source_id="test", start=0, end=source.length)
        parent = DocView([span], sources)

        child = parent.slice(6, 11)

        coords = parent.project(child)

        assert len(coords) == 1
        assert coords[0] == (6, 11)


class TestIterGranularities(unittest.TestCase):
    """不同粒度迭代测试。"""

    def test_all_granularities(self) -> None:
        source = Source(
            source_id="test",
            text="Hello World.\nNew line.\n\nNew paragraph.",
        )
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=source.length)
        view = DocView([span], sources)

        chars = list(view.iter(Granularity.CHAR))
        assert len(chars) == view.length

        words = list(view.iter(Granularity.WORD))
        assert len(words) >= 1

        lines = list(view.iter(Granularity.LINE))
        assert len(lines) >= 1

        paragraphs = list(view.iter(Granularity.PARAGRAPH))
        assert len(paragraphs) >= 1

        sources_iter = list(view.iter(Granularity.SOURCE))
        assert len(sources_iter) == 1


class TestSaveLoadRoundtrip(unittest.TestCase):
    """保存加载往返测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_roundtrip_with_operations(self) -> None:
        source1 = Source(source_id="ch1", text="第一章内容。")
        source2 = Source(source_id="ch2", text="第二章内容。")
        original = Book(
            book_id="roundtrip",
            sources={"ch1": source1, "ch2": source2},
            source_order=["ch1", "ch2"],
            title="往返测试",
            language="zh",
        )

        output_dir = Path(self.temp_dir) / "roundtrip"
        original.save(output_dir)

        loaded = Book.load(output_dir)

        assert loaded.book_id == original.book_id
        assert loaded.title == original.title

        root = loaded.root_view()
        hits = root.search("章")
        assert len(hits) >= 2


class TestRegexSplitterIntegration(unittest.TestCase):
    """RegexSplitter 集成测试。"""

    def test_chapter_split(self) -> None:
        source = Source(
            source_id="test",
            text="第一章 引言\n内容一\n第二章 方法\n内容二\n第三章 结论\n内容三",
        )
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=source.length)
        view = DocView([span], sources)

        splitter = RegexSplitter(pattern=r"^第[一二三四五六七八九十]+章")
        chapters = view.split(splitter)

        assert len(chapters) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
