"""
核心模块单元测试。

测试 Source, Span, DocView, Granularity。
"""

import re
import unittest
from types import MappingProxyType

import pytest

from docmodel.core import DocView, Granularity, Source, Span


class TestSource(unittest.TestCase):
    """Source 类测试。"""

    def test_source_creation(self) -> None:
        source = Source(
            source_id="test",
            text="Hello World",
            meta={"filename": "test.txt"},
        )
        assert source.source_id == "test"
        assert source.text == "Hello World"
        assert source.meta["filename"] == "test.txt"

    def test_source_length(self) -> None:
        source = Source(source_id="test", text="Hello")
        assert source.length == 5

    def test_source_empty_text(self) -> None:
        source = Source(source_id="empty", text="")
        assert source.length == 0

    def test_source_frozen(self) -> None:
        source = Source(source_id="test", text="Hello")
        with pytest.raises(AttributeError):
            source.text = "Modified"

    def test_source_meta_frozen(self) -> None:
        source = Source(source_id="test", text="Hello", meta={"key": "value"})
        with pytest.raises(TypeError):
            source.meta["key"] = "modified"

    def test_source_default_meta(self) -> None:
        source = Source(source_id="test", text="Hello")
        assert source.meta == {}

    def test_source_meta_dict_conversion(self) -> None:
        source = Source(source_id="test", text="Hello", meta={"key": "value"})
        assert isinstance(source.meta, MappingProxyType)

    def test_source_unicode_text(self) -> None:
        # 测试基本中文
        source1 = Source(source_id="zh", text="你好世界")
        assert source1.length == 4
        assert source1.text == "你好世界"
        
        # 测试 emoji
        source2 = Source(source_id="emoji", text="🌍🎯")
        assert source2.length == 2
        
        # 测试混合文本
        source3 = Source(source_id="mixed", text="Hello 世界 🌍")
        assert source3.length == 10
        
        # 测试特殊字符
        source4 = Source(source_id="special", text="éñàç")
        assert source4.length == 4

    def test_source_long_text(self) -> None:
        long_text = "a" * 1000000
        source = Source(source_id="long", text=long_text)
        assert source.length == 1000000


class TestSpan(unittest.TestCase):
    """Span 类测试。"""

    def test_span_creation(self) -> None:
        span = Span(source_id="test", start=0, end=10)
        assert span.source_id == "test"
        assert span.start == 0
        assert span.end == 10

    def test_span_length(self) -> None:
        span = Span(source_id="test", start=5, end=15)
        assert span.length == 10

    def test_span_zero_length(self) -> None:
        span = Span(source_id="test", start=5, end=5)
        assert span.length == 0

    def test_span_contains(self) -> None:
        span = Span(source_id="test", start=5, end=15)
        assert span.contains(5)
        assert span.contains(10)
        assert span.contains(14)
        assert not span.contains(4)
        assert not span.contains(15)

    def test_span_overlaps(self) -> None:
        span1 = Span(source_id="test", start=0, end=10)
        span2 = Span(source_id="test", start=5, end=15)
        span3 = Span(source_id="test", start=10, end=20)
        span4 = Span(source_id="other", start=0, end=10)

        assert span1.overlaps(span2)
        assert not span1.overlaps(span3)
        assert not span1.overlaps(span4)

    def test_span_negative_start_raises(self) -> None:
        with pytest.raises(ValueError):
            Span(source_id="test", start=-1, end=10)

    def test_span_end_less_than_start_raises(self) -> None:
        with pytest.raises(ValueError):
            Span(source_id="test", start=10, end=5)

    def test_span_frozen(self) -> None:
        span = Span(source_id="test", start=0, end=10)
        with pytest.raises(AttributeError):
            span.start = 5


class TestDocView(unittest.TestCase):
    """DocView 类测试。"""

    def setUp(self) -> None:
        self.source1 = Source(source_id="ch1", text="Hello World")
        self.source2 = Source(source_id="ch2", text="Python Programming")
        self.sources = {"ch1": self.source1, "ch2": self.source2}

    def test_docview_creation(self) -> None:
        span = Span(source_id="ch1", start=0, end=11)
        view = DocView([span], self.sources)
        assert view.length == 11
        assert view.text() == "Hello World"

    def test_docview_multiple_spans(self) -> None:
        span1 = Span(source_id="ch1", start=0, end=5)
        span2 = Span(source_id="ch2", start=0, end=6)
        view = DocView([span1, span2], self.sources)
        assert view.length == 11
        assert view.text() == "HelloPython"

    def test_docview_empty(self) -> None:
        view = DocView([], self.sources)
        assert view.length == 0
        assert view.text() == ""

    def test_docview_parent(self) -> None:
        span = Span(source_id="ch1", start=0, end=11)
        parent = DocView([span], self.sources)
        child = parent.slice(0, 5)
        assert child.parent == parent

    def test_docview_tags(self) -> None:
        span = Span(source_id="ch1", start=0, end=11)
        view = DocView([span], self.sources, tags={"chapter": 1})
        assert view.tags["chapter"] == 1

    def test_docview_invalid_source_raises(self) -> None:
        span = Span(source_id="nonexistent", start=0, end=10)
        with pytest.raises(ValueError, match="Source not found"):
            DocView([span], self.sources)

    def test_docview_span_exceeds_source_raises(self) -> None:
        span = Span(source_id="ch1", start=0, end=100)
        with pytest.raises(ValueError, match="exceeds source length"):
            DocView([span], self.sources)


class TestDocViewSlice(unittest.TestCase):
    """DocView.slice 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(source_id="test", text="Hello World Python")
        self.sources = {"test": self.source}

    def test_slice_beginning(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        sliced = view.slice(0, 5)
        assert sliced.text() == "Hello"

    def test_slice_middle(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        sliced = view.slice(6, 11)
        assert sliced.text() == "World"

    def test_slice_end(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        sliced = view.slice(12, 18)
        assert sliced.text() == "Python"

    def test_slice_full(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        sliced = view.slice(0, 18)
        assert sliced.text() == "Hello World Python"

    def test_slice_empty(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        sliced = view.slice(5, 5)
        assert sliced.text() == ""
        assert sliced.length == 0

    def test_slice_negative_start_raises(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        with pytest.raises(ValueError, match="cannot be negative"):
            view.slice(-1, 5)

    def test_slice_end_less_than_start_raises(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        with pytest.raises(ValueError, match="cannot be less than start"):
            view.slice(10, 5)

    def test_slice_exceeds_length_raises(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources)
        with pytest.raises(ValueError, match="exceeds view length"):
            view.slice(0, 100)

    def test_slice_preserves_tags(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        view = DocView([span], self.sources, tags={"key": "value"})
        sliced = view.slice(0, 5)
        assert sliced.tags == view.tags

    def test_slice_by_span(self) -> None:
        span = Span(source_id="test", start=6, end=11)
        view = DocView([], self.sources)
        sliced = view.slice_by_span(span)
        assert sliced.text() == "World"
    
    def test_slice_by_cross_multi_spans(self) -> None:
        span1 = Span(source_id="test", start=0, end=5)
        span2 = Span(source_id="test", start=6, end=11)
        view = DocView([span1, span2], self.sources)
        sliced = view.slice(2, 8)
        assert sliced.text() == "lloWor"

    def test_slice_by_span_invalid_source_raises(self) -> None:
        span = Span(source_id="nonexistent", start=0, end=5)
        view = DocView([], self.sources)
        with pytest.raises(ValueError, match="Source not found"):
            view.slice_by_span(span)


class TestDocViewSplit(unittest.TestCase):
    """DocView.split 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(source_id="test", text="Hello World")
        self.sources = {"test": self.source}

    def test_split_with_simple_splitter(self) -> None:
        def simple_splitter(text: str) -> list[tuple[int, int]]:
            return [(0, 5), (6, 11)]

        span = Span(source_id="test", start=0, end=11)
        view = DocView([span], self.sources)
        chunks = view.split(simple_splitter)

        assert len(chunks) == 2
        assert chunks[0].text() == "Hello"
        assert chunks[1].text() == "World"

    def test_split_empty_result(self) -> None:
        def empty_splitter(text: str) -> list[tuple[int, int]]:
            return []

        span = Span(source_id="test", start=0, end=11)
        view = DocView([span], self.sources)
        chunks = view.split(empty_splitter)
        assert len(chunks) == 0


class TestDocViewSearch(unittest.TestCase):
    """DocView.search 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(source_id="test", text="Hello World Hello")
        self.sources = {"test": self.source}

    def test_search_string(self) -> None:
        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        results = view.search("Hello")

        assert len(results) == 2
        assert results[0].text() == "Hello"
        assert results[1].text() == "Hello"

    def test_search_regex(self) -> None:
        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        pattern = re.compile(r"Hel+o")
        results = view.search(pattern)

        assert len(results) == 2

    def tedt_sesrch_regex_case2(self) -> None:
        """
        测试在多个文档中搜索正则表达式。
        """
        text1 = "date1: 2018-12-"
        text2 = "23, data2: 2012-11-23. oooo"
        span1 = Span(source_id="d1", start=0, end=len(text1))
        span2 = Span(source_id="d2", start=0, end=len(text2))
        source1 = Source(source_id="d1", text=text1)
        source2 = Source(source_id="d2", text=text2)

        doc = DocView([span1, span2], {"d1": source1, "d2": source2})
        results = doc.search(re.compile(r"\d{4}-\d{2}-\d{2}"))

        assert len(results) == 2
        assert results[0].text() == "2018-12-23"
        assert results[1].text() == "2012-11-23"

    def test_search_callable(self) -> None:
        def custom_search(text: str):
            for m in re.finditer(r"World", text):
                yield (m.start(), m.end())

        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        results = view.search(custom_search)

        assert len(results) == 1
        assert results[0].text() == "World"

    def test_search_no_match(self) -> None:
        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        results = view.search("NotFound")

        assert len(results) == 0

    def test_search_limit(self) -> None:
        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        results = view.search("Hello", limit=1)

        assert len(results) == 1

    def test_search_overlapping(self) -> None:
        source = Source(source_id="test", text="ababa")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], sources)

        results = view.search("aba", overlapping=True)
        assert len(results) == 1

        results = view.search("aba", overlapping=False)
        assert len(results) == 1

    def test_search_invalid_query_type_raises(self) -> None:
        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        with pytest.raises(TypeError, match="must be str, re.Pattern, or callable"):
            view.search(123)  # type: ignore


class TestDocViewIter(unittest.TestCase):
    """DocView.iter 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(
            source_id="test",
            text="Hello World.\nThis is a test.\n\nAnother paragraph."
        )
        self.sources = {"test": self.source}
        self.source_length = len(self.source.text)

    def test_iter_chars(self) -> None:
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], self.sources)
        chars = list(view.iter(Granularity.CHAR))

        assert len(chars) == 5
        assert chars[0].text() == "H"
        assert chars[4].text() == "o"

    def test_iter_words(self) -> None:
        span = Span(source_id="test", start=0, end=11)
        view = DocView([span], self.sources)
        words = list(view.iter(Granularity.WORD))

        assert len(words) == 2
        assert words[0].text() == "Hello"
        assert words[1].text() == "World"

    def test_iter_sentences(self) -> None:
        span = Span(source_id="test", start=0, end=17)
        view = DocView([span], self.sources)
        sentences = list(view.iter(Granularity.SENTENCE))

        assert len(sentences) >= 1
        assert "Hello World" in sentences[0].text()

    def test_iter_lines(self) -> None:
        span = Span(source_id="test", start=0, end=self.source_length)
        view = DocView([span], self.sources)
        lines = list(view.iter(Granularity.LINE))

        assert len(lines) >= 1
        assert lines[0].text() == "Hello World."

    def test_iter_paragraphs(self) -> None:
        span = Span(source_id="test", start=0, end=self.source_length)
        view = DocView([span], self.sources)
        paragraphs = list(view.iter(Granularity.PARAGRAPH))

        assert len(paragraphs) >= 1

    def test_iter_sources(self) -> None:
        span = Span(source_id="test", start=0, end=self.source_length)
        view = DocView([span], self.sources)
        sources = list(view.iter(Granularity.SOURCE))

        assert len(sources) == 1

        text = "I love China."
        span2 = Span(source_id="test2", start=0, end=len(text))
        source = Source(source_id="test2", text=text)
        sources = {"test2": source}
        sources.update(self.sources)
        docview = DocView([span, span2], sources)
        results = list(docview.iter(Granularity.SOURCE))

        assert len(results) == 2

    def test_iter_empty_view(self) -> None:
        view = DocView([], self.sources)
        chars = list(view.iter(Granularity.CHAR))
        assert len(chars) == 0
        chars = list(view.iter(Granularity.WORD))
        assert len(chars) == 0
        chars = list(view.iter(Granularity.SENTENCE))
        assert len(chars) == 0
        chars = list(view.iter(Granularity.PARAGRAPH))
        assert len(chars) == 0
        chars = list(view.iter(Granularity.LINE))
        assert len(chars) == 0
        chars = list(view.iter(Granularity.SOURCE))
        assert len(chars) == 0


class TestDocViewFold(unittest.TestCase):
    """DocView.fold 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(source_id="test", text="Hello")
        self.sources = {"test": self.source}

    def test_fold_sum(self) -> None:
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], self.sources)

        def sum_chars(total: int, char_view: DocView) -> int:
            return total + len(char_view.text())

        result = view.fold(Granularity.CHAR, 0, sum_chars)
        assert result == 5

    def test_fold_collect(self) -> None:
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], self.sources)

        def collect(chars: list, char_view: DocView) -> list:
            chars.append(char_view.text())
            return chars

        result = view.fold(Granularity.CHAR, [], collect)
        assert result == ["H", "e", "l", "l", "o"]

    def test_fold_empty_view(self) -> None:
        view = DocView([], self.sources)

        def step(state: int, _: DocView) -> int:
            return state + 1

        result = view.fold(Granularity.CHAR, 0, step)
        assert result == 0


class TestDocViewProject(unittest.TestCase):
    """DocView.project 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(source_id="test", text="Hello World Python")
        self.sources = {"test": self.source}
    
        # 增加的新测试用例
        self.source1 = Source(source_id="ch1", text="Chapter 1 content")
        self.source2 = Source(source_id="ch2", text="Chapter 2 content")
        self.source3 = Source(source_id="ch3", text="Chapter 3 content")
        self.sources = {
            "ch1": self.source1, 
            "ch2": self.source2, 
            "ch3": self.source3,
            **self.sources
        }

    def test_project_child(self) -> None:
        span = Span(source_id="test", start=0, end=18)
        parent = DocView([span], self.sources)
        child = parent.slice(6, 11)

        result = parent.project(child)
        assert len(result) == 1
        assert result[0] == (6, 11)

    def test_project_child_different_source(self) -> None:
        source2 = Source(source_id="other", text="Different")
        sources = {"test": self.source, "other": source2}

        span1 = Span(source_id="test", start=0, end=11)
        parent = DocView([span1], sources)

        span2 = Span(source_id="other", start=0, end=9)
        child = DocView([span2], sources)

        result = parent.project(child)
        assert len(result) == 0

    def test_to_book_spans_singlesource(self) -> None:
        span = Span(source_id="test", start=5, end=10)
        view = DocView([span], self.sources)

        result = view.to_book_spans()
        assert len(result) == 1
        assert result[0] == span

    def test_project_multisource_child(self) -> None:
        """测试跨多个 Source 的子视图投影"""
        # 父视图跨越 3 个章节
        parent_spans = [
            Span("ch1", 0, len(self.source1.text)),
            Span("ch2", 0, len(self.source2.text)),
            Span("ch3", 0, len(self.source3.text)),
        ]
        parent = DocView(parent_spans, self.sources)
        
        # 子视图跨越章节边界
        child_spans = [
            Span("ch2", 5, len(self.source2.text)),
            Span("ch3", 0, 10),
        ]
        child = DocView(child_spans, self.sources)
        
        result = parent.project(child)
        # 应该返回父视图中的本地坐标区间
        assert len(result) == 2

    def test_project_nested_view(self) -> None:
        """测试嵌套视图的链式投影"""
        parent = DocView(
            [Span("ch1", 0, len(self.source1.text))], 
            self.sources
        )
        child = parent.slice(8, 16)  # "content"
        grandchild = child.slice(2, 6)  # "nten"
        
        # grandchild 在 child 中的位置
        result1 = child.project(grandchild)
        assert result1 == [(2, 6)]
        
        # grandchild 在 parent 中的位置
        result2 = parent.project(grandchild)
        assert result2 == [(10, 14)]  # 8+2=10, 8+6=14

    def test_project_empty_view(self) -> None:
        """测试空视图投影"""
        parent = DocView([Span("ch1", 0, 10)], self.sources)
        child = DocView([], self.sources)
        
        result = parent.project(child)
        assert len(result) == 0

    def test_to_book_spans_multisource(self) -> None:
        """测试多 Source 视图的全局 Span 转换"""
        view = DocView(
            [Span("ch1", 5, 10), Span("ch2", 0, 5)], 
            self.sources
        )
        
        spans = view.to_book_spans()
        assert len(spans) == 2
        assert spans[0].source_id == "ch1"
        assert spans[1].source_id == "ch2"

class TestDocViewExcerpt(unittest.TestCase):
    """DocView.excerpt_with_context 方法测试。"""

    def setUp(self) -> None:
        self.source = Source(source_id="test", text="Hello World Python Programming")
        self.sources = {"test": self.source}

    def test_excerpt_with_context(self) -> None:
        parent_span = Span(source_id="test", start=0, end=30)
        parent = DocView([parent_span], self.sources)
        
        child = parent.slice(6, 11)
        excerpt = child.excerpt_with_context(5)

        assert excerpt.text() == "ello World Pyth"

    def test_excerpt_at_beginning(self) -> None:
        parent_span = Span(source_id="test", start=0, end=30)
        parent = DocView([parent_span], self.sources)
        
        child = parent.slice(0, 5)
        excerpt = child.excerpt_with_context(6)

        assert excerpt.text() == "Hello World"

    def test_excerpt_at_end(self) -> None:
        parent_span = Span(source_id="test", start=0, end=30)
        parent = DocView([parent_span], self.sources)
        
        child = parent.slice(20, 30)
        excerpt = child.excerpt_with_context(7)

        assert excerpt.text() == "ython Programming"

    def test_excerpt_with_large_context(self) -> None:
        parent_span = Span(source_id="test", start=0, end=30)
        parent = DocView([parent_span], self.sources)
        
        child = parent.slice(6, 11)
        excerpt = child.excerpt_with_context(100)

        assert excerpt.text() == "Hello World Python Programming"

    def test_excerpt_empty_view(self) -> None:
        view = DocView([], self.sources)
        excerpt = view.excerpt_with_context(10)

        assert excerpt.length == 0
        assert excerpt.text() == ""

    def test_excerpt_no_parent(self) -> None:
        span = Span(source_id="test", start=6, end=11)
        view = DocView([span], self.sources)
        excerpt = view.excerpt_with_context(5)

        assert excerpt.text() == "World"


class TestDocViewRepr(unittest.TestCase):
    """DocView.__repr__ 方法测试。"""

    def test_repr_short_text(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], sources)

        repr_str = repr(view)
        assert "DocView" in repr_str
        assert "length=5" in repr_str

    def test_repr_long_text(self) -> None:
        source = Source(source_id="test", text="a" * 100)
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=100)
        view = DocView([span], sources)

        repr_str = repr(view)
        assert "..." in repr_str


class TestDocViewEquality(unittest.TestCase):
    """DocView 相等性测试。"""

    def test_equal_views(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=5)

        view1 = DocView([span], sources)
        view2 = DocView([span], sources)

        assert view1 == view2

    def test_different_spans(self) -> None:
        source = Source(source_id="test", text="Hello World")
        sources = {"test": source}
        span1 = Span(source_id="test", start=0, end=5)
        span2 = Span(source_id="test", start=6, end=11)

        view1 = DocView([span1], sources)
        view2 = DocView([span2], sources)

        assert view1 != view2

    def test_not_equal_to_non_view(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], sources)

        assert view != "not a view"
        assert view != 123


class TestGranularity(unittest.TestCase):
    """Granularity 枚举测试。"""

    def test_granularity_values(self) -> None:
        assert Granularity.CHAR.value == "char"
        assert Granularity.WORD.value == "word"
        assert Granularity.SENTENCE.value == "sentence"
        assert Granularity.LINE.value == "line"
        assert Granularity.PARAGRAPH.value == "paragraph"
        assert Granularity.SOURCE.value == "source"

    def test_granularity_members(self) -> None:
        assert len(Granularity) == 6


class TestEdgeCases(unittest.TestCase):
    """边界情况测试。"""

    def test_empty_source_text(self) -> None:
        source = Source(source_id="empty", text="")
        assert source.length == 0

    def test_single_char_source(self) -> None:
        source = Source(source_id="single", text="a")
        sources = {"single": source}
        span = Span(source_id="single", start=0, end=1)
        view = DocView([span], sources)

        assert view.length == 1
        assert view.text() == "a"

    def test_span_at_boundary(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        span = Span(source_id="test", start=0, end=5)
        view = DocView([span], sources)

        assert view.slice(0, 0).text() == ""
        assert view.slice(5, 5).text() == ""

    def test_view_with_no_spans(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        view = DocView([], sources)

        assert view.length == 0
        assert view.text() == ""
        assert list(view.iter(Granularity.CHAR)) == []

    def test_search_in_empty_view(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        view = DocView([], sources)

        results = view.search("test")
        assert len(results) == 0

    def test_split_empty_view(self) -> None:
        source = Source(source_id="test", text="Hello")
        sources = {"test": source}
        view = DocView([], sources)

        def splitter(text: str) -> list[tuple[int, int]]:
            return [(0, len(text))] if text else []

        chunks = view.split(splitter)
        assert len(chunks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
