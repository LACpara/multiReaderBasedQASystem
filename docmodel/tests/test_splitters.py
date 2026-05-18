"""
Splitter 模块单元测试。

测试 RegexSplitter, RecursiveSplitter, WindowSplitter, SentenceSplitter, TagSplitter。
"""

import unittest

import pytest

from docmodel.splitters import (
    RegexSplitter,
    RecursiveSplitter,
    SentenceSplitter,
    TagSplitter,
    WindowSplitter,
    _merge_regions,
)


class TestRegexSplitter(unittest.TestCase):
    """RegexSplitter 测试。"""

    def test_split_by_pattern(self) -> None:
        splitter = RegexSplitter(pattern=r"\n\n+")
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        regions = splitter(text)

        assert len(regions) == 3
        assert text[regions[0][0] : regions[0][1]] == "Paragraph 1"
        assert text[regions[1][0] : regions[1][1]] == "Paragraph 2"
        assert text[regions[2][0] : regions[2][1]] == "Paragraph 3"

    def test_split_no_match(self) -> None:
        splitter = RegexSplitter(pattern=r"XXX")
        text = "Hello World"
        regions = splitter(text)

        assert len(regions) == 1
        assert regions[0] == (0, 11)

    def test_split_empty_text(self) -> None:
        splitter = RegexSplitter(pattern=r"\n")
        regions = splitter("")

        assert len(regions) == 0

    def test_split_chapter_pattern(self) -> None:
        splitter = RegexSplitter(pattern=r"^第[一二三四五六七八九十百]+章")
        text = "第一章内容\n第二章内容\n第三章内容"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_split_include_match_false(self) -> None:
        splitter = RegexSplitter(pattern=r"\n", include_match=False)
        text = "Line1\nLine2\nLine3"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_repr(self) -> None:
        splitter = RegexSplitter(pattern=r"\n+")
        assert "RegexSplitter" in repr(splitter)
        assert r"\n+" in repr(splitter)


class TestRecursiveSplitter(unittest.TestCase):
    """RecursiveSplitter 测试。"""

    def test_split_small_text(self) -> None:
        splitter = RecursiveSplitter(max_size=100)
        text = "Short text"
        regions = splitter(text)

        assert len(regions) == 1
        assert regions[0] == (0, 10)

    def test_split_by_separator(self) -> None:
        splitter = RecursiveSplitter(separators=["\n\n", "\n"], max_size=20)
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_split_with_overlap(self) -> None:
        splitter = RecursiveSplitter(max_size=20, overlap=5)
        text = "This is a longer text that needs to be split into multiple chunks."
        regions = splitter(text)

        assert len(regions) >= 2

    def test_split_empty_text(self) -> None:
        splitter = RecursiveSplitter()
        regions = splitter("")

        assert len(regions) == 0

    def test_split_chinese_text(self) -> None:
        splitter = RecursiveSplitter(separators=["\n\n", "\n", "。"], max_size=20)
        text = "这是第一句话。这是第二句话。这是第三句话。"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_split_no_valid_separator(self) -> None:
        splitter = RecursiveSplitter(separators=["XXX"], max_size=10)
        text = "This is a long text without the separator"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_split_preserves_content(self) -> None:
        splitter = RecursiveSplitter(max_size=20)
        text = "Hello World"
        regions = splitter(text)

        total_text = "".join(text[r[0] : r[1]] for r in regions)
        assert total_text == text

    def test_repr(self) -> None:
        splitter = RecursiveSplitter(max_size=500, overlap=50)
        assert "RecursiveSplitter" in repr(splitter)
        assert "500" in repr(splitter)


class TestWindowSplitter(unittest.TestCase):
    """WindowSplitter 测试。"""

    def test_split_with_window(self) -> None:
        splitter = WindowSplitter(size=10, stride=5)
        text = "Hello World Python"
        regions = splitter(text)

        assert len(regions) >= 2
        assert regions[0] == (0, 10)

    def test_split_text_shorter_than_window(self) -> None:
        splitter = WindowSplitter(size=100, stride=50)
        text = "Short"
        regions = splitter(text)

        assert len(regions) == 1
        assert regions[0] == (0, 5)

    def test_split_empty_text(self) -> None:
        splitter = WindowSplitter()
        regions = splitter("")

        assert len(regions) == 0

    def test_split_with_equal_size_and_stride(self) -> None:
        splitter = WindowSplitter(size=10, stride=10)
        text = "01234567890123456789"
        regions = splitter(text)

        assert len(regions) == 2
        assert regions[0] == (0, 10)
        assert regions[1] == (10, 20)

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            WindowSplitter(size=0, stride=5)

    def test_invalid_stride_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            WindowSplitter(size=10, stride=0)

    def test_negative_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            WindowSplitter(size=-1, stride=5)

    def test_repr(self) -> None:
        splitter = WindowSplitter(size=100, stride=50)
        assert "WindowSplitter" in repr(splitter)
        assert "100" in repr(splitter)


class TestSentenceSplitter(unittest.TestCase):
    """SentenceSplitter 测试。"""

    def test_split_chinese_sentences(self) -> None:
        splitter = SentenceSplitter(lang="zh")
        text = "这是第一句话。这是第二句话！这是第三句话？"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_split_english_sentences(self) -> None:
        splitter = SentenceSplitter(lang="en")
        text = "First sentence. Second sentence! Third sentence?"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_split_empty_text(self) -> None:
        splitter = SentenceSplitter()
        regions = splitter("")

        assert len(regions) == 0

    def test_split_single_sentence(self) -> None:
        splitter = SentenceSplitter(lang="zh")
        text = "这是一句话"
        regions = splitter(text)

        assert len(regions) == 1

    def test_split_preserves_content(self) -> None:
        splitter = SentenceSplitter(lang="zh")
        text = "第一句。第二句。"
        regions = splitter(text)

        total_text = "".join(text[r[0] : r[1]] for r in regions)
        assert total_text == text

    def test_repr(self) -> None:
        splitter = SentenceSplitter(lang="zh")
        assert "SentenceSplitter" in repr(splitter)
        assert "zh" in repr(splitter)


class TestTagSplitter(unittest.TestCase):
    """TagSplitter 测试。"""

    def test_call_raises_not_implemented(self) -> None:
        splitter = TagSplitter(tag_key="chapter")
        with pytest.raises(NotImplementedError, match="requires access to DocView.tags"):
            splitter("some text")

    def test_split_with_tags(self) -> None:
        splitter = TagSplitter(tag_key="chapter")
        boundaries = [(0, 10), (10, 20)]
        result = splitter.split_with_tags(None, boundaries)  # type: ignore

        assert result == boundaries

    def test_repr(self) -> None:
        splitter = TagSplitter(tag_key="chapter")
        assert "TagSplitter" in repr(splitter)
        assert "chapter" in repr(splitter)


class TestMergeRegions(unittest.TestCase):
    """_merge_regions 辅助函数测试。"""

    def test_merge_empty(self) -> None:
        result = _merge_regions([])
        assert result == []

    def test_merge_single(self) -> None:
        result = _merge_regions([(0, 10)])
        assert result == [(0, 10)]

    def test_merge_adjacent(self) -> None:
        result = _merge_regions([(0, 10), (10, 20)])
        assert result == [(0, 20)]

    def test_merge_overlapping(self) -> None:
        result = _merge_regions([(0, 15), (10, 20)])
        assert result == [(0, 20)]

    def test_merge_non_overlapping(self) -> None:
        result = _merge_regions([(0, 10), (20, 30)])
        assert result == [(0, 10), (20, 30)]

    def test_merge_unsorted(self) -> None:
        result = _merge_regions([(20, 30), (0, 10), (10, 20)])
        assert result == [(0, 30)]


class TestSplitterProtocol(unittest.TestCase):
    """Splitter 协议测试。"""

    def test_custom_splitter(self) -> None:
        def custom_splitter(text: str) -> list[tuple[int, int]]:
            return [(0, len(text) // 2), (len(text) // 2, len(text))]

        text = "Hello World"
        regions = custom_splitter(text)

        assert len(regions) == 2
        assert regions[0][1] == regions[1][0]


class TestEdgeCases(unittest.TestCase):
    """边界情况测试。"""

    def test_regex_splitter_multiline(self) -> None:
        splitter = RegexSplitter(pattern=r"^Chapter")
        text = "Chapter 1\nContent\nChapter 2\nMore content"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_recursive_splitter_very_small_max_size(self) -> None:
        splitter = RecursiveSplitter(max_size=1)
        text = "ABC"
        regions = splitter(text)

        assert len(regions) >= 1

    def test_window_splitter_stride_larger_than_size(self) -> None:
        splitter = WindowSplitter(size=5, stride=10)
        text = "0123456789"
        regions = splitter(text)

        assert len(regions) == 1
        assert regions[0] == (0, 5)

    def test_sentence_splitter_newlines(self) -> None:
        splitter = SentenceSplitter(lang="zh")
        text = "第一句\n第二句\n第三句"
        regions = splitter(text)

        assert len(regions) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
