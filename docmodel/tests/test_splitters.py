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
        splitter = RegexSplitter(pattern=r"^第[一二三四五六七八九十百]+章\n")
        text = "第一章\n内容一\n\n第二章\n内容二\n\n第三章\n内容三"
        regions = splitter(text)

        assert len(regions) == 3
        s1 = slice(*regions[0])
        s2 = slice(*regions[1])
        s3 = slice(*regions[2])

        assert text[s1].startswith("内容一")
        assert text[s2].startswith("内容二")
        assert text[s3].startswith("内容三")

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
        """小文本不切分。"""
        splitter = RecursiveSplitter(max_size=100)
        text = "Short text"
        regions = splitter(text)

        assert len(regions) == 1
        assert regions[0] == (0, 10)

    def test_split_by_separator(self) -> None:
        """按分隔符切分。"""
        splitter = RecursiveSplitter(separators=["\n\n", "\n"], max_size=20)
        text = "Paragraph 1\n\nParagraph 2\nParagraph 3"
        regions = splitter(text)

        assert len(regions) >= 1
        # 验证块大小不超过 max_size
        assert all(end - start <= splitter.max_size for start, end in regions)

    def test_recursive_degradation(self) -> None:
        """递归降级过程测试。"""
        # 设置 overlap=0 以验证内容完整性（无重复）
        splitter = RecursiveSplitter(separators=["\n\n", "\n", "。"], max_size=10, overlap=0)
        text = "这是第一段话\n\n这是第二段话\n这是第三句话。这是第四句话。这是第五句话。这是第六六六六六六六六句话。"
        regions = splitter(text)

        assert len(regions) > 1
        # 验证内容完整性（overlap=0 时无重复）
        total_text = "".join(text[slice(*r)] for r in regions)
        assert total_text == text
        assert all(end - start <= splitter.max_size for start, end in regions)

    def test_hard_split_fallback(self) -> None:
        """所有分隔符失效时硬切分测试。"""
        # 设置 overlap=0 以验证内容完整性（无重复）
        splitter = RecursiveSplitter(separators=["XXX", "YYY"], max_size=10, overlap=0)
        text = "This is a long text without any matching separators"
        regions = splitter(text)

        assert len(regions) > 1
        # 验证每个块大小不超过 max_size
        assert all(end - start <= 10 for start, end in regions)
        # 验证内容完整性（overlap=0 时无重复）
        total_text = "".join(text[slice(*r)] for r in regions)
        assert total_text == text

    def test_single_part_over_max_size(self) -> None:
        """单个部分大于 max_size 的场景。"""
        # 设置 overlap=0 以验证内容完整性（无重复）
        splitter = RecursiveSplitter(separators=["\n\n"], max_size=20, overlap=0)
        text = "This is a very long single paragraph that exceeds max size"
        regions = splitter(text)

        assert len(regions) > 1
        # 验证内容完整性（overlap=0 时无重复）
        total_text = "".join(text[slice(*r)] for r in regions)
        assert total_text == text

    def test_split_with_overlap_correctness(self) -> None:
        """overlap 参数应该产生真正的重叠。"""
        # 设计目标：
        # - max_size=10, overlap=3
        # - 相邻块之间应该有 3 个字符的重叠
        splitter = RecursiveSplitter(max_size=10, overlap=3)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        regions = splitter(text)

        assert len(regions) >= 2
        
        # 验证所有块大小不超过 max_size
        for start, end in regions:
            assert end - start <= splitter.max_size
        
        # 验证重叠效果：相邻块之间应该有 overlap 大小的重叠
        for i in range(1, len(regions)):
            prev_start, prev_end = regions[i - 1]
            curr_start, curr_end = regions[i]
            
            # 计算重叠大小
            overlap_size = prev_end - curr_start
            assert overlap_size == splitter.overlap, (
                f"块{i-1}和块{i}之间应该有 {splitter.overlap} 个字符重叠，"
                f"实际重叠 {overlap_size} 个字符"
            )

    def test_split_empty_text(self) -> None:
        """空文本处理。"""
        splitter = RecursiveSplitter()
        regions = splitter("")

        assert len(regions) == 0

    def test_split_chinese_text(self) -> None:
        """中文文本切分。"""
        # 设置 overlap=0 以验证内容完整性（无重复）
        splitter = RecursiveSplitter(separators=["\n\n", "\n", "。"], max_size=20, overlap=0)
        text = "这是第一句话。这是第二句话。这是第三句话。"
        regions = splitter(text)

        assert len(regions) >= 1
        # 验证内容完整性（overlap=0 时无重复）
        total_text = "".join(text[slice(*r)] for r in regions)
        assert total_text == text

    def test_split_no_valid_separator(self) -> None:
        """无有效分隔符时回退。"""
        splitter = RecursiveSplitter(separators=["XXX"], max_size=10)
        text = "This is a long text without the separator"
        regions = splitter(text)

        assert len(regions) >= 1
        # 验证每个块大小不超过 max_size
        assert all(end - start <= splitter.max_size for start, end in regions)

    def test_empty_chunk_filtering(self) -> None:
        """空块过滤测试。"""
        # 使用较小的 max_size 确保会被切分
        splitter = RecursiveSplitter(separators=["\n\n"], max_size=20)
        # 创建足够长的文本，中间有空段落
        text = "\n\n\n\n" + "A" * 15 + "\n\n\n\n" + "B" * 15 + "\n\n\n\n"
        regions = splitter(text)

        # 验证过滤了空块
        for start, end in regions:
            chunk = text[start:end]
            assert chunk.strip() != ""

    def test_split_preserves_content(self) -> None:
        """内容完整性验证。"""
        splitter = RecursiveSplitter(max_size=20)
        text = "Hello World"
        regions = splitter(text)

        total_text = "".join(text[slice(*r)] for r in regions)
        assert total_text == text

    def test_zero_overlap(self) -> None:
        """overlap=0 的场景。"""
        splitter = RecursiveSplitter(max_size=20, overlap=0)
        text = "This is a text that will be split without overlap"
        regions = splitter(text)

        assert len(regions) > 1
        # 验证无重叠
        for i in range(1, len(regions)):
            assert regions[i][0] == regions[i - 1][1]

    def test_large_overlap(self) -> None:
        """overlap 很大的场景（接近 max_size）。"""
        # 设计目标：当 overlap 接近 max_size 时，应该产生较大的重叠
        splitter = RecursiveSplitter(max_size=10, overlap=8)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        regions = splitter(text)

        assert len(regions) > 2
        
        # 验证所有块大小不超过 max_size
        for start, end in regions:
            assert end - start <= splitter.max_size
        
        # 验证每个相邻块之间的重叠都等于 overlap
        for i in range(1, len(regions)):
            prev_end = regions[i - 1][1]
            curr_start = regions[i][0]
            overlap_size = prev_end - curr_start
            assert overlap_size == splitter.overlap

    def test_separator_preservation(self) -> None:
        """分隔符保留验证。"""
        # 设置 overlap=0，避免 overlap 逻辑干扰测试
        splitter = RecursiveSplitter(separators=["\n\n"], max_size=20, keep_separator=True, overlap=0)
        part1 = "First paragraph with enough characters"
        part2 = "Second paragraph also has enough characters"
        text = part1 + "\n\n" + part2
        regions = splitter(text)

        # 验证分隔符被正确保留
        reconstructed = ""
        for start, end in regions:
            reconstructed += text[start:end]
        assert reconstructed == text
    
    def test_separator_drop(self) -> None:
        """分隔符丢弃"""
        # 设置 overlap=0，避免 overlap 逻辑干扰测试
        splitter = RecursiveSplitter(separators=["\n\n"], max_size=40, keep_separator=False, overlap=0)
        part1 = "First paragraph with enough characters"
        part2 = "Second paragraph also has enough characters"
        text = part1 + "\n\n" + part2
        regions = splitter(text)
        total_text = "".join([text[slice(*r)] for r in regions])
        assert total_text == part1 + part2

    def test_repr(self) -> None:
        """__repr__ 方法测试。"""
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
