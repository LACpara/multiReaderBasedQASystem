"""
ReaderTreeBuilder 分割决策逻辑单元测试

测试 _should_split() 方法的所有条件分支组合，确保分割决策逻辑正确。
"""

import pytest
from unittest.mock import Mock

from hmr.config import IngestionConfig
from hmr.reader_builder import ReaderTreeBuilder
from hmr.domain import ReaderNode, ReaderKnowledge
from hmr.llm.base import ReaderLLMService
from hmr.storage.base import KnowledgeStore
from hmr.vector.base import VectorIndex
from hmr.text_splitter import SemanticTextSplitter
from hmr.complexity import ComplexityEstimator


@pytest.fixture
def mock_llm_service():
    """Mock LLM 服务"""
    mock = Mock(spec=ReaderLLMService)
    fake_knowledge = ReaderKnowledge(
        summary="test_summary",
        source_excerpt="test source content"
    )
    mock.extract_knowledge.return_value = fake_knowledge
    mock.build_capability_questions.return_value = ["Q1", "Q2"]
    return mock


@pytest.fixture
def mock_store():
    """Mock 知识存储"""
    return Mock(spec=KnowledgeStore)


@pytest.fixture
def mock_vector_index():
    """Mock 向量索引"""
    return Mock(spec=VectorIndex)


@pytest.fixture
def default_config():
    """默认配置"""
    return IngestionConfig(
        max_leaf_chars=100,
        max_depth=4,
        complexity_threshold=1000.0
    )


@pytest.fixture
def builder(default_config, mock_llm_service, mock_store, mock_vector_index):
    """创建 ReaderTreeBuilder 实例"""
    return ReaderTreeBuilder(
        config=default_config,
        llm_service=mock_llm_service,
        store=mock_store,
        vector_index=mock_vector_index,
        complexity_estimator=Mock(spec=ComplexityEstimator),
        document_spliter=Mock(spec=SemanticTextSplitter)
    )


class TestShouldSplitLogic:
    """_should_split() 方法的分割决策逻辑测试"""

    # =========================================================================
    # TC-RTB-U-031: 深度超限测试
    # =========================================================================
    def test_should_split_depth_exceeded(self, builder):
        """
        TC-RTB-U-031: 当 depth >= max_depth 时，_should_split() 应返回 False

        测试场景：depth 等于 max_depth，不应进行分割
        """
        text = "This is a complex text that should be split" * 10
        depth = 4  # 等于 max_depth

        result = builder._should_split(text, depth)

        assert result is False, f"depth={depth} >= max_depth=4 时不应分割"

    # =========================================================================
    # TC-RTB-U-032: 复杂性不足测试
    # =========================================================================
    def test_should_split_low_complexity(self, builder, default_config):
        """
        TC-RTB-U-032: 当 score < complexity_threshold 时，_should_split() 应返回 False

        测试场景：文本长度超过阈值但复杂性评分不足
        """
        text = "simple text " * 50  # 长度超过 max_leaf_chars
        depth = 0

        # Mock 复杂度评分器返回低于阈值的结果
        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 500.0  # < threshold (1000.0)

        result = builder._should_split(text, depth)

        assert result is False, "复杂性评分低于阈值时不应分割"
        builder.complexity.score.assert_called_once_with(text)

    # =========================================================================
    # TC-RTB-U-033: 长度不足测试
    # =========================================================================
    def test_should_split_short_text(self, builder, default_config):
        """
        TC-RTB-U-033: 当 len(text) <= max_leaf_chars 时，_should_split() 应返回 False

        测试场景：复杂性评分足够但文本长度不足
        """
        text = "short text"  # 长度小于 max_leaf_chars (100)
        depth = 0

        # Mock 复杂度评分器返回足够高的评分
        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 2000.0  # > threshold

        result = builder._should_split(text, depth)

        assert result is False, "文本长度不足时不应分割"
        builder.complexity.score.assert_called_once_with(text)

    # =========================================================================
    # TC-RTB-U-034: 复杂性达标但长度不足测试
    # =========================================================================
    def test_should_split_complex_ok_length_not(self, builder, default_config):
        """
        TC-RTB-U-034: 当 score >= threshold 但 len(text) <= max_leaf 时，_should_split() 应返回 False

        测试场景：复杂性达标但长度不足
        注意：max_leaf_chars=100，"complex" 只有 7 个字符
        """
        text = "complex"  # 长度 7 < max_leaf_chars (100)
        depth = 0

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 2000.0  # > threshold

        result = builder._should_split(text, depth)

        assert result is False, "复杂性达标但长度不足时不应分割"

    # =========================================================================
    # TC-RTB-U-035: 长度达标但复杂性不足测试
    # =========================================================================
    def test_should_split_length_ok_complex_not(self, builder, default_config):
        """
        TC-RTB-U-035: 当 len(text) > max_leaf 但 score < threshold 时，_should_split() 应返回 False

        测试场景：长度足够但复杂性不足
        """
        text = "simple repetitive simple repetitive simple " * 10  # 长但简单
        depth = 0

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 300.0  # < threshold

        result = builder._should_split(text, depth)

        assert result is False, "长度达标但复杂性不足时不应分割"

    # =========================================================================
    # TC-RTB-U-036: 所有条件满足测试
    # =========================================================================
    def test_should_split_all_conditions_met(self, builder, default_config):
        """
        TC-RTB-U-036: 当 score >= threshold AND len > max_leaf AND depth < max_depth 时，_should_split() 应返回 True

        测试场景：所有分割条件都满足
        """
        text = "complex technical concept with multiple layers and components " * 10
        depth = 0

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 1500.0  # > threshold (1000.0)

        result = builder._should_split(text, depth)

        assert result is True, "所有条件满足时应进行分割"
        builder.complexity.score.assert_called_once_with(text)

    # =========================================================================
    # TC-RTB-U-037: 空文本测试
    # =========================================================================
    def test_should_split_empty_text(self, builder, default_config):
        """
        TC-RTB-U-037: 当 text 为空字符串时，_should_split() 应返回 False

        测试场景：空文本不应进行分割
        """
        text = ""
        depth = 0

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 0.0  # 空文本复杂度为 0

        result = builder._should_split(text, depth)

        assert result is False, "空文本不应进行分割"

    # =========================================================================
    # 边界条件补充测试
    # =========================================================================

    def test_should_split_depth_one_less_than_max(self, builder, default_config):
        """
        补充测试：depth = max_depth - 1 时应该可以分割
        """
        text = "complex text " * 20
        depth = default_config.max_depth - 1  # 3

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 1500.0  # > threshold

        result = builder._should_split(text, depth)

        # depth=3 < max_depth=4，应该可以分割
        assert result is True, "depth=3 < max_depth=4 时应该可以分割"

    def test_should_split_at_threshold_boundary(self, builder, default_config):
        """
        补充测试：刚好达到复杂性阈值边界的情况
        """
        text = "complex text " * 20
        depth = 0

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 1000.0  # == threshold

        result = builder._should_split(text, depth)

        # score >= threshold，应该分割
        assert result is True, "刚好达到阈值时应分割"

    def test_should_split_at_length_boundary(self, builder, default_config):
        """
        补充测试：刚好达到长度阈值边界的情况
        """
        text = "x" * 101  # 刚好 > max_leaf_chars (100)
        depth = 0

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 2000.0  # > threshold

        result = builder._should_split(text, depth)

        assert result is True, "刚好超过长度阈值时应分割"

    def test_should_split_depth_negative_not_allowed(self, builder, default_config):
        """
        补充测试：depth 为负数时的行为（理论上不应该发生）
        """
        text = "any text" * 100
        depth = -1

        builder.complexity = Mock(spec=ComplexityEstimator)
        builder.complexity.score.return_value = 2000.0

        # depth < 0 会被 _should_split 中 depth >= max_depth 判断拦截
        result = builder._should_split(text, depth)

        assert result is False, "depth < 0 时不应分割（被 max_depth 检查拦截）"
