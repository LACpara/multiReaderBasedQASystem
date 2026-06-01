"""
ReaderTreeBuilder 多级分割功能单元测试

测试 depth=0 → depth=1 → depth=2 等多级递归分割场景。
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


# 用于生成足够长文本的辅助函数
def make_long_chunk(suffix: str) -> str:
    """生成一个超过 max_leaf_chars (100) 的长文本 chunk"""
    return ("This is a long text chunk that definitely exceeds the threshold. " * 5) + suffix


def make_short_chunk(suffix: str) -> str:
    """生成一个短于 max_leaf_chars (100) 的短文本 chunk"""
    return ("A short text" + suffix)[:49]


@pytest.fixture
def mock_llm_service():
    """Mock LLM 服务"""
    mock = Mock(spec=ReaderLLMService)
    fake_knowledge = ReaderKnowledge(
        summary="test_summary",
        capability_questions=["Q1", "Q2"],
        source_excerpt="test"
    )
    mock.extract_knowledge.return_value = fake_knowledge
    mock.build_capability_questions.return_value = ["Capability Q1", "Capability Q2"]
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
def config():
    """测试配置"""
    return IngestionConfig(
        max_leaf_chars=100,
        max_depth=4,
        complexity_threshold=1000.0
    )


def create_complexity_that_splits():
    """创建高复杂度评分器（触发分割）"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.return_value = 2000.0  # > threshold
    return mock


class TestMultiLevelSplit:
    """多级分割功能测试"""

    # =========================================================================
    # TC-RTB-U-021: 2 级完整树
    # =========================================================================
    def test_two_level_full_tree(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-021: depth=0 分割为 3 节点，每个子节点再分割为 2

        测试场景：2 级完整分割树
        预期结果：总计 1 (根) + 3 (depth1) + 6 (depth2) = 10 个节点
        """
        # 配置 max_depth=2，限制只分割 2 层
        limited_config = IngestionConfig(
            max_leaf_chars=50,
            max_depth=2,
            complexity_threshold=500.0
        )

        # Mock splitter: 每次调用返回预设的 chunks
        # 通过 max_depth 来控制树的深度
        splitter = Mock(spec=SemanticTextSplitter)
        splitter.split.return_value = [
            make_long_chunk("_1"),
            make_long_chunk("_2"),
            make_long_chunk("_3")
        ]

        complexity = create_complexity_that_splits()

        builder = ReaderTreeBuilder(
            config=limited_config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        long_text = make_long_chunk("_root")

        root = builder.ingest_document(
            document_id="doc-2level",
            title="Two Level Test",
            text=long_text
        )

        assert isinstance(root, ReaderNode)

        # 验证树结构
        upsert_calls = mock_store.upsert_reader.call_args_list
        depth0_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 0]
        depth1_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 1]
        depth2_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 2]

        # 验证每层节点数：1 + 3 + 9 = 13 (因为每个 depth1 节点又分割为 3 个 depth2)
        # 但实际上每个 depth1 节点分割时会得到 3 个子节点，所以是 1 + 3 + 9 = 13
        assert len(depth0_nodes) == 1
        assert len(depth1_nodes) == 3
        # 由于 splitter 每次返回 3 个 chunks，depth1 每个节点会再分割
        assert len(depth2_nodes) == 9

        # 验证所有 depth2 节点是叶子（因为 max_depth=2）
        assert all(node.is_leaf for node in depth2_nodes)

        # 总节点数
        assert len(upsert_calls) == 13  # 1 + 3 + 9

    # =========================================================================
    # TC-RTB-U-022: 部分子节点分割（锯齿状树）
    # =========================================================================
    def test_partial_children_split(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-022: depth=0 分割为 3 节点，仅部分子节点继续分割

        测试场景：锯齿状树
        预期结果：不同分支有不同深度
        """
        # 配置 max_depth=1，第一层分割后不再分割
        limited_config = IngestionConfig(
            max_leaf_chars=50,
            max_depth=2,
            complexity_threshold=500.0
        )

        # Mock splitter: 每次调用返回 3 个 chunks
        splitter = Mock(spec=SemanticTextSplitter)
        splitter.split.side_effect = [
            [make_short_chunk("_1"),
             make_long_chunk("_2"),
             make_short_chunk("_3")],

            [make_long_chunk("_4"),
             make_long_chunk("_5")]
        ]

        complexity = Mock(spec=ComplexityEstimator)
        # 第一次是根节点的分割，然后顺位第一个和第三个子节点不分割，仅第二个分割
        complexity.score.side_effect = [1e10, 0, 1e10, 0]

        builder = ReaderTreeBuilder(
            config=limited_config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        long_text = make_long_chunk("_root")

        root = builder.ingest_document(
            document_id="doc-partial",
            title="Partial Split Test",
            text=long_text
        )

        assert isinstance(root, ReaderNode)
        assert splitter.split.call_count == 2

        # 验证树结构：1 根 + 3 depth1 = 4 个节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        depth0_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 0]
        depth1_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 1]
        depth2_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 2]

        assert len(depth0_nodes) == 1
        assert len(depth1_nodes) == 3
        assert len(depth2_nodes) == 2

        # 验证所有 depth1 节点是叶子（因为 max_depth=1）
        assert any(not node.is_leaf for node in depth1_nodes)
        assert sum(node.is_leaf for node in depth1_nodes) == 2

    # =========================================================================
    # TC-RTB-U-023: 深度边界测试
    # =========================================================================
    def test_depth_limit_enforcement(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-023: max_depth=2 时，depth=2 节点即使满足分割条件也不分割

        测试场景：深度达到限制时停止分割
        预期结果：在 max_depth 处停止，形成完整 3 层树
        """
        # 配置 max_depth=2
        limited_config = IngestionConfig(
            max_leaf_chars=50,
            max_depth=2,
            complexity_threshold=500.0
        )

        # Mock splitter: 每次调用返回 2 个 chunks
        splitter = Mock(spec=SemanticTextSplitter)
        splitter.split.return_value = [
            make_long_chunk("_1"),
            make_long_chunk("_2"),
        ]

        complexity = create_complexity_that_splits()

        builder = ReaderTreeBuilder(
            config=limited_config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        long_text = make_long_chunk("_root")

        root = builder.ingest_document(
            document_id="doc-depthlimit",
            title="Depth Limit Test",
            text=long_text
        )

        # 验证树结构：1 根 + 2 depth1 + 4 depth2 = 7
        upsert_calls = mock_store.upsert_reader.call_args_list
        depth0_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 0]
        depth1_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 1]
        depth2_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 2]

        # 验证深度不超过 max_depth
        assert max(c[0][0].depth for c in upsert_calls) <= 2

        # 验证每层节点数
        assert len(depth0_nodes) == 1
        assert len(depth1_nodes) == 2
        assert len(depth2_nodes) == 4  # 2 * 2 = 4

        # 验证总节点数
        assert len(upsert_calls) == 7  # 1 + 2 + 4

        # 验证 depth2 节点是叶子
        assert all(node.is_leaf for node in depth2_nodes)

    # =========================================================================
    # TC-RTB-U-025: max_depth=0
    # =========================================================================
    def test_max_depth_zero(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-025: max_depth=0 时，即使满足分割条件也不分割

        测试场景：禁用递归分割
        预期结果：只创建根节点
        """
        no_split_config = IngestionConfig(
            max_leaf_chars=50,
            max_depth=0,  # 禁用递归
            complexity_threshold=500.0
        )

        splitter = Mock(spec=SemanticTextSplitter)
        splitter.split.return_value = [make_long_chunk("_1"), make_long_chunk("_2")]

        complexity = create_complexity_that_splits()

        builder = ReaderTreeBuilder(
            config=no_split_config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        long_text = make_long_chunk("_root")

        root = builder.ingest_document(
            document_id="doc-nodepth",
            title="No Depth Test",
            text=long_text
        )

        # 验证只创建了根节点
        assert root.is_leaf is True
        assert root.child_ids == []
        assert mock_llm_service.extract_knowledge.call_count == 1
        assert mock_store.upsert_reader.call_count == 1

        # splitter 不应该被调用（因为 depth >= max_depth 优先判断）
        splitter.split.assert_not_called()

    # =========================================================================
    # 补充测试：单子节点分割
    # =========================================================================

    def test_single_child_split(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：分割只产生单个子节点
        """
        limited_config = IngestionConfig(
            max_leaf_chars=50,
            max_depth=4,
            complexity_threshold=500.0
        )

        # Mock splitter: 每次返回 1 个 chunk（不满足 >1 条件，不会创建子节点）
        splitter = Mock(spec=SemanticTextSplitter)
        splitter.split.return_value = [make_long_chunk("_1")]

        complexity = create_complexity_that_splits()

        builder = ReaderTreeBuilder(
            config=limited_config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        long_text = make_long_chunk("_root")

        root = builder.ingest_document(
            document_id="doc-singlechild",
            title="Single Child Test",
            text=long_text
        )

        # 验证只创建了根节点（因为 splitter 返回 1 个 chunk，len<=1 不创建子节点）
        assert root.is_leaf is True
        assert root.child_ids == []
        assert mock_llm_service.extract_knowledge.call_count == 1

    # =========================================================================
    # 补充测试：验证 ordinal 重置
    # =========================================================================

    def test_ordinal_resets_at_each_level(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证 ordinal 在每层深度重新从 0 开始计数
        """
        limited_config = IngestionConfig(
            max_leaf_chars=50,
            max_depth=1,
            complexity_threshold=500.0
        )

        splitter = Mock(spec=SemanticTextSplitter)
        splitter.split.return_value = [
            make_long_chunk("_1"),
            make_long_chunk("_2"),
            make_long_chunk("_3"),
        ]

        complexity = create_complexity_that_splits()

        builder = ReaderTreeBuilder(
            config=limited_config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        long_text = make_long_chunk("_root")

        builder.ingest_document(
            document_id="doc-ordinal",
            title="Ordinal Reset Test",
            text=long_text
        )

        # 验证 depth1 节点的 ordinal 从 0 开始
        upsert_calls = mock_store.upsert_reader.call_args_list
        depth1_nodes = [c[0][0] for c in upsert_calls if c[0][0].depth == 1]
        depth1_nodes.sort(key=lambda n: n.ordinal)

        assert len(depth1_nodes) == 3
        assert depth1_nodes[0].ordinal == 0
        assert depth1_nodes[1].ordinal == 1
        assert depth1_nodes[2].ordinal == 2
