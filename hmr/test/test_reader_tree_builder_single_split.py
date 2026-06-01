"""
ReaderTreeBuilder 单级分割功能单元测试

测试 depth=0 的根节点分割为 depth=1 子节点的场景。
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


def create_splitter_that_returns(chunks_list):
    """创建返回预设 chunks 的 Mock Splitter"""
    mock = Mock(spec=SemanticTextSplitter)
    mock.split.return_value = chunks_list
    return mock


def create_complexity_that_splits(split_nums = 0):
    """创建高复杂度评分器（触发分割）"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.side_effect = [1e10] + [0] * split_nums
    return mock


def create_complexity_that_not_splits():
    """创建低复杂度评分器（不触发分割）"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.return_value = 0  # < threshold
    return mock


class TestSingleLevelSplit:
    """单级分割功能测试（depth=0 → depth=1）"""

    # =========================================================================
    # TC-RTB-U-011: 2 个子节点
    # =========================================================================
    def test_single_level_split_2_children(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-011: Splitter 返回 2 个 chunks 时，根节点应有 2 个子节点

        测试场景：根节点分割为 2 个子节点
        预期结果：根节点 child_ids 长度为 2，两个子节点均为叶子
        """
        splitter = create_splitter_that_returns(["chunk-1", "chunk-2"])
        complexity = create_complexity_that_splits(2)

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "This is a long text that exceeds the max leaf chars threshold. " * 5

        root = builder.ingest_document(
            document_id="doc-2children",
            title="Two Children Test",
            text=long_text
        )

        # 验证根节点
        assert root.is_leaf is False
        assert len(root.child_ids) == 2

        # 验证 llm 服务调用次数
        assert mock_llm_service.extract_knowledge.call_count == 3  # 1 root + 2 children

    # =========================================================================
    # TC-RTB-U-012: 5 个子节点
    # =========================================================================
    def test_single_level_split_5_children(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-012: Splitter 返回 5 个 chunks 时，根节点应有 5 个子节点

        测试场景：根节点分割为 5 个子节点
        预期结果：根节点 child_ids 长度为 5，ordinal 从 0 到 4
        """
        chunks = ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
        splitter = create_splitter_that_returns(chunks)
        complexity = create_complexity_that_splits(len(chunks))

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "Content for chunk splitting that exceeds the threshold. " * 5

        root = builder.ingest_document(
            document_id="doc-5children",
            title="Five Children Test",
            text=long_text
        )

        # 验证根节点
        assert root.is_leaf is False
        assert len(root.child_ids) == 5

        # 验证 LLM 调用次数：1 root + 5 children = 6
        assert mock_llm_service.extract_knowledge.call_count == 6

        # 验证持久化调用次数：1 root + 5 children = 6
        assert mock_store.upsert_reader.call_count == 6
        assert mock_vector_index.upsert_reader.call_count == 6

    # =========================================================================
    # TC-RTB-U-013: 子节点标题格式
    # =========================================================================
    def test_child_node_title_format(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-013: 子节点标题格式应为 "{parent_title} / part-{index+1}"

        测试场景：根节点分割为 3 个子节点
        预期结果：子节点标题分别为 "/ part-1", "/ part-2", "/ part-3"
        """
        chunks = ["chunk-1", "chunk-2", "chunk-3"]
        splitter = create_splitter_that_returns(chunks)
        complexity = create_complexity_that_splits(len(chunks))

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "Long text for title format testing that exceeds the limit. " * 5

        root = builder.ingest_document(
            document_id="doc-title",
            title="Parent Title",
            text=long_text
        )

        # 获取所有 upsert_reader 调用，提取子节点标题
        upsert_calls = mock_store.upsert_reader.call_args_list
        child_titles = [
            call[0][0].title
            for call in upsert_calls
            if call[0][0].depth == 1
        ]

        # 验证子节点标题格式
        assert len(child_titles) == 3
        assert child_titles[0] == "Parent Title / part-1"
        assert child_titles[1] == "Parent Title / part-2"
        assert child_titles[2] == "Parent Title / part-3"

        # 验证根节点标题未变
        assert root.title == "Parent Title"

    # =========================================================================
    # TC-RTB-U-014: 单个 chunk 不分割
    # =========================================================================
    def test_no_split_when_single_chunk(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-014: Splitter 返回 1 个 chunk 时，不应创建子节点

        测试场景：Splitter 返回单 chunk，但复杂度满足分割条件
        预期结果：根节点仍为叶子节点，不创建子节点
        """
        splitter = create_splitter_that_returns(["single-chunk-that-is-long-enough"])
        complexity = Mock(spec=ComplexityEstimator)
        complexity.score.return_value = 1e10

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "This text is long enough but will be returned as a single chunk. " * 5

        root = builder.ingest_document(
            document_id="doc-single",
            title="Single Chunk Test",
            text=long_text
        )

        # 验证根节点是叶子
        assert root.is_leaf is True
        assert root.child_ids == []

        # 验证只创建了根节点
        assert mock_llm_service.extract_knowledge.call_count == 1
        assert mock_store.upsert_reader.call_count == 1

    # =========================================================================
    # TC-RTB-U-015: 空 chunks 不分割
    # =========================================================================
    def test_no_split_when_empty_chunks(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-015: Splitter 返回空列表时，不应创建子节点

        测试场景：Splitter 返回空列表
        预期结果：根节点仍为叶子节点，不创建子节点
        """
        splitter = create_splitter_that_returns([])
        complexity = create_complexity_that_splits()

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "Long text for empty chunks test that exceeds the limit. " * 5

        root = builder.ingest_document(
            document_id="doc-empty",
            title="Empty Chunks Test",
            text=long_text
        )

        # 验证根节点是叶子
        assert root.is_leaf is True
        assert root.child_ids == []

        # 验证只创建了根节点
        assert mock_llm_service.extract_knowledge.call_count == 1
        assert mock_store.upsert_reader.call_count == 1

    # =========================================================================
    # 补充测试：子节点深度和 ordinal 验证
    # =========================================================================

    def test_children_depth_and_ordinal(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证子节点的 depth 和 ordinal 属性
        """
        chunks = ["chunk-1", "chunk-2", "chunk-3"]
        splitter = create_splitter_that_returns(chunks)
        complexity = create_complexity_that_splits(len(chunks))

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "Long text for depth and ordinal testing. " * 5

        root = builder.ingest_document(
            document_id="doc-depth",
            title="Depth Test",
            text=long_text
        )

        # 获取所有 upsert_reader 调用
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 验证根节点
        root_node = next(n for n in all_nodes if n.depth == 0)
        assert root_node.ordinal == 0
        assert root_node.parent_id is None

        # 验证子节点
        child_nodes = [n for n in all_nodes if n.depth == 1]
        assert len(child_nodes) == 3

        for i, child in enumerate(child_nodes):
            assert child.depth == 1
            assert child.ordinal == i
            assert child.parent_id == root_node.reader_id
            assert child.text == chunks[i]

    def test_child_ids_reference_completeness(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证根节点的 child_ids 包含所有子节点的 reader_id
        """
        chunks = ["A", "B", "C"]
        splitter = create_splitter_that_returns(chunks)
        complexity = create_complexity_that_splits(len(chunks))

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "Long text for child reference completeness testing. " * 5

        root = builder.ingest_document(
            document_id="doc-childref",
            title="Child Ref Test",
            text=long_text
        )

        # 获取所有 upsert_reader 调用
        upsert_calls = mock_store.upsert_reader.call_args_list
        child_nodes = [call[0][0] for call in upsert_calls if call[0][0].depth == 1]

        # 验证 child_ids 包含所有子节点 ID
        assert len(root.child_ids) == 3
        for child in child_nodes:
            assert child.reader_id in root.child_ids

    def test_children_knowledge_extracted(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证每个子节点都调用了 LLM 进行知识提取
        """
        chunks = ["chunk-A", "chunk-B"]
        splitter = create_splitter_that_returns(chunks)
        complexity = create_complexity_that_splits(len(chunks))

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            complexity_estimator=complexity,
            document_spliter=splitter
        )

        # 使用超过 max_leaf_chars (100) 的长文本
        long_text = "Long text for LLM knowledge extraction testing. " * 5

        builder.ingest_document(
            document_id="doc-llm",
            title="LLM Test",
            text=long_text
        )

        # 验证 LLM 被调用了 3 次：根节点 + 2 个子节点
        assert mock_llm_service.extract_knowledge.call_count == 3
        assert mock_llm_service.build_capability_questions.call_count == 3

        # 验证子节点的文本正确传递给了 LLM
        extract_calls = mock_llm_service.extract_knowledge.call_args_list
        child_texts = [call[0][0] for call in extract_calls if call[1].get('title', '').startswith('LLM Test / part')]
        assert len(child_texts) == 2
        assert "chunk-A" in child_texts or "chunk-B" in child_texts
