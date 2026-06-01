"""
ReaderTreeBuilder 基本创建功能单元测试

测试 ingest_document() 方法的基本功能，包括单叶子节点创建、属性验证、持久化调用等。
"""

import pytest
import re
from unittest.mock import Mock, call

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
        entities=["Entity1", "Entity2"],
        relations=["relation1"],
        exceptions=["exception1"],
        capability_questions=["Q1", "Q2", "Q3"],
        source_excerpt="test source excerpt"
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
def mock_splitter():
    """Mock 文本分割器 - 返回空列表表示不分割"""
    mock = Mock(spec=SemanticTextSplitter)
    mock.split.return_value = []  # 不产生任何 chunks
    return mock


@pytest.fixture
def mock_complexity():
    """Mock 复杂度评估器 - 返回低分表示不分割"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.return_value = 500.0  # < threshold (1000.0)
    return mock


@pytest.fixture
def config():
    """测试配置"""
    return IngestionConfig(
        max_leaf_chars=100,
        max_depth=4,
        complexity_threshold=1000.0
    )


@pytest.fixture
def builder(config, mock_llm_service, mock_store, mock_vector_index, mock_splitter, mock_complexity):
    """创建 ReaderTreeBuilder 实例"""
    return ReaderTreeBuilder(
        config=config,
        llm_service=mock_llm_service,
        store=mock_store,
        vector_index=mock_vector_index,
        complexity_estimator=mock_complexity,
        document_spliter=mock_splitter
    )


class TestBasicIngestion:
    """基本摄入功能测试"""

    # =========================================================================
    # TC-RTB-U-001: 单叶子节点创建
    # =========================================================================
    def test_single_leaf_node_creation(self, builder):
        """
        TC-RTB-U-001: 不满足任何分割条件时，应创建单个叶子节点

        测试场景：
        - 文本简短 (长度 < max_leaf_chars)
        - 复杂性评分低 (< complexity_threshold)
        预期结果：创建单个 ReaderNode，无子节点
        """
        document_id = "doc-001"
        title = "Test Document"
        text = "This is a short test document."

        root = builder.ingest_document(
            document_id=document_id,
            title=title,
            text=text
        )

        # 验证返回的是 ReaderNode
        assert isinstance(root, ReaderNode)
        # 验证是叶子节点
        assert root.is_leaf is True
        assert root.child_ids == []
        # 验证根节点属性
        assert root.document_id == document_id
        assert root.title == title
        assert root.text == text
        assert root.depth == 0
        assert root.ordinal == 0
        assert root.parent_id is None

    # =========================================================================
    # TC-RTB-U-002: Reader 基本属性验证
    # =========================================================================
    def test_reader_node_attributes(self, builder):
        """
        TC-RTB-U-002: 验证创建的 ReaderNode 所有必填字段正确赋值

        测试场景：创建单个节点后，验证所有属性
        """
        document_id = "doc-002"
        title = "Attribute Test Document"
        text = "Testing all attributes of ReaderNode."

        root = builder.ingest_document(
            document_id=document_id,
            title=title,
            text=text
        )

        # 验证 reader_id 格式
        assert root.reader_id.startswith("reader::")
        assert document_id in root.reader_id
        assert "::0::0::" in root.reader_id  # depth=0, ordinal=0

        # 验证必填属性
        assert root.reader_id is not None
        assert root.document_id == document_id
        assert root.title == title
        assert root.parent_id is None
        assert root.depth == 0
        assert root.ordinal == 0
        assert root.text == text

        # 验证 knowledge 对象
        assert isinstance(root.knowledge, ReaderKnowledge)
        assert root.knowledge.summary == "test_summary"
        assert root.knowledge.capability_questions == ["Capability Q1", "Capability Q2"]

        # 验证时间戳格式
        assert root.created_at is not None
        iso_patern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.*\d*")
        assert re.match(iso_patern, root.created_at) is not None
        

    # =========================================================================
    # TC-RTB-U-003: LLM 调用验证
    # =========================================================================
    def test_llm_calls_per_node(self, builder, mock_llm_service):
        """
        TC-RTB-U-003: 每个节点应调用 LLM 服务的 extract_knowledge 和 build_capability_questions

        测试场景：创建单个节点
        预期结果：LLM 的两个方法各被调用 1 次
        """
        document_id = "doc-003"
        title = "LLM Call Test"
        text = "Testing LLM service calls."

        builder.ingest_document(
            document_id=document_id,
            title=title,
            text=text
        )

        # 验证 extract_knowledge 调用
        assert mock_llm_service.extract_knowledge.call_count == 1
        call_args, call_kargs = mock_llm_service.extract_knowledge.call_args
        # call_args: call(text, title=title) -> positional=(text,), kwargs={'title': title}
        assert call_args[0] == text
        assert call_kargs['title'] == title

        # 验证 build_capability_questions 调用
        assert mock_llm_service.build_capability_questions.call_count == 1
        call_args, call_kargs = mock_llm_service.build_capability_questions.call_args
        # 验证传入的是 extract_knowledge 返回的 knowledge 对象
        assert isinstance(call_args[0], ReaderKnowledge)
        assert call_kargs['title'] == title

    # =========================================================================
    # TC-RTB-U-004: 持久化调用验证
    # =========================================================================
    def test_persistence_calls(self, builder, mock_store, mock_vector_index):
        """
        TC-RTB-U-004: 每个节点应调用 store.upsert_reader 和 vector_index.upsert_reader

        测试场景：创建单个节点
        预期结果：两个持久化方法各被调用 1 次
        """
        document_id = "doc-004"
        title = "Persistence Test"
        text = "Testing persistence calls."

        builder.ingest_document(
            document_id=document_id,
            title=title,
            text=text
        )

        # 验证 store.upsert_reader 调用
        mock_store.upsert_reader.assert_called_once()
        store_call_args = mock_store.upsert_reader.call_args[0][0]
        assert isinstance(store_call_args, ReaderNode)

        # 验证 vector_index.upsert_reader 调用
        mock_vector_index.upsert_reader.assert_called_once()
        vector_call_args = mock_vector_index.upsert_reader.call_args[0][0]
        assert isinstance(vector_call_args, ReaderNode)

    # =========================================================================
    # TC-RTB-U-005: 文档删除验证
    # =========================================================================
    def test_document_deletion_before_build(self, builder, mock_store, mock_vector_index):
        """
        TC-RTB-U-005: 摄入同一文档时，应先删除旧数据再创建新数据

        测试场景：对同一 document_id 摄入两次
        预期结果：delete_document 方法在 upsert_reader 之前被调用
        """
        document_id = "doc-005"
        title = "Deletion Test"
        text = "Testing document deletion."

        # 第一次摄入
        builder.ingest_document(
            document_id=document_id,
            title=title,
            text=text
        )

        # 第二次摄入同一文档
        builder.ingest_document(
            document_id=document_id,
            title=title,
            text=text
        )

        # 单叶子节点情况下，每次摄入只产生 1 个节点（根节点）
        # 验证 delete_document 被调用了 2 次（每次摄入前都删除）
        assert mock_store.delete_document.call_count == 2
        assert mock_vector_index.delete_document.call_count == 2

        # 验证删除的 document_id 正确
        for call_obj in mock_store.delete_document.call_args_list:
            assert call_obj[0][0] == document_id

        # 验证 upsert_reader 被调用了 2 次（每次摄入 1 个节点）
        assert mock_store.upsert_reader.call_count == 2
        assert mock_vector_index.upsert_reader.call_count == 2

        # 验证调用顺序：delete → upsert → delete → upsert
        store_calls = mock_store.method_calls
        call_names = [c[0] for c in store_calls]
        assert call_names[0] == "delete_document"
        assert call_names[1] == "upsert_reader"
        assert call_names[2] == "delete_document"
        assert call_names[3] == "upsert_reader"

    # =========================================================================
    # 补充测试：默认值和边界情况
    # =========================================================================

    def test_default_complexity_estimator(self, config, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：不传入 complexity_estimator 时使用默认值
        """
        mock_splitter = Mock(spec=SemanticTextSplitter)
        mock_splitter.split.return_value = []

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            document_spliter=mock_splitter
            # 不传入 complexity_estimator
        )

        # 验证使用了默认的 ComplexityEstimator
        assert builder.complexity is not None
        assert isinstance(builder.complexity, ComplexityEstimator)

        root = builder.ingest_document(
            document_id="doc-default",
            title="Default Test",
            text="Test"
        )

        assert root.is_leaf is True

    def test_empty_text_document(self, builder, mock_llm_service):
        """
        补充测试：空文本文档的摄入
        """
        root = builder.ingest_document(
            document_id="doc-empty",
            title="Empty Text",
            text=""
        )

        assert root is None
        assert mock_llm_service.extract_knowledge.call_count == 0

    def test_special_characters_in_document(self, builder, mock_llm_service):
        """
        补充测试：包含特殊字符的文档
        """
        root = builder.ingest_document(
            document_id="doc-special",
            title="Special Chars: 🎉 中文 & Emoji",
            text="Hello 🎉 🌍 World! 中文测试 & 特殊符号 @#$%"
        )

        assert isinstance(root, ReaderNode)
        assert root.title == "Special Chars: 🎉 中文 & Emoji"
        assert "Hello" in root.text
        assert "🎉" in root.text
        assert "🌍" in root.text
        mock_llm_service.extract_knowledge.assert_called_once()

    def test_splitter_return_empty_text(self, config, mock_llm_service, mock_store, mock_vector_index):
        mock_splitter = Mock(spec=SemanticTextSplitter)

        long_text = "hello world" * 100
        empty_text = ""

        mock_splitter.split.side_effect = [
            [long_text, empty_text], Exception("Splitter should not be called over 2 times !")]

        mock_complexity = Mock(spec=ComplexityEstimator)
        mock_complexity.score.return_value = 1e10

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=mock_llm_service,
            store=mock_store,
            vector_index=mock_vector_index,
            document_spliter=mock_splitter,
            complexity_estimator=mock_complexity
        )

        root = builder.ingest_document(
            document_id="doc-id",
            title="test splitter return empty text",
            text=long_text
        )

        assert isinstance(root, ReaderNode)
        assert root.is_leaf
        assert root.child_ids == []

        assert mock_splitter.split.call_count == 1
        assert mock_complexity.score.call_count == 1
