"""
ReaderTreeBuilder 端到端功能测试

使用真实数据库和组件进行完整流程测试。
遵循测试设计文档：test_reader_tree_builder_spec_revised.md
"""

from venv import logger

import pytest
from unittest.mock import Mock, MagicMock
from typing import Callable, List

from hmr.config import IngestionConfig
from hmr.reader_builder import ReaderTreeBuilder
from hmr.domain import ReaderNode, ReaderKnowledge
from hmr.llm.base import ReaderLLMService
from hmr.llm.heuristic_service import HeuristicReaderLLMService
from hmr.storage.sqlite_store import SQLiteKnowledgeStore
from hmr.vector.chroma_index import ChromaVectorIndex
from hmr.vector.base import VectorIndex
from hmr.text_splitter import SemanticTextSplitter
from hmr.complexity import ComplexityEstimator


# =============================================================================
# 测试数据样本（按照设计文档要求）
# =============================================================================
E2E_TEST_DOCUMENTS = {
    "E2E-D1": {
        "name": "简单技术文档",
        "text": "这是一个简单的技术文档，包含基本的概念说明。内容较短，复杂度较低。文档主要用于测试单节点创建场景，不会触发分割。",
    },
    "E2E-D2": {
        "name": "中等复杂度文档",
        "text": "这是一个中等复杂度的技术文档。它包含了多个技术概念，如数据库系统、向量索引、检索机制、智能体策略等。这些概念之间有一定的关联性，但整体结构相对清晰。文档长度适中，适合进行单级分割测试。文档还包含一些关系描述，如系统采用分布式架构，通过API接口进行数据交互，基于向量数据库实现快速检索。",
    },
    "E2E-D3": {
        "name": "高复杂度文档",
        "text": "这是一个高复杂度的技术文档，包含多个段落和复杂概念。第一段介绍了系统的整体架构设计，包括微服务架构、消息队列、分布式缓存等核心组件。第二段详细描述了数据处理流程，包括数据采集、清洗、转换、存储等多个环节。第三段讨论了系统的性能优化策略，包括索引优化、查询优化、缓存策略等。第四段介绍了系统的安全性设计，包括身份认证、权限管理、数据加密等方面。第五段讨论了系统的可扩展性设计，包括水平扩展、垂直扩展、弹性伸缩等策略。整个文档内容丰富，涉及多个技术领域，适合进行多级分割测试。",
    },
    "E2E-D4": {
        "name": "多段落文档",
        "text": "第一段内容：这是第一个段落，介绍了文档的基本背景和目的。\n\n第二段内容：这是第二个段落，详细描述了第一个核心概念及其相关特性。\n\n第三段内容：这是第三个段落，介绍了第二个核心概念及其应用场景。\n\n第四段内容：这是第四个段落，讨论了两个概念之间的关系和交互方式。\n\n第五段内容：这是第五个段落，总结了文档的主要内容和核心要点。",
    },
    "E2E-D5": {
        "name": "超长文档",
        "text": "段落内容重复" * 50,
    },
}


# =============================================================================
# 辅助验证函数
# =============================================================================
def verify_tree_structure(root_node: ReaderNode, all_nodes: List[ReaderNode]) -> None:
    """验证树结构的完整性和一致性"""
    node_by_id = {node.reader_id: node for node in all_nodes}
    
    # 验证根节点没有父节点
    assert root_node.parent_id is None, "根节点不应有父节点"
    assert root_node.depth == 0, "根节点深度应为0"
    assert root_node.ordinal == 0, "根节点序号应为0"
    
    # 验证每个子节点的 parent_id 指向正确的父节点
    for node in all_nodes:
        if node.parent_id is not None:
            assert node.parent_id in node_by_id, f"父节点 {node.parent_id} 不存在"
            parent_node = node_by_id[node.parent_id]
            assert node.depth == parent_node.depth + 1, \
                f"节点 {node.reader_id} 的深度不正确"
            assert node.reader_id in parent_node.child_ids, \
                f"父节点 {parent_node.reader_id} 的 child_ids 不包含子节点 {node.reader_id}"
    
    # 验证每个父节点的 child_ids 包含所有子节点
    for node in all_nodes:
        for child_id in node.child_ids:
            assert child_id in node_by_id, f"子节点 {child_id} 不存在"
            child_node = node_by_id[child_id]
            assert child_node.parent_id == node.reader_id, \
                f"子节点 {child_id} 的 parent_id 不正确"
    
    # 验证 ordinal 顺序正确（同一父节点的直接子节点之间连续从0开始）
    nodes_by_parent = {}
    for node in all_nodes:
        if node.parent_id not in nodes_by_parent:
            nodes_by_parent[node.parent_id] = []
        nodes_by_parent[node.parent_id].append(node)
    
    for parent_id, nodes in nodes_by_parent.items():
        ordinals = sorted([node.ordinal for node in nodes])
        expected_ordinals = list(range(len(nodes)))
        assert ordinals == expected_ordinals, \
            f"父节点 {parent_id} 的子节点 ordinal 不连续或不正确: {ordinals}"
    
    # 验证 is_leaf 属性正确
    for node in all_nodes:
        expected_is_leaf = len(node.child_ids) == 0
        assert node.is_leaf == expected_is_leaf, \
            f"节点 {node.reader_id} 的 is_leaf 属性不正确"


def count_nodes_by_depth(nodes: List[ReaderNode]) -> dict[int, int]:
    """统计每层节点数量"""
    counts = {}
    for node in nodes:
        counts[node.depth] = counts.get(node.depth, 0) + 1
    return counts


def verify_tree_connectivity(root_node: ReaderNode, all_nodes: List[ReaderNode]) -> None:
    """验证每个节点都可以从根节点到达"""
    node_by_id = {node.reader_id: node for node in all_nodes}
    
    def traverse(node: ReaderNode, visited: set) -> None:
        visited.add(node.reader_id)
        for child_id in node.child_ids:
            if child_id not in visited:
                traverse(node_by_id[child_id], visited)
    
    visited = set()
    traverse(root_node, visited)
    
    assert len(visited) == len(all_nodes), \
        f"树结构不完整，可访问节点数 {len(visited)} != 总节点数 {len(all_nodes)}"


def create_mock_splitter(chunk_map: dict[str, List[str]]) -> Mock:
    """创建返回预设 chunks 的 Mock Splitter，根据文本内容返回"""
    mock = Mock(spec=SemanticTextSplitter)
    mock.split.side_effect = lambda text: chunk_map.get(text, [])
    return mock


def create_mock_complexity(score_map: dict[str, float]) -> Mock:
    """创建返回预设复杂度评分的 Mock Estimator"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.side_effect = lambda text: score_map.get(text, 100.0)
    return mock


# =============================================================================
# 端到端测试类
# =============================================================================
class TestReaderTreeBuilderEndToEnd:
    """ReaderTreeBuilder 端到端集成测试（内存数据库）"""

    @pytest.fixture
    def memory_knowledge_store(self):
        """返回内存 SQLite 存储"""
        store = SQLiteKnowledgeStore()  # 不传路径，使用内存模式
        store.init_schema()
        yield store
        store.close()

    @pytest.fixture
    def memory_vector_index(self):
        """返回 Mock 的 VectorIndex（模拟内存模式）"""
        try:
            import chromadb
            vector_index = ChromaVectorIndex(collection_name="default-collection-name")
        except ImportError:
            logger.warning("dependency `chromdb` is not found. use mock vector index component instead.")
            mock_index = Mock(spec=VectorIndex)
            # 存储 upsert 的节点
            indexed_nodes = {}
            
            def upsert_reader(node):
                indexed_nodes[node.reader_id] = node
            
            def delete_document(document_id):
                indexed_nodes.clear()
            
            def query(question, *, top_k, where):
                document_id = where.get("document_id")
                results = []
                for reader_id, node in indexed_nodes.items():
                    if document_id is None or node.document_id == document_id:
                        from hmr.domain import VectorCandidate
                        results.append(VectorCandidate(
                            reader_id=reader_id,
                            score=1.0,
                            document="test",
                            metadata={"document_id": node.document_id}
                        ))
                return results[:top_k]
            
            mock_index.upsert_reader = upsert_reader
            mock_index.delete_document = delete_document
            mock_index.query = query
            mock_index.close = Mock()
            vector_index = mock_index
        
        return vector_index

    @pytest.fixture
    def heuristic_llm_service(self):
        """返回真实的 HeuristicReaderLLMService"""
        return HeuristicReaderLLMService()

    @pytest.fixture
    def config(self):
        """返回测试配置"""
        return IngestionConfig(
            max_leaf_chars=100,
            max_depth=4,
            complexity_threshold=1000.0
        )

    @pytest.fixture
    def default_builder(
        self,
        config,
        heuristic_llm_service,
        memory_knowledge_store,
        memory_vector_index
    ):
        """创建默认配置的 builder（使用真实组件）"""
        return ReaderTreeBuilder(
            config=config,
            llm_service=heuristic_llm_service,
            store=memory_knowledge_store,
            vector_index=memory_vector_index,
            complexity_estimator=ComplexityEstimator(),
            document_spliter=SemanticTextSplitter(max_chars=config.max_leaf_chars)
        )

    # =========================================================================
    # TC-RTB-E-001: 简单文档端到端
    # =========================================================================
    def test_simple_document_e2e(
        self,
        default_builder,
        memory_knowledge_store,
        memory_vector_index
    ):
        """
        TC-RTB-E-001: 简单文档端到端测试

        测试场景：无分割，单节点
        预期结果：节点在 SQLite 和 Chroma 中正确保存，可查询
        """
        text = E2E_TEST_DOCUMENTS["E2E-D1"]["text"]

        root = default_builder.ingest_document(
            document_id="doc-e2e-1",
            title="简单文档测试",
            text=text
        )

        # 验证根节点存在
        assert root is not None
        assert root.is_leaf, "简单文档不应被分割"

        # 验证能从存储中查询到节点
        stored_node = memory_knowledge_store.get_reader(root.reader_id)
        assert stored_node is not None
        assert stored_node.document_id == "doc-e2e-1"
        assert stored_node.title == "简单文档测试"

        # 验证所有文档节点都被存储
        all_nodes = memory_knowledge_store.list_document_readers(document_id="doc-e2e-1")
        assert len(all_nodes) == 1

        # 验证向量索引中能查询到节点
        result = memory_vector_index.query(
            "测试查询",
            top_k=10,
            where={"document_id": "doc-e2e-1"}
        )
        assert len(result) == 1
        assert result[0].reader_id == root.reader_id

    # =========================================================================
    # TC-RTB-E-002: 多级分割端到端
    # =========================================================================
    def test_multi_level_split_e2e(
        self,
        config,
        heuristic_llm_service,
        memory_knowledge_store,
        memory_vector_index
    ):
        """
        TC-RTB-E-002: 多级分割端到端测试

        测试场景：2级分割，复杂树结构，同一深度节点来自不同父节点
        预期结果：树结构完整性，ordinal 在不同父节点之间可以重叠但在同一父节点内必须连续
        预期树结构：
            Root (depth=0)
            ├── A (depth=1, ordinal=0)
            │   ├── A1 (depth=2, ordinal=0)
            │   └── A2 (depth=2, ordinal=1)
            ├── B (depth=1, ordinal=1)
            │   ├── B1 (depth=2, ordinal=0)
            │   └── B2 (depth=2, ordinal=1)
            └── C (depth=1, ordinal=2)  # 不分割
        """
        # 足够长的文本以满足长度条件
        long_text = "这是一个需要多级分割的长文档。" * 50
        
        # Mock splitter 控制分割行为 - 创建复杂树结构
        mock_splitter = create_mock_splitter({
            long_text: ["chunk-A" * 20, "chunk-B" * 20, "chunk-C" * 20],  # depth=0 分割为3个
            "chunk-A" * 20: ["chunk-A1" * 20, "chunk-A2" * 20],            # A 分割为2个
            "chunk-B" * 20: ["chunk-B1" * 20, "chunk-B2" * 20],            # B 分割为2个
            "chunk-C" * 20: [],                                             # C 不分割
            "chunk-A1" * 20: [],                                            # A1 不分割
            "chunk-A2" * 20: [],                                            # A2 不分割
            "chunk-B1" * 20: [],                                            # B1 不分割
            "chunk-B2" * 20: [],                                            # B2 不分割
        })

        # Mock complexity estimator 返回高评分，确保触发分割
        mock_complexity = create_mock_complexity({
            long_text: 2000.0,
            "chunk-A" * 20: 2000.0,   # 分割
            "chunk-B" * 20: 2000.0,   # 分割
            "chunk-C" * 20: 500.0,    # 不分割
            "chunk-A1" * 20: 500.0,   # 不分割
            "chunk-A2" * 20: 500.0,   # 不分割
            "chunk-B1" * 20: 500.0,   # 不分割
            "chunk-B2" * 20: 500.0,   # 不分割
        })

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=heuristic_llm_service,
            store=memory_knowledge_store,
            vector_index=memory_vector_index,
            complexity_estimator=mock_complexity,
            document_spliter=mock_splitter
        )

        root = builder.ingest_document(
            document_id="doc-e2e-2",
            title="多级分割测试",
            text=long_text
        )

        # 获取所有节点
        all_nodes = memory_knowledge_store.list_document_readers(document_id="doc-e2e-2")

        # 预期：1(根) + 3(depth1) + 4(depth2) = 8 个节点
        # 根节点分割为3个子节点(A,B,C)，其中A和B各分割为2个孙子节点
        assert len(all_nodes) == 8, f"预期8个节点，实际{len(all_nodes)}个"

        # 验证树结构
        verify_tree_structure(root, all_nodes)
        verify_tree_connectivity(root, all_nodes)

        # 验证层级分布
        counts = count_nodes_by_depth(all_nodes)
        assert counts == {0: 1, 1: 3, 2: 4}, f"层级分布不正确: {counts}"

        # 验证同一深度的节点来自不同父节点（ordinal 会重叠）
        nodes_by_depth = {}
        for node in all_nodes:
            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = []
            nodes_by_depth[node.depth].append(node)
        
        # depth=2 有4个节点，来自2个不同父节点，ordinal 应该是 [0,1,0,1]
        depth2_ordinals = sorted([node.ordinal for node in nodes_by_depth[2]])
        assert depth2_ordinals == [0, 0, 1, 1], \
            f"depth=2 的 ordinal 应该重叠: {depth2_ordinals}"

    # =========================================================================
    # TC-RTB-E-003: 文档覆盖写入
    # =========================================================================
    def test_document_overwrite_e2e(
        self,
        default_builder,
        memory_knowledge_store,
        memory_vector_index
    ):
        """
        TC-RTB-E-003: 文档覆盖写入测试

        测试场景：同一 document_id 摄入两次
        预期结果：第一次的节点被删除，第二次的节点正确保存
        """
        text1 = E2E_TEST_DOCUMENTS["E2E-D1"]["text"]
        text2 = E2E_TEST_DOCUMENTS["E2E-D2"]["text"]

        # 第一次摄入
        root1 = default_builder.ingest_document(
            document_id="doc-e2e-3",
            title="第一次写入",
            text=text1
        )
        reader_id_1 = root1.reader_id

        # 第二次摄入同一 document_id
        root2 = default_builder.ingest_document(
            document_id="doc-e2e-3",
            title="第二次写入",
            text=text2
        )
        reader_id_2 = root2.reader_id

        # 验证两次的 reader_id 不同
        assert reader_id_1 != reader_id_2

        # 验证第一次的节点被删除
        stored_1 = memory_knowledge_store.get_reader(reader_id_1)
        assert stored_1 is None, "第一次的节点应该被删除"

        # 验证第二次的节点正确保存
        stored_2 = memory_knowledge_store.get_reader(reader_id_2)
        assert stored_2 is not None
        assert stored_2.title == "第二次写入"

        # 验证向量索引中只有第二次的节点
        result = memory_vector_index.query(
            "测试查询",
            top_k=10,
            where={"document_id": "doc-e2e-3"}
        )
        reader_ids = [r.reader_id for r in result]
        assert reader_id_1 not in reader_ids
        assert reader_id_2 in reader_ids

    # =========================================================================
    # TC-RTB-E-004: 树遍历验证
    # =========================================================================
    def test_tree_traversal_e2e(
        self,
        config,
        heuristic_llm_service,
        memory_knowledge_store,
        memory_vector_index
    ):
        """
        TC-RTB-E-004: 树遍历验证测试

        测试场景：构建树后通过 API 遍历
        预期结果：list_children 返回正确的子节点顺序
        """
        # 足够长的文本以满足长度条件
        long_text = "这是一个需要分割的文档内容，长度超过最大叶子节点字符限制。" * 20

        # Mock splitter 返回5个 chunks
        mock_splitter = create_mock_splitter({
            long_text: ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        })

        # Mock complexity estimator 返回高评分，确保触发分割
        mock_complexity = create_mock_complexity({
            long_text: 2000.0,
        })

        builder = ReaderTreeBuilder(
            config=config,
            llm_service=heuristic_llm_service,
            store=memory_knowledge_store,
            vector_index=memory_vector_index,
            complexity_estimator=mock_complexity,
            document_spliter=mock_splitter
        )

        root = builder.ingest_document(
            document_id="doc-e2e-4",
            title="树遍历测试",
            text=long_text
        )

        # 通过 list_children 获取子节点
        children = memory_knowledge_store.list_children(root.reader_id)

        # 验证子节点数量
        assert len(children) == 5

        # 验证子节点顺序（ordinal 从0开始递增）
        for i, child in enumerate(children):
            assert child.ordinal == i, f"子节点{i}的序号不正确"
            assert child.parent_id == root.reader_id
            assert child.depth == 1
            assert f"树遍历测试 / part-{i+1}" == child.title

    # =========================================================================
    # TC-RTB-E-005: 知识提取验证
    # =========================================================================
    def test_knowledge_extraction_e2e(
        self,
        default_builder,
        memory_knowledge_store
    ):
        """
        TC-RTB-E-005: 知识提取验证测试

        测试场景：使用 HeuristicService 提取知识
        预期结果：ReaderKnowledge 包含预期的 summary、entities 等
        """
        text = E2E_TEST_DOCUMENTS["E2E-D2"]["text"]

        root = default_builder.ingest_document(
            document_id="doc-e2e-5",
            title="知识提取测试",
            text=text
        )

        # 获取存储中的节点
        stored_node = memory_knowledge_store.get_reader(root.reader_id)

        # 验证知识被正确提取和存储
        assert stored_node is not None
        assert stored_node.knowledge is not None
        assert stored_node.knowledge.summary != "", "summary 不应为空"
        assert len(stored_node.knowledge.entities) > 0, "应有实体提取"
        assert len(stored_node.knowledge.capability_questions) > 0, "应有能力问题"

        # 验证能力问题数量符合预期（最多8个）
        assert len(stored_node.knowledge.capability_questions) <= 8

    # =========================================================================
    # TC-RTB-E-006: 真实复杂性评估
    # =========================================================================
    def test_real_complexity_estimator_e2e(
        self,
        config,
        heuristic_llm_service,
        memory_knowledge_store,
        memory_vector_index
    ):
        """
        TC-RTB-E-006: 真实复杂性评估测试

        测试场景：使用真实 ComplexityEstimator
        预期结果：验证分割决策与复杂度评分一致
        """
        # 使用包含多种技术概念的复杂文本，确保复杂度评分超过阈值
        # 通过增加更多不同的技术术语来提高复杂度评分
        text = ("微服务架构是一种分布式系统设计模式，它将应用程序分解为小型独立服务。" +
                "Redis是一种高性能的分布式缓存系统，支持多种数据结构如字符串、哈希、列表和集合。" +
                "Kafka是一种高吞吐量的消息队列系统，采用发布订阅模式处理实时数据流。" +
                "Chroma是一种专为AI应用设计的向量数据库，支持高效的相似度搜索。" +
                "Pinecone提供托管的向量索引服务，支持大规模向量数据的存储和查询。" +
                "机器学习涉及监督学习、无监督学习和强化学习三种主要范式。" +
                "深度学习使用多层神经网络进行特征提取和模式识别。" +
                "自然语言处理涉及文本分类、情感分析、命名实体识别和机器翻译等任务。" +
                "REST API是一种轻量级的通信协议，基于HTTP协议实现客户端服务器交互。" +
                "Docker提供容器化部署方案，简化应用程序的打包和部署流程。" +
                "Kubernetes用于容器编排管理，实现自动化部署、扩展和管理容器化应用。" +
                "CI/CD实现持续集成和持续部署，加速软件开发和交付流程。") * 2
        
        complexity_estimator = ComplexityEstimator()
        
        # 预先计算复杂度评分
        score = complexity_estimator.score(text)
        assert score >= config.complexity_threshold, \
            f"文档复杂度评分 {score} 应 >= 阈值 {config.complexity_threshold}"
        
        # 确保文本长度超过 max_leaf_chars
        assert len(text) > config.max_leaf_chars, \
            f"文本长度 {len(text)} 应 > max_leaf_chars {config.max_leaf_chars}"
        
        # 使用 Mock Splitter 确保返回多个 chunks（SemanticTextSplitter 不会拆分长句子）
        mock_splitter = Mock(spec=SemanticTextSplitter)
        mock_splitter.split.return_value = ["chunk1 content", "chunk2 content", "chunk3 content"]
        
        builder = ReaderTreeBuilder(
            config=config,
            llm_service=heuristic_llm_service,
            store=memory_knowledge_store,
            vector_index=memory_vector_index,
            complexity_estimator=complexity_estimator,
            document_spliter=mock_splitter
        )

        root = builder.ingest_document(
            document_id="doc-e2e-6",
            title="真实复杂性评估测试",
            text=text
        )

        # 验证文档被分割（因为复杂度达标）
        assert root is not None
        assert not root.is_leaf, "高复杂度文档应该被分割"
        assert len(root.child_ids) > 0, "应有子节点"

    # =========================================================================
    # TC-RTB-E-007: 向量索引验证
    # =========================================================================
    def test_vector_index_e2e(
        self,
        config,
        heuristic_llm_service,
        memory_knowledge_store,
        memory_vector_index
    ):
        """
        TC-RTB-E-007: 向量索引验证测试

        测试场景：构建完成后查询 VectorIndex
        预期结果：可通过 document_id 查询到所有节点
        """
        # 使用复杂文本确保分割发生
        text = ("微服务架构是一种分布式系统设计模式。Redis是一种高性能的分布式缓存系统。" +
                "Kafka是一种高吞吐量的消息队列系统。Chroma是一种向量数据库。" +
                "Pinecone提供向量索引服务。机器学习涉及监督学习和无监督学习。" +
                "深度学习使用神经网络进行特征提取。自然语言处理涉及文本分类和情感分析。") * 3
        
        # 使用 Mock 的复杂度评估器确保分割发生
        mock_complexity = Mock(spec=ComplexityEstimator)
        mock_complexity.score.return_value = 2000.0  # 高评分确保分割
        
        builder = ReaderTreeBuilder(
            config=config,
            llm_service=heuristic_llm_service,
            store=memory_knowledge_store,
            vector_index=memory_vector_index,
            complexity_estimator=mock_complexity,
            document_spliter=SemanticTextSplitter(max_chars=50)
        )

        root = builder.ingest_document(
            document_id="doc-e2e-7",
            title="向量索引验证测试",
            text=text
        )

        # 获取存储中的所有节点
        all_nodes = memory_knowledge_store.list_document_readers(document_id="doc-e2e-7")
        stored_reader_ids = {node.reader_id for node in all_nodes}

        # 通过向量索引查询所有文档节点
        result = memory_vector_index.query(
            "测试查询",
            top_k=20,
            where={"document_id": "doc-e2e-7"}
        )
        indexed_reader_ids = {r.reader_id for r in result}

        # 验证存储中的节点都在向量索引中
        assert stored_reader_ids == indexed_reader_ids, \
            f"存储节点 {stored_reader_ids} 与索引节点 {indexed_reader_ids} 不一致"

        # 验证根节点在索引中
        assert root.reader_id in indexed_reader_ids


# =============================================================================
# 边界情况测试类
# =============================================================================
class TestEndToEndEdgeCases:
    """端到端边界情况测试"""

    @pytest.fixture
    def config(self):
        """返回测试配置"""
        return IngestionConfig(
            max_leaf_chars=100,
            max_depth=4,
            complexity_threshold=1000.0
        )

    @pytest.fixture
    def setup_builder(self, config):
        """创建测试用的 builder"""
        store = SQLiteKnowledgeStore()
        store.init_schema()
        
        # 使用 Mock 的 VectorIndex
        vector_index = Mock(spec=VectorIndex)
        indexed_nodes = {}
        
        def upsert_reader(node):
            indexed_nodes[node.reader_id] = node
        
        def delete_document(document_id):
            indexed_nodes.clear()
        
        def query(question, *, top_k, where):
            document_id = where.get("document_id")
            results = []
            for reader_id, node in indexed_nodes.items():
                if document_id is None or node.document_id == document_id:
                    from hmr.domain import VectorCandidate
                    results.append(VectorCandidate(
                        reader_id=reader_id,
                        score=1.0,
                        document="test",
                        metadata={"document_id": node.document_id}
                    ))
            return results[:top_k]
        
        vector_index.upsert_reader = upsert_reader
        vector_index.delete_document = delete_document
        vector_index.query = query
        vector_index.close = Mock()
        
        llm_service = HeuristicReaderLLMService()
        
        def _builder(
            splitter: SemanticTextSplitter | Mock = None,
            complexity: ComplexityEstimator | Mock = None
        ):
            return ReaderTreeBuilder(
                config=config,
                llm_service=llm_service,
                store=store,
                vector_index=vector_index,
                complexity_estimator=complexity or ComplexityEstimator(),
                document_spliter=splitter or SemanticTextSplitter(max_chars=config.max_leaf_chars)
            )
        
        yield _builder, store, vector_index
        
        store.close()

    # =========================================================================
    # 边界测试：空文档
    # =========================================================================
    def test_empty_document(self, setup_builder):
        """
        边界测试：摄入空文档
        预期结果：不构建读者节点，返回 None
        """
        builder, store, vector_index = setup_builder

        root = builder().ingest_document(
            document_id="doc-empty",
            title="空文档测试",
            text=""
        )

        assert root is None, "空文档不应创建节点"

        # 验证存储中没有该文档的节点
        nodes = store.list_document_readers(document_id="doc-empty")
        assert len(nodes) == 0, "空文档不应有节点"

    # =========================================================================
    # 边界测试：深度限制
    # =========================================================================
    def test_max_depth_enforcement(self, setup_builder, config):
        """
        边界测试：深度限制生效
        预期结果：达到 max_depth 后不再分割
        """
        builder_func, store, vector_index = setup_builder

        # 使用较低的深度限制
        config.max_depth = 1

        # 足够长的文本以满足长度条件
        long_text = "这是一个测试文档，长度足够长以触发分割。" * 20

        # Mock splitter 每次都返回多个 chunks
        mock_splitter = create_mock_splitter({
            long_text: ["chunk1", "chunk2"],      # depth=0 分割
            "chunk1": ["subchunk1", "subchunk2"], # depth=1 不应分割（达到深度限制）
            "chunk2": []
        })

        # Mock complexity estimator 返回高评分确保 depth=0 时触发分割
        mock_complexity = create_mock_complexity({
            long_text: 2000.0,
            "chunk1": 2000.0,  # 即使评分高，也不会分割（达到深度限制）
            "chunk2": 2000.0,
        })

        builder = builder_func(splitter=mock_splitter, complexity=mock_complexity)

        root = builder.ingest_document(
            document_id="doc-depth-limit",
            title="深度限制测试",
            text=long_text
        )

        # 获取所有节点
        all_nodes = store.list_document_readers(document_id="doc-depth-limit")

        # 预期：1(根) + 2(depth1) = 3 个节点，depth1 的节点不应有子节点
        counts = count_nodes_by_depth(all_nodes)
        assert counts == {0: 1, 1: 2}, f"深度限制未生效: {counts}"

        # 验证 depth=1 的节点都是叶子
        for node in all_nodes:
            if node.depth == 1:
                assert node.is_leaf, "depth=1 的节点应为叶子节点"

    # =========================================================================
    # 边界测试：单个子节点不分割
    # =========================================================================
    def test_single_child_no_split(self, setup_builder):
        """
        边界测试：分割器返回单个 chunk 时不分割
        预期结果：只创建根节点，不创建子节点
        """
        builder_func, store, vector_index = setup_builder

        # 使用足够长的文本以触发分割检查
        long_text = "这是一个测试文档，长度足够长以触发分割检查。" * 20
        
        # Mock splitter 返回单个 chunk（即使文本很长）
        mock_splitter = create_mock_splitter({
            long_text: ["single_chunk"]  # 返回单个 chunk，不应分割
        })

        # Mock complexity 返回高评分确保触发分割检查
        mock_complexity = create_mock_complexity({
            long_text: 2000.0,
        })

        builder = builder_func(splitter=mock_splitter, complexity=mock_complexity)

        root = builder.ingest_document(
            document_id="doc-single-child",
            title="单 chunk 测试",
            text=long_text
        )

        # 验证只有根节点，没有子节点
        assert root is not None
        assert root.is_leaf, "单 chunk 不应创建子节点"

        all_nodes = store.list_document_readers(document_id="doc-single-child")
        assert len(all_nodes) == 1, "单 chunk 应只有一个节点"

    # =========================================================================
    # 边界测试：特殊字符处理
    # =========================================================================
    def test_special_characters_document(self, setup_builder):
        """
        边界测试：包含特殊字符的文档
        预期结果：特殊字符不出现乱码，文档被正确处理
        """
        builder_func, store, vector_index = setup_builder

        special_text = "Document with special chars: \n\t\r\"'\\ and emoji: 🎉🌍"
        special_text += "中文内容测试：测试特殊字符处理能力。"

        root = builder_func().ingest_document(
            document_id="doc-special",
            title="特殊字符测试",
            text=special_text
        )

        # 验证文档被正确处理
        assert root is not None
        stored = store.get_reader(root.reader_id)
        assert stored is not None
        
        # 验证特殊字符没有乱码
        assert "\uFFFD" not in stored.text, "不应有替换字符（乱码）"
        assert "\uFFFD" not in stored.title, "标题不应有替换字符"

    # =========================================================================
    # 边界测试：文档删除级联
    # =========================================================================
    def test_document_deletion_cascade(self, setup_builder):
        """
        边界测试：删除文档时级联删除所有关联节点
        预期结果：所有节点从存储和向量索引中移除
        """
        builder_func, store, vector_index = setup_builder

        # 足够长的文本以满足长度条件
        long_text = "这是一个测试文档，长度足够长以触发分割。" * 20

        # Mock splitter 创建多个节点
        mock_splitter = create_mock_splitter({
            long_text: ["chunk1", "chunk2", "chunk3"]
        })

        # Mock complexity estimator 返回高评分确保触发分割
        mock_complexity = create_mock_complexity({
            long_text: 2000.0,
        })

        builder = builder_func(splitter=mock_splitter, complexity=mock_complexity)

        root = builder.ingest_document(
            document_id="doc-delete",
            title="删除测试",
            text=long_text
        )

        # 记录初始节点数
        initial_nodes = store.list_document_readers(document_id="doc-delete")
        initial_count = len(initial_nodes)
        assert initial_count > 1, "应有多个节点"

        # 删除文档
        store.delete_document("doc-delete")
        vector_index.delete_document("doc-delete")

        # 验证所有节点被删除
        remaining_nodes = store.list_document_readers(document_id="doc-delete")
        assert len(remaining_nodes) == 0, "所有节点应被删除"

        # 验证向量索引中也没有该文档的节点
        result = vector_index.query(
            "测试查询",
            top_k=10,
            where={"document_id": "doc-delete"}
        )
        assert len(result) == 0, "向量索引中不应有该文档的节点"