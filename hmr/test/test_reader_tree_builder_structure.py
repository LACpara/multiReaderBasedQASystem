"""
ReaderTreeBuilder 树结构一致性验证单元测试

测试构建的 Reader 树结构中各节点属性的一致性和正确性。
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


def create_complexity_that_splits():
    """创建高复杂度评分器（触发分割）"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.return_value = 2000.0  # > threshold
    return mock


def create_builder_with_three_level_tree(mock_llm_service, mock_store, mock_vector_index):
    """创建一个 3 层树的 builder 用于测试"""
    config = IngestionConfig(
        max_leaf_chars=50,
        max_depth=2,
        complexity_threshold=500.0
    )

    splitter = Mock(spec=SemanticTextSplitter)
    splitter.split.return_value = [
        make_long_chunk("_1"),
        make_long_chunk("_2"),
    ]

    complexity = create_complexity_that_splits()

    builder = ReaderTreeBuilder(
        config=config,
        llm_service=mock_llm_service,
        store=mock_store,
        vector_index=mock_vector_index,
        complexity_estimator=complexity,
        document_spliter=splitter
    )

    return builder


class TestTreeStructureConsistency:
    """树结构一致性验证测试"""

    # =========================================================================
    # TC-RTB-U-041: parent_id 指向正确
    # =========================================================================
    def test_parent_id_correctness(self, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-041: 每个子节点的 parent_id 应等于父节点的 reader_id

        测试场景：构建多层树结构
        预期结果：所有非根节点的 parent_id 都正确指向父节点
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-parentid",
            title="Parent ID Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 验证每个非根节点的 parent_id
        for node in all_nodes:
            if node.depth == 0:
                # 根节点的 parent_id 应为 None
                assert node.parent_id is None
            else:
                # 非根节点的 parent_id 应指向父节点
                parent = next((n for n in all_nodes if n.reader_id == node.parent_id), None)
                assert parent is not None, f"Node {node.reader_id} has invalid parent_id"
                assert parent.depth == node.depth - 1, \
                    f"Node depth={node.depth} but parent depth={parent.depth}"

    # =========================================================================
    # TC-RTB-U-042: ordinal 顺序正确
    # =========================================================================
    def test_ordinal_sequential_order(self, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-042: 兄弟节点的 ordinal 应连续且从 0 开始

        测试场景：构建多层树结构
        预期结果：每个父节点下的子节点 ordinal 从 0 开始连续编号

        注意：ordinal 在每个父节点下独立重新计数，不是在整个深度层全局连续
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-ordinal",
            title="Ordinal Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 验证每个父节点的子节点 ordinal 连续
        for parent in all_nodes:
            children = [n for n in all_nodes if n.parent_id == parent.reader_id]
            if not children:
                continue
            children.sort(key=lambda n: n.ordinal)
            expected_ordinals = list(range(len(children)))
            actual_ordinals = [n.ordinal for n in children]
            assert actual_ordinals == expected_ordinals, \
                f"Parent {parent.reader_id}: expected ordinals {expected_ordinals}, got {actual_ordinals}"

        # 额外验证：depth=1 节点 ordinal 连续（0, 1）
        depth1_nodes = [n for n in all_nodes if n.depth == 1]
        depth1_nodes.sort(key=lambda n: n.ordinal)
        assert [n.ordinal for n in depth1_nodes] == [0, 1]

        # 验证 depth=2 节点：每个 depth=1 父节点下各有 2 个子节点，ordinal 都是 (0, 1)
        depth1_to_depth2 = {}
        for n in all_nodes:
            if n.depth == 2:
                parent_id = n.parent_id
                depth1_to_depth2.setdefault(parent_id, []).append(n)

        for parent_id, children in depth1_to_depth2.items():
            children.sort(key=lambda n: n.ordinal)
            assert [n.ordinal for n in children] == [0, 1], \
                f"Parent {parent_id} children ordinals should be [0, 1]"

    # =========================================================================
    # TC-RTB-U-043: depth 层级正确
    # =========================================================================
    def test_depth_hierarchy_correct(self, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-043: 每个节点的 depth 应等于 parent.depth + 1

        测试场景：构建多层树结构
        预期结果：每个节点的深度都是父节点深度 + 1
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-depth",
            title="Depth Hierarchy Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 验证每个节点的深度
        for node in all_nodes:
            if node.depth == 0:
                # 根节点
                assert node.parent_id is None
            else:
                # 非根节点
                parent = next((n for n in all_nodes if n.reader_id == node.parent_id), None)
                assert parent is not None
                assert node.depth == parent.depth + 1, \
                    f"Node {node.reader_id} has depth={node.depth} but parent has depth={parent.depth}"

    # =========================================================================
    # TC-RTB-U-044: child_ids 引用完整
    # =========================================================================
    def test_child_ids_complete_reference(self, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-044: 父节点的 child_ids 应包含所有子节点的 reader_id

        测试场景：构建多层树结构
        预期结果：child_ids 包含所有子节点，且不包含不存在的节点
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-childids",
            title="Child IDs Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]
        all_reader_ids = {n.reader_id for n in all_nodes}

        # 验证每个节点的 child_ids
        for node in all_nodes:
            # 验证 child_ids 中的每个 ID 都存在
            for child_id in node.child_ids:
                assert child_id in all_reader_ids, \
                    f"Node {node.reader_id} references non-existent child {child_id}"

            # 验证父节点包含所有子节点
            children = [n for n in all_nodes if n.parent_id == node.reader_id]
            assert set(node.child_ids) == {c.reader_id for c in children}, \
                f"Node {node.reader_id} child_ids mismatch: " \
                f"expected {[c.reader_id for c in children]}, got {node.child_ids}"

    # =========================================================================
    # TC-RTB-U-045: is_leaf 属性正确
    # =========================================================================
    def test_is_leaf_property_consistent(self, mock_llm_service, mock_store, mock_vector_index):
        """
        TC-RTB-U-045: is_leaf 应正确反映节点是否有子节点

        测试场景：构建多层树结构
        预期结果：is_leaf=True 当且仅当 child_ids 为空
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-isleaf",
            title="Is Leaf Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 验证 is_leaf 属性
        for node in all_nodes:
            expected_is_leaf = len(node.child_ids) == 0
            assert node.is_leaf == expected_is_leaf, \
                f"Node {node.reader_id}: is_leaf={node.is_leaf}, child_ids={node.child_ids}"

    # =========================================================================
    # 补充测试：树结构的完整性
    # =========================================================================

    def test_tree_connectivity(self, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证树结构是连通的（所有节点都可从根节点到达）
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-connectivity",
            title="Connectivity Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 构建 reader_id -> node 映射
        node_map = {n.reader_id: n for n in all_nodes}

        # 验证每个节点都可以从根节点到达
        def can_reach_from_root(node):
            if node.reader_id == root.reader_id:
                return True
            if node.parent_id is None:
                return node.depth == 0
            parent = node_map.get(node.parent_id)
            if parent is None:
                return False
            return can_reach_from_root(parent)

        for node in all_nodes:
            assert can_reach_from_root(node), \
                f"Node {node.reader_id} is not reachable from root"

    def test_tree_node_count_by_depth(self, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证树结构每层节点数量符合预期

        配置：splitter 每次返回 2 个 chunks，max_depth=2
        预期：depth=0: 1, depth=1: 2, depth=2: 4
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-nodecount",
            title="Node Count Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 统计每层节点数
        depth_counts = {}
        for node in all_nodes:
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1

        # 验证节点数：1 + 2 + 4 = 7
        assert depth_counts.get(0, 0) == 1   # 根节点
        assert depth_counts.get(1, 0) == 2   # depth=1: 根节点分割为 2
        assert depth_counts.get(2, 0) == 4   # depth=2: 每个 depth=1 节点分割为 2
        assert len(all_nodes) == 7

    def test_root_has_no_parent(self, mock_llm_service, mock_store, mock_vector_index):
        """
        补充测试：验证根节点是唯一的且没有父节点
        """
        builder = create_builder_with_three_level_tree(
            mock_llm_service, mock_store, mock_vector_index
        )

        root = builder.ingest_document(
            document_id="doc-root",
            title="Root Test",
            text=make_long_chunk("_root")
        )

        # 获取所有节点
        upsert_calls = mock_store.upsert_reader.call_args_list
        all_nodes = [call[0][0] for call in upsert_calls]

        # 验证只有一个根节点
        root_nodes = [n for n in all_nodes if n.depth == 0]
        assert len(root_nodes) == 1
        assert root_nodes[0].reader_id == root.reader_id

        # 验证根节点没有父节点
        assert root.parent_id is None
        assert root.depth == 0
        assert root.ordinal == 0
