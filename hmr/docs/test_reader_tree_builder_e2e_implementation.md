# ReaderTreeBuilder 端到端测试实现说明文档

## 1. 文档概述

本文档详细描述 `hmr/test/test_reader_tree_builder_e2e.py` 的测试实现，与当前稳定的测试代码严格对应。

---

## 2. 测试架构与组件配置

### 2.1 测试分层结构

```
┌─────────────────────────────────────────────────────────────┐
│ 测试类                    │ 测试范围                          │
├─────────────────────────────────────────────────────────────┤
│ TestReaderTreeBuilderEndToEnd │ 核心端到端测试用例             │
│ TestEndToEndEdgeCases         │ 边界情况测试                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件配置策略

| 组件 | 实现方式 | 说明 |
|------|---------|------|
| LLM Service | `HeuristicReaderLLMService` | 真实实现，确定性启发式算法 |
| KnowledgeStore | `SQLiteKnowledgeStore()` | 内存模式，不传路径参数 |
| VectorIndex | `ChromaVectorIndex` / Mock | 优先使用真实实现，缺失时回退到 Mock |
| SemanticTextSplitter | Mock | 完全控制分割行为 |
| ComplexityEstimator | Mock / 真实 | 根据测试需求选择 |

### 2.3 Fixture 定义

| Fixture | 类型 | 作用 |
|---------|------|------|
| `memory_knowledge_store` | 函数 | 返回内存 SQLite 存储 |
| `memory_vector_index` | 函数 | 返回向量索引（真实/Mock） |
| `heuristic_llm_service` | 函数 | 返回 HeuristicReaderLLMService |
| `config` | 函数 | 返回 IngestionConfig 配置 |
| `default_builder` | 函数 | 创建默认配置的 ReaderTreeBuilder |
| `setup_builder` | 函数 | 边界测试专用的 builder 工厂 |

---

## 3. 测试数据样本

### 3.1 E2E_TEST_DOCUMENTS 字典

| 样本 ID | 名称 | 特征 | 用途 |
|--------|------|------|------|
| E2E-D1 | 简单技术文档 | 约 100 字符，低复杂度 | 单节点创建测试 |
| E2E-D2 | 中等复杂度文档 | 约 300 字符，包含技术术语 | 知识提取测试 |
| E2E-D3 | 高复杂度文档 | 约 500 字符，多段落 | 多级分割测试 |
| E2E-D4 | 多段落文档 | 5 个段落 | 段落分割测试 |
| E2E-D5 | 超长文档 | 重复文本 * 50 | 触发长度条件测试 |

---

## 4. 辅助函数实现

### 4.1 树结构验证函数

#### `verify_tree_structure(root_node, all_nodes)`

验证内容：
1. **根节点验证**：parent_id=None, depth=0, ordinal=0
2. **父-子关系验证**：子节点的 parent_id 指向正确的父节点，depth = parent.depth + 1
3. **child_ids 完整性验证**：父节点的 child_ids 包含所有子节点
4. **ordinal 顺序验证**：同一父节点的子节点 ordinal 连续从 0 开始
5. **is_leaf 属性验证**：有 child_ids 的节点 is_leaf=False

#### `count_nodes_by_depth(nodes)`

统计每层节点数量，返回 `dict[int, int]`

#### `verify_tree_connectivity(root_node, all_nodes)`

验证所有节点都可从根节点遍历到达

### 4.2 Mock 创建函数

#### `create_mock_splitter(chunk_map: dict[str, List[str]])`

根据文本内容返回预设的 chunks：

```python
mock_splitter = create_mock_splitter({
    "text1": ["chunk1", "chunk2"],
    "text2": ["chunk3"],
})
```

#### `create_mock_complexity(score_map: dict[str, float])`

根据文本内容返回预设的复杂度评分：

```python
mock_complexity = create_mock_complexity({
    "complex_text": 2000.0,
    "simple_text": 500.0,
})
```

---

## 5. 端到端测试用例实现

### 5.1 测试用例清单

| 测试 ID | 测试方法 | 测试场景 | Mock 策略 |
|---------|---------|---------|----------|
| TC-RTB-E-001 | `test_simple_document_e2e` | 简单文档，无分割 | 无 Mock，使用真实组件 |
| TC-RTB-E-002 | `test_multi_level_split_e2e` | 多级分割，复杂树结构 | Mock Splitter + Mock Complexity |
| TC-RTB-E-003 | `test_document_overwrite_e2e` | 文档覆盖写入 | 无 Mock，使用真实组件 |
| TC-RTB-E-004 | `test_tree_traversal_e2e` | 树遍历验证 | Mock Splitter + Mock Complexity |
| TC-RTB-E-005 | `test_knowledge_extraction_e2e` | 知识提取验证 | 无 Mock，使用真实组件 |
| TC-RTB-E-006 | `test_real_complexity_estimator_e2e` | 真实复杂性评估 | Mock Splitter，真实 Complexity |
| TC-RTB-E-007 | `test_vector_index_e2e` | 向量索引验证 | Mock Complexity，真实 Splitter |

### 5.2 测试用例详细说明

#### TC-RTB-E-001: test_simple_document_e2e

**测试场景**：简单文档不触发分割

**验证点**：
- 根节点 is_leaf=True
- 节点在 SQLite 中正确存储
- 节点在 VectorIndex 中可查询

**使用组件**：default_builder（全部真实组件）

---

#### TC-RTB-E-002: test_multi_level_split_e2e

**测试场景**：多级分割，复杂树结构

**预期树结构**：
```
Root (depth=0)
├── A (depth=1, ordinal=0)
│   ├── A1 (depth=2, ordinal=0)
│   └── A2 (depth=2, ordinal=1)
├── B (depth=1, ordinal=1)
│   ├── B1 (depth=2, ordinal=0)
│   └── B2 (depth=2, ordinal=1)
└── C (depth=1, ordinal=2)  # 不分割
```

**验证点**：
- 总计 8 个节点（1根 + 3子 + 4孙子）
- 树结构完整性验证
- ordinal 在不同父节点间重叠（depth=2 的 ordinal 为 [0,0,1,1]）

**Mock 配置**：
- Splitter：按文本内容返回预设 chunks
- Complexity：控制哪些节点分割（2000.0）哪些不分割（500.0）

---

#### TC-RTB-E-003: test_document_overwrite_e2e

**测试场景**：同一 document_id 摄入两次

**验证点**：
- 第二次摄入后，第一次的节点被删除
- 第二次的节点正确保存
- 向量索引中只有第二次的节点

**使用组件**：default_builder（全部真实组件）

---

#### TC-RTB-E-004: test_tree_traversal_e2e

**测试场景**：树遍历 API 验证

**验证点**：
- `list_children` 返回正确数量的子节点（5个）
- 子节点 ordinal 从 0 递增
- 子节点标题格式正确：`{parent_title} / part-{ordinal+1}`

**Mock 配置**：
- Splitter：返回 5 个 chunks
- Complexity：返回高评分触发分割

---

#### TC-RTB-E-005: test_knowledge_extraction_e2e

**测试场景**：知识提取功能验证

**验证点**：
- ReaderKnowledge.summary 不为空
- ReaderKnowledge.entities 有内容
- ReaderKnowledge.capability_questions 有内容（最多8个）

**使用组件**：default_builder（全部真实组件）

---

#### TC-RTB-E-006: test_real_complexity_estimator_e2e

**测试场景**：真实复杂度评估器验证

**验证点**：
- 复杂文本的复杂度评分 >= threshold（1000.0）
- 文本长度 > max_leaf_chars（100）
- 文档被正确分割

**Mock 配置**：
- Splitter：Mock（强制返回多个 chunks，绕过 SemanticTextSplitter 限制）
- Complexity：真实实现

**技术说明**：使用 Mock Splitter 是因为真实的 SemanticTextSplitter 不会拆分长句子，可能导致分割失败。

---

#### TC-RTB-E-007: test_vector_index_e2e

**测试场景**：向量索引一致性验证

**验证点**：
- 存储中的所有节点都在向量索引中
- 向量索引中的节点与存储中的节点一致

**Mock 配置**：
- Complexity：Mock（返回高评分确保分割）
- Splitter：真实 SemanticTextSplitter（小 max_chars=50 确保分割）

---

## 6. 边界情况测试实现

### 6.1 测试用例清单

| 测试方法 | 测试场景 | 预期结果 |
|---------|---------|---------|
| `test_empty_document` | 空文档摄入 | 返回 None，不创建节点 |
| `test_max_depth_enforcement` | 深度限制生效 | 达到 max_depth 后不再分割 |
| `test_single_child_no_split` | 分割器返回单个 chunk | 不创建子节点 |
| `test_special_characters_document` | 特殊字符文档 | 特殊字符不出现乱码 |
| `test_document_deletion_cascade` | 删除文档 | 所有关联节点被删除 |

### 6.2 边界测试详细说明

#### test_empty_document

**测试场景**：空文本摄入

**验证点**：
- `ingest_document` 返回 None
- 存储中无该文档的节点

---

#### test_max_depth_enforcement

**测试场景**：深度限制验证

**配置**：
- max_depth = 1

**验证点**：
- 根节点（depth=0）可以分割
- depth=1 的节点不分割（即使满足其他条件）
- depth=1 的节点 is_leaf=True

**Mock 配置**：
- Splitter：返回多个 chunks
- Complexity：所有文本返回高评分

---

#### test_single_child_no_split

**测试场景**：分割器返回单个 chunk

**验证点**：
- 即使文本很长、复杂度很高
- 只创建根节点，不创建子节点

**Mock 配置**：
- Splitter：返回 `["single_chunk"]`
- Complexity：返回高评分

---

#### test_special_characters_document

**测试场景**：包含特殊字符的文档

**验证点**：
- 文档能正常处理
- 不出现乱码
- 节点正确创建

---

#### test_document_deletion_cascade

**测试场景**：文档删除级联

**验证点**：
- 删除文档后，所有关联节点从存储中删除
- 所有关联节点从向量索引中删除

---

## 7. 测试配置参数

### 7.1 默认配置

```python
IngestionConfig(
    max_leaf_chars=100,      # 叶子节点最大字符数
    max_depth=4,             # 最大递归深度
    complexity_threshold=1000.0  # 复杂度阈值
)
```

### 7.2 分割触发条件

文档分割需要同时满足：
1. `len(text) > max_leaf_chars`
2. `complexity.score(text) >= complexity_threshold`
3. `depth < max_depth`

---

## 8. Mock 策略总结

### 8.1 Mock 使用原则

| 场景 | Splitter | Complexity | 原因 |
|------|----------|------------|------|
| 控制分割行为 | Mock | 任意 | 需要精确控制分割结果 |
| 验证复杂度评估 | 任意 | 真实 | 测试复杂度评估器本身 |
| 验证真实分割 | 真实 | 任意 | 测试真实分割器行为 |
| 边界情况测试 | Mock | Mock | 精确控制边界条件 |

### 8.2 Mock Splitter 设计选择

当前实现使用**字典映射**方式（按文本内容返回），而非顺序列表方式。原因：
- 递归分割时，每次调用 `split()` 的文本参数不同
- 需要根据文本内容返回对应的 chunks
- 顺序列表方式无法处理递归场景

---

## 9. 测试覆盖率

### 9.1 覆盖目标

| 覆盖率类型 | 目标值 | 覆盖重点 |
|---------|-------|---------|
| 语句覆盖率 | ≥ 98% | `_build_node()`, `_build_children_if_needed()`, `_should_split()` |
| 分支覆盖率 | ≥ 98% | `_should_split()` 的所有条件分支 |

### 9.2 当前覆盖情况

当前测试覆盖：
- ✅ 单节点创建
- ✅ 单级分割（2-5个子节点）
- ✅ 多级分割（复杂树结构）
- ✅ 深度限制边界
- ✅ 空文档边界
- ✅ 单 chunk 不分割边界
- ✅ 文档覆盖写入
- ✅ 树遍历 API
- ✅ 知识提取功能
- ✅ 向量索引一致性
- ✅ 文档删除级联

---

## 附录：测试执行

### 执行所有端到端测试

```bash
python -m pytest hmr/test/test_reader_tree_builder_e2e.py -v
```

### 执行特定测试

```bash
# 执行单个测试用例
python -m pytest hmr/test/test_reader_tree_builder_e2e.py::TestReaderTreeBuilderEndToEnd::test_multi_level_split_e2e -v

# 执行边界测试
python -m pytest hmr/test/test_reader_tree_builder_e2e.py::TestEndToEndEdgeCases -v
```

---

**文档版本**: v1.0  
**生成日期**: 2026-06-02  
**对应代码**: `hmr/test/test_reader_tree_builder_e2e.py`