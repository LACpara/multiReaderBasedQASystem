# ReaderTreeBuilder 测试用例设计文档（修订版）

## 1. 测试概述

### 1.1 修订要点

| 变更项      | 变更内容                                    |
| -------- | --------------------------------------- |
| 切分器 Mock | 完全 Mock 掉 SemanticTextSplitter，不验证其切分逻辑 |
| 重点方向     | 聚焦于 Reader 递归创建逻辑的完整性和正确性               |
| 多级分割测试   | 设计全面的边界场景测试用例                           |
| 端到端测试    | 新增内存数据库模式的端到端构建测试                       |

### 1.2 被测对象

| 对象                | 路径                      | 职责                    |
| ----------------- | ----------------------- | --------------------- |
| ReaderTreeBuilder | `hmr/reader_builder.py` | 从输入文档构建递归的 Reader 树结构 |

***

## 2. 测试策略与架构

### 2.1 测试分类与 Mock 策略

| 测试类别  | 描述       | LLM Service    | KnowledgeStore | VectorIndex   | SemanticTextSplitter | ComplexityEstimator |
| ----- | -------- | -------------- | -------------- | ------------- | -------------------- | ------------------- |
| 单元测试  | 测试递归创建逻辑 | Mock           | Mock           | Mock          | **Mock**             | Mock/真实             |
| 端到端测试 | 完整构建流程   | 真实 (Heuristic) | **内存 SQLite**  | **内存 Chroma** | Mock                 | 真实                  |

### 2.2 核心测试关注点

1. **Reader 递归创建逻辑** - 深度、广度、父子关系
2. **分割决策** - `_should_split()` 的各种组合条件
3. **树结构验证** - depth/ordinal/parent\_id/child\_ids 的一致性
4. **多级分割边界场景** - 深度限制、空 chunks、各种配置组合

***

## 3. 测试用例设计

### 3.1 单元测试：Reader 递归创建逻辑

#### 3.1.1 基本创建测试

| 测试 ID        | 测试名称                                          | 测试场景                  | 预期结果                                                        |
| ------------ | --------------------------------------------- | --------------------- | ----------------------------------------------------------- |
| TC-RTB-U-001 | 单叶子节点创建                                       | 不满足任何分割条件             | 创建单个 ReaderNode，无 children，depth=0，ordinal=0                |
| TC-RTB-U-002 | Reader 基本属性验证                                 | 验证所有必填字段              | reader\_id 格式正确，所有属性正确赋值                                    |
| TC-RTB-U-003 | LLM 调用验证                                      | 每个节点调用 LLM            | extract\_knowledge 和 build\_capability\_questions 各调用 N 次   |
| TC-RTB-U-004 | 持久化调用验证                                       | 每个节点被持久化              | store.upsert\_reader 和 vector\_index.upsert\_reader 各调用 N 次 |
| TC-RTB-U-005 | 文档删除验证                                        | 重复摄入同一文档              | delete\_document 在构建前被调用                                    |
| 补充测试 - 1     | ***test\_default\_complexity\_estimator***    | 实例化 builder 的时候不传入评估器 | builder.complexity 使用了默认实现实例化                               |
| 补充测试 - 2     | ***test\_empty\_text\_document***             | 空白文档的摄入               | 不构建读者节点                                                     |
| 补充测试 - 3     | ***test\_special\_characters\_in\_document*** | 包含特殊字符文档的摄入           | 特殊字符不出现乱码                                                   |
| 补充测试 - 4     | ***test\_splitter\_return\_empty\_text***     | 分割器返回空白文档             | 忽略空白分割结果                                                    |

#### 3.1.2 单级分割测试（depth=0 → depth=1）

| 测试 ID        | 测试名称         | Mock 配置              | 预期结果                                             |
| ------------ | ------------ | -------------------- | ------------------------------------------------ |
| TC-RTB-U-011 | 2 个子节点       | splitter 返回 2 chunks | 根节点有 2 个子节点，child\_ids 长度为 2                     |
| TC-RTB-U-012 | 5 个子节点       | splitter 返回 5 chunks | 根节点有 5 个子节点，ordinal 从 0 到 4                      |
| TC-RTB-U-013 | 子节点标题格式      | splitter 返回 3 chunks | 子节点标题为 `{parent} / part-1`、`/ part-2`、`/ part-3` |
| TC-RTB-U-014 | 单个 chunk 不分割 | splitter 返回 1 chunk  | 不创建子节点，根节点为叶子                                    |
| TC-RTB-U-015 | 空 chunks 不分割 | splitter 返回空列表       | 不创建子节点                                           |

#### 3.1.3 多级分割深度测试

| 测试 ID             | 测试名称                                         | 测试配置                              | 树结构预期                                       |
| ----------------- | -------------------------------------------- | --------------------------------- | ------------------------------------------- |
| TC-RTB-U-021      | 2 级完整树                                       | depth=0 分割为 3 节点，每个子节点再分割为 2      | 总计 1 (根) + 3 (depth1) + 6 (depth2) = 10 个节点 |
| TC-RTB-U-022      | 部分子节点分割                                      | depth=0 分割为 3 节点，仅第 2 个子节点继续分割    | 子树结构为：根(3子) → 子1(叶)、子2(2子)、子3(叶)            |
| TC-RTB-U-023      | 深度边界测试                                       | max\_depth=2，depth=2 节点也满足分割条件    | depth=2 节点不继续分割，成为叶子                        |
| TC-RTB-U-024 (弃用) | 锯齿状深度树                                       | 配置不同节点有不同分割决策                     | 混合深度的树结构正确构建                                |
| TC-RTB-U-025      | max\_depth=0                                 | max\_depth=0                      | 只创建根节点，即使满足分割条件也不分割                         |
| 补充测试 - 1          | ***test\_single\_child\_split***             | max\_depth=4, spliter 每次均返回单个分割结果 | 只创建根节点，即使满足分割条件                             |
| 补充测试 - 2          | ***test\_ordinal\_resets\_at\_each\_level*** | max\_depth=1                      | depth=1 （子节点）的序号是 0, 1, 2 （进入新一分割层级序号重置）    |

#### 3.1.4 分割决策逻辑测试（核心）

| 测试 ID        | 测试名称       | 条件组合                                                   | should\_split 预期 |
| ------------ | ---------- | ------------------------------------------------------ | ---------------- |
| TC-RTB-U-031 | 深度超限       | depth = max\_depth                                     | False            |
| TC-RTB-U-032 | 复杂性不足      | score < threshold                                      | False            |
| TC-RTB-U-033 | 长度不足       | len(text) ≤ max\_leaf                                  | False            |
| TC-RTB-U-034 | 复杂性达标但长度不足 | score ≥ threshold, len ≤ max\_leaf                     | False            |
| TC-RTB-U-035 | 长度达标但复杂性不足 | score < threshold, len > max\_leaf                     | False            |
| TC-RTB-U-036 | 所有条件满足     | score ≥ threshold, len > max\_leaf, depth < max\_depth | True             |
| TC-RTB-U-037 | 空文本        | text = ""                                              | False            |

#### 3.1.5 树结构一致性验证

| 测试 ID        | 测试名称                                     | 验证点                                     |
| ------------ | ---------------------------------------- | --------------------------------------- |
| TC-RTB-U-041 | parent\_id 指向正确                          | 每个子节点的 parent\_id 等于父节点 reader\_id      |
| TC-RTB-U-042 | ordinal 顺序正确                             | 兄弟节点的 ordinal 连续且从 0 开始                 |
| TC-RTB-U-043 | depth 层级正确                               | 每个节点的 depth = parent.depth + 1          |
| TC-RTB-U-044 | child\_ids 引用完整                          | 父节点 child\_ids 包含所有子节点的 reader\_id      |
| TC-RTB-U-045 | is\_leaf 属性正确                            | 有 child\_ids 的节点 is\_leaf=False，否则 True |
| 补充测试 - 1     | ***test\_tree\_connectivity***           | 每个节点都可以从根节点到达                           |
| 补充测试 - 2     | ***test\_tree\_node\_count\_by\_depth*** | 验证树结构的每层节点数量符合预期                        |
| 补充测试 - 3     | ***test\_root\_has\_no\_parent***        | 验证根节点是唯一的且没有父节点                         |

***

### 3.2 端到端测试：内存数据库完整构建

#### 3.2.1 内存数据库配置

| 组件                   | 实现方式                                              |
| -------------------- | ------------------------------------------------- |
| LLM Service          | `HeuristicReaderLLMService`（确定性，无外部依赖）            |
| KnowledgeStore       | `SQLiteKnowledgeStore(db_path=":memory:")`        |
| VectorIndex          | `ChromaVectorIndex` 配合 Chroma 的 `EphemeralClient` |
| SemanticTextSplitter | Mock，控制返回特定 chunks                                |
| ComplexityEstimator  | 真实实现（或 Mock 控制评分）                                 |

#### 3.2.2 端到端测试用例

| 测试 ID        | 测试名称    | 测试场景                     | 验证维度                                     |
| ------------ | ------- | ------------------------ | ---------------------------------------- |
| TC-RTB-E-001 | 简单文档端到端 | 无分割，单节点                  | 节点在 SQLite 和 Chroma 中正确保存，可查询            |
| TC-RTB-E-002 | 多级分割端到端 | 2 级分割，5 个节点              | 树结构完整性，list\_document\_readers 返回正确数量    |
| TC-RTB-E-003 | 文档覆盖写入  | 同一 document\_id 摄入两次     | 第一次的节点被删除，第二次的节点正确保存                     |
| TC-RTB-E-004 | 树遍历验证   | 构建树后通过 API 遍历            | list\_children 返回正确的子节点顺序                |
| TC-RTB-E-005 | 知识提取验证  | 使用 HeuristicService 提取   | ReaderKnowledge 包含预期的 summary、entities 等 |
| TC-RTB-E-006 | 真实复杂性评估 | 使用真实 ComplexityEstimator | 验证分割决策与复杂度评分一致                           |
| TC-RTB-E-007 | 向量索引验证  | 构建完成后查询 VectorIndex      | 可通过 document\_id 查询到所有节点                 |

***

## 4. 测试数据与 Mock 配置

### 4.1 Mock Splitter 配置策略

通过 Mock 的 `split()` 方法返回预设的 chunks 序列，完全控制分割行为：

```python
# 示例：控制 splitter 返回特定 chunks
mock_splitter = Mock(spec=SemanticTextSplitter)
mock_splitter.split.side_effect = [
    ["chunk1-1", "chunk1-2", "chunk1-3"],  # depth=0 分割为 3 个
    ["chunk2-1", "chunk2-2"],              # depth=1, ordinal=0 分割为 2 个
    [],                                    # depth=1, ordinal=1 不分割
    ["chunk2-3"]                           # depth=1, ordinal=2 返回 1 个（不分割）
]
```

### 4.2 Mock ComplexityEstimator 配置

通过 Mock 的 `score()` 方法返回预设值，精确控制分割决策：

```python
# 示例：控制不同文本返回不同复杂度评分
mock_complexity = Mock(spec=ComplexityEstimator)
mock_complexity.score.side_effect = lambda text: {
    "very_complex_text": 2000.0,   # ≥ threshold
    "simple_text": 500.0,          # < threshold
    "".get(text, 100.0)
}
```

### 4.3 测试文档样本（端到端测试）

| 样本 ID  | 名称      | 内容特征              |
| ------ | ------- | ----------------- |
| E2E-D1 | 简单技术文档  | 约 300 字符，低复杂性     |
| E2E-D2 | 中等复杂度文档 | 约 1000 字符，包含技术术语  |
| E2E-D3 | 高复杂度文档  | 约 2000 字符，多段落，多概念 |
| E2E-D4 | 多段落文档   | 5 个段落，每段约 200 字符  |

***

## 5. 测试类结构设计

```python
# === 单元测试：递归创建逻辑 ===
class TestReaderTreeBuilderRecursion:
    """ReaderTreeBuilder 递归创建逻辑单元测试"""
    
    # === 基本创建测试 ===
    def test_single_leaf_node_creation(self): ...          # TC-RTB-U-001
    def test_reader_node_attributes(self): ...             # TC-RTB-U-002
    def test_llm_calls_per_node(self): ...                 # TC-RTB-U-003
    def test_persistence_calls(self): ...                  # TC-RTB-U-004
    def test_document_deletion_before_build(self): ...     # TC-RTB-U-005
    
    # === 单级分割测试 ===
    def test_single_level_split_2_children(self): ...      # TC-RTB-U-011
    def test_single_level_split_5_children(self): ...      # TC-RTB-U-012
    def test_child_node_title_format(self): ...            # TC-RTB-U-013
    def test_no_split_when_single_chunk(self): ...         # TC-RTB-U-014
    def test_no_split_when_empty_chunks(self): ...         # TC-RTB-U-015
    
    # === 多级分割测试 ===
    def test_two_level_full_tree(self): ...                # TC-RTB-U-021
    def test_partial_children_split(self): ...             # TC-RTB-U-022
    def test_depth_limit_enforcement(self): ...            # TC-RTB-U-023
    def test_jagged_depth_tree(self): ...                  # TC-RTB-U-024
    def test_max_depth_zero(self): ...                     # TC-RTB-U-025
    
    # === 分割决策逻辑测试 ===
    def test_should_split_depth_exceeded(self): ...        # TC-RTB-U-031
    def test_should_split_low_complexity(self): ...        # TC-RTB-U-032
    def test_should_split_short_text(self): ...            # TC-RTB-U-033
    def test_should_split_complex_ok_length_not(self): ... # TC-RTB-U-034
    def test_should_split_length_ok_complex_not(self): ... # TC-RTB-U-035
    def test_should_split_all_conditions_met(self): ...    # TC-RTB-U-036
    def test_should_split_empty_text(self): ...            # TC-RTB-U-037
    
    # === 树结构一致性验证 ===
    def test_parent_id_correctness(self): ...              # TC-RTB-U-041
    def test_ordinal_sequential_order(self): ...           # TC-RTB-U-042
    def test_depth_hierarchy_correct(self): ...            # TC-RTB-U-043
    def test_child_ids_complete_reference(self): ...       # TC-RTB-U-044
    def test_is_leaf_property_consistent(self): ...        # TC-RTB-U-045


# === 端到端测试：内存数据库 ===
class TestReaderTreeBuilderEndToEnd:
    """ReaderTreeBuilder 端到端集成测试（内存数据库）"""
    
    @pytest.fixture
    def memory_knowledge_store(self):
        """返回内存 SQLite 存储"""
        ...
    
    @pytest.fixture
    def memory_vector_index(self, tmp_path):
        """返回临时 Chroma 索引（或 EphemeralClient）"""
        ...
    
    # === 端到端测试用例 ===
    def test_simple_document_e2e(self): ...                # TC-RTB-E-001
    def test_multi_level_split_e2e(self): ...              # TC-RTB-E-002
    def test_document_overwrite_e2e(self): ...             # TC-RTB-E-003
    def test_tree_traversal_e2e(self): ...                 # TC-RTB-E-004
    def test_knowledge_extraction_e2e(self): ...           # TC-RTB-E-005
    def test_real_complexity_estimator_e2e(self): ...      # TC-RTB-E-006
    def test_vector_index_e2e(self): ...                   # TC-RTB-E-007
```

***

## 6. 测试工具与辅助函数

### 6.1 Mock 辅助函数

```python
def create_mock_splitter(chunk_sequences):
    """创建返回预设 chunks 序列的 Mock Splitter"""
    mock = Mock(spec=SemanticTextSplitter)
    mock.split.side_effect = chunk_sequences
    return mock

def create_mock_complexity(score_map):
    """创建返回预设复杂度评分的 Mock Estimator"""
    mock = Mock(spec=ComplexityEstimator)
    mock.score.side_effect = lambda text: score_map.get(text, 100.0)
    return mock
```

### 6.2 树结构验证辅助函数

```python
def verify_tree_structure(root_node, all_nodes):
    """验证树结构的完整性和一致性"""
    # 验证 parent_id 关系
    # 验证 depth 层级
    # 验证 ordinal 顺序
    # 验证 child_ids 完整性
    ...
```

***

## 7. 测试覆盖率目标

| 覆盖率类型 | 目标值   | 重点覆盖                                                              |
| ----- | ----- | ----------------------------------------------------------------- |
| 语句覆盖率 | ≥ 98% | `_build_node()`, `_build_children_if_needed()`, `_should_split()` |
| 分支覆盖率 | ≥ 98% | `_should_split()` 的所有条件分支                                         |
| 方法覆盖率 | 100%  | 所有方法                                                              |
| 类覆盖率  | 100%  | ReaderTreeBuilder                                                 |

***

## 8. 测试风险评估

| 风险等级 | 风险描述                  | 缓解措施              |
| ---- | --------------------- | ----------------- |
| 中    | 端到端测试依赖 ChromaDB      | 检查环境，失败时跳过端到端测试   |
| 低    | Mock 配置复杂易出错          | 提供辅助函数，简化 Mock 配置 |
| 低    | 树结构验证逻辑复杂             | 提供通用验证辅助函数        |
| 低    | HeuristicService 行为变化 | 测试基于稳定的启发式规则      |

