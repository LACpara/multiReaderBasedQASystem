# ReaderTreeBuilder 测试用例设计文档

## 1. 测试概述

本测试文档针对 `hmr.reader_builder.ReaderTreeBuilder` 类设计完备的测试用例，覆盖其核心功能、边界条件、异常场景等。

### 1.1 被测对象

| 对象                | 路径                      | 职责                    |
| ----------------- | ----------------------- | --------------------- |
| ReaderTreeBuilder | `hmr/reader_builder.py` | 从输入文档构建递归的 Reader 树结构 |

### 1.2 核心功能点

| 功能点  | 描述              | 关联方法                                                    |
| ---- | --------------- | ------------------------------------------------------- |
| 文档摄入 | 将文档转化为 Reader 树 | `ingest_document()`                                     |
| 节点构建 | 创建单个 Reader 节点  | `_build_node()`                                         |
| 递归分割 | 根据条件决定是否构建子节点   | `_build_children_if_needed()`                           |
| 分割决策 | 判断文本是否需要分割      | `_should_split()`                                       |
| 知识提取 | 通过 LLM 提取结构化知识  | `_make_reader()`                                        |
| 持久化  | 将节点保存到存储和向量索引   | `store.upsert_reader()`, `vector_index.upsert_reader()` |

***

## 2. 测试用例设计

### 2.1 测试环境与前置条件

| 组件                   | Mock/真实 | 说明          |
| -------------------- | ------- | ----------- |
| ReaderLLMService     | Mock    | 避免真实 LLM 调用 |
| KnowledgeStore       | Mock    | 避免真实数据库操作   |
| VectorIndex          | Mock    | 避免真实向量数据库操作 |
| ComplexityEstimator  | Mock/真实 | 根据测试需求选择    |
| SemanticTextSplitter | Mock/真实 | 根据测试需求选择    |

### 2.2 测试用例矩阵

#### 2.2.1 基本功能测试

| 测试 ID      | 测试名称           | 测试场景                    | 预期结果                                                      |
| ---------- | -------------- | ----------------------- | --------------------------------------------------------- |
| TC-RTB-001 | 空文档摄入          | 传入空字符串文本                | 不创建 Reader 节点                                             |
| TC-RTB-002 | 简单文档摄入         | 短文本（< max\_leaf\_chars） | 成功创建单个叶子节点 Reader                                         |
| TC-RTB-003 | 中等文档摄入         | 文本长度适中，复杂性低             | 成功创建单个叶子节点 Reader                                         |
| TC-RTB-004 | Reader ID 格式验证 | 验证生成的 reader\_id 格式     | 格式为 `reader::{document_id}::{depth}::{ordinal}::{random}` |
| TC-RTB-005 | 标题继承验证         | 验证子节点标题格式               | 子节点标题格式为 `{parent_title} / part-{index+1}`                |

#### 2.2.2 递归分割测试

| 测试 ID      | 测试名称    | 测试场景                | 预期结果                |
| ---------- | ------- | ------------------- | ------------------- |
| TC-RTB-011 | 单级分割    | 文档需要分割但子节点不需要       | 根节点有多个子节点，子节点均为叶子节点 |
| TC-RTB-012 | 多级分割    | 文档需要多级递归分割          | 创建多层 Reader 树结构     |
| TC-RTB-013 | 深度限制    | 达到 max\_depth 后不再分割 | 在最大深度处停止分割          |
| TC-RTB-014 | 分割后只有一段 | 分割后 chunks 长度 <= 1  | 不创建子节点，当前节点为叶子节点    |
| TC-RTB-015 | 刚好达到阈值  | 复杂性评分刚好等于阈值         | 不进行分割               |

#### 2.2.3 分割决策逻辑测试

| 测试 ID      | 测试名称   | 测试场景                                   | 预期结果         |
| ---------- | ------ | -------------------------------------- | ------------ |
| TC-RTB-021 | 深度超过限制 | depth >= max\_depth                    | 返回 False，不分割 |
| TC-RTB-022 | 复杂性未达标 | score < complexity\_threshold          | 返回 False，不分割 |
| TC-RTB-023 | 长度未达标  | len(text) <= max\_leaf\_chars          | 返回 False，不分割 |
| TC-RTB-024 | 所有条件满足 | score >= threshold AND len > max\_leaf | 返回 True，进行分割 |
| TC-RTB-025 | 空文本评估  | 传入空字符串                                 | 返回 False，不分割 |

#### 2.2.4 持久化测试

| 测试 ID      | 测试名称    | 测试场景        | 预期结果                                                         |
| ---------- | ------- | ----------- | ------------------------------------------------------------ |
| TC-RTB-031 | 节点存储验证  | 构建单个节点      | `store.upsert_reader` 被调用 1 次                                |
| TC-RTB-032 | 多节点存储验证 | 构建包含子节点的树   | `store.upsert_reader` 被调用 N 次（N=节点总数）                        |
| TC-RTB-033 | 向量索引验证  | 构建 Reader 树 | `vector_index.upsert_reader` 被调用 N 次                         |
| TC-RTB-034 | 文档删除验证  | 重复摄入同一文档    | `store.delete_document` 和 `vector_index.delete_document` 被调用 |
| TC-RTB-035 | 存储调用顺序  | 验证操作顺序      | delete → build → upsert                                      |

#### 2.2.5 边界条件测试

| 测试 ID      | 测试名称    | 测试场景                    | 预期结果                      |
| ---------- | ------- | ----------------------- | ------------------------- |
| TC-RTB-041 | 最大深度为0  | max\_depth=0            | 只创建根节点，不分割                |
| TC-RTB-042 | 超大文档    | 远超 max\_leaf\_chars 的文档 | 递归分割直到达到深度限制或满足条件         |
| TC-RTB-043 | 复杂性阈值为0 | complexity\_threshold=0 | 长度超过 max\_leaf\_chars 就分割 |
| TC-RTB-044 | 特殊字符文档  | 包含 emoji、特殊符号的文档        | 正常处理，不抛出异常                |
| TC-RTB-045 | 超长单一段落  | 无法按段落分割的长文本             | 使用句子分割作为后备                |

#### 2.2.6 依赖组件交互测试

| 测试 ID      | 测试名称       | 测试场景            | 预期结果                                             |
| ---------- | ---------- | --------------- | ------------------------------------------------ |
| TC-RTB-051 | LLM 知识提取调用 | 构建 Reader 节点    | `llm_service.extract_knowledge` 被调用              |
| TC-RTB-052 | LLM 能力问题生成 | 构建 Reader 节点    | `llm_service.build_capability_questions` 被调用     |
| TC-RTB-053 | 自定义复杂性评估器  | 使用自定义 estimator | 使用自定义评估器进行评分                                     |
| TC-RTB-054 | 自定义文本分割器   | 使用自定义 splitter  | 使用自定义分割器进行分割                                     |
| TC-RTB-055 | 默认组件使用     | 不传入可选参数         | 使用默认的 ComplexityEstimator 和 SemanticTextSplitter |

***

## 3. 测试数据设计

### 3.1 测试文档样本

| 样本 ID | 名称     | 内容特征                       | 用途                     |
| ----- | ------ | -------------------------- | ---------------------- |
| D1    | 空文档    | `""`                       | TC-RTB-001, TC-RTB-025 |
| D2    | 极短文档   | `"Hello World"` (11 chars) | TC-RTB-002             |
| D3    | 简单文档   | 约 500 字符的简单英文段落            | TC-RTB-003             |
| D4    | 复杂文档   | 约 2000 字符的技术文档，包含多段落       | TC-RTB-011, TC-RTB-024 |
| D5    | 超复杂文档  | 约 5000 字符的学术论文，高复杂性        | TC-RTB-012, TC-RTB-042 |
| D6    | 单一长段落  | 约 1500 字符的无分段文本            | TC-RTB-045             |
| D7    | 特殊字符文档 | 包含 emoji、中文、特殊符号的混合文本      | TC-RTB-044             |

### 3.2 配置参数组合

| 配置组合 | max\_depth | max\_leaf\_chars | complexity\_threshold | 用途     |
| ---- | ---------- | ---------------- | --------------------- | ------ |
| C1   | 4          | 900              | 1050.0                | 默认配置   |
| C2   | 0          | 900              | 1050.0                | 无递归    |
| C3   | 4          | 100              | 1050.0                | 严格长度限制 |
| C4   | 4          | 900              | 0.0                   | 无复杂性限制 |
| C5   | 1          | 900              | 1050.0                | 单级分割   |

***

## 4. 测试执行流程

### 4.1 通用测试步骤

```
1. 准备测试环境
   - 创建 Mock 对象（LLMService, KnowledgeStore, VectorIndex）
   - 配置 Mock 返回值

2. 初始化 ReaderTreeBuilder
   - 创建 IngestionConfig
   - 实例化 ReaderTreeBuilder

3. 执行被测方法
   - 调用 ingest_document() 或其他方法

4. 验证结果
   - 验证返回的 ReaderNode 结构正确
   - 验证 Mock 方法调用次数和参数
   - 验证树结构的正确性

5. 清理测试环境
   - 重置 Mock 状态
```

### 4.2 测试类结构设计

```python
class TestReaderTreeBuilder:
    """ReaderTreeBuilder 单元测试类"""
    
    # === 基本功能测试 ===
    def test_ingest_empty_document(self): ...          # TC-RTB-001
    def test_ingest_short_document(self): ...         # TC-RTB-002
    def test_ingest_medium_document(self): ...        # TC-RTB-003
    def test_reader_id_format(self): ...              # TC-RTB-004
    def test_child_title_inheritance(self): ...       # TC-RTB-005
    
    # === 递归分割测试 ===
    def test_single_level_split(self): ...            # TC-RTB-011
    def test_multi_level_split(self): ...             # TC-RTB-012
    def test_depth_limit_enforced(self): ...          # TC-RTB-013
    def test_no_split_when_single_chunk(self): ...    # TC-RTB-014
    def test_split_at_threshold(self): ...            # TC-RTB-015
    
    # === 分割决策逻辑测试 ===
    def test_should_split_depth_exceeded(self): ...   # TC-RTB-021
    def test_should_split_low_complexity(self): ...   # TC-RTB-022
    def test_should_split_short_text(self): ...       # TC-RTB-023
    def test_should_split_all_conditions_met(self): ... # TC-RTB-024
    def test_should_split_empty_text(self): ...       # TC-RTB-025
    
    # === 持久化测试 ===
    def test_single_node_persistence(self): ...       # TC-RTB-031
    def test_multi_node_persistence(self): ...        # TC-RTB-032
    def test_vector_index_upsert(self): ...           # TC-RTB-033
    def test_document_deletion_on_rewrite(self): ...  # TC-RTB-034
    def test_persistence_order(self): ...             # TC-RTB-035
    
    # === 边界条件测试 ===
    def test_max_depth_zero(self): ...                # TC-RTB-041
    def test_extremely_large_document(self): ...      # TC-RTB-042
    def test_zero_complexity_threshold(self): ...     # TC-RTB-043
    def test_special_characters_document(self): ...   # TC-RTB-044
    def test_long_single_paragraph(self): ...         # TC-RTB-045
    
    # === 依赖组件交互测试 ===
    def test_llm_extract_knowledge_called(self): ... # TC-RTB-051
    def test_llm_build_capability_called(self): ...  # TC-RTB-052
    def test_custom_complexity_estimator(self): ...  # TC-RTB-053
    def test_custom_text_splitter(self): ...         # TC-RTB-054
    def test_default_components_used(self): ...      # TC-RTB-055
```

***

## 5. 测试断言要点

### 5.1 ReaderNode 结构断言

| 断言项          | 验证内容                               |
| ------------ | ---------------------------------- |
| reader\_id   | 格式正确，包含 document\_id、depth、ordinal |
| document\_id | 与输入一致                              |
| title        | 正确继承和格式化                           |
| parent\_id   | 根节点为 None，子节点正确指向父节点               |
| depth        | 正确的层级深度                            |
| ordinal      | 正确的兄弟节点顺序                          |
| text         | 正确的文本内容                            |
| knowledge    | 非空，包含提取的知识                         |
| child\_ids   | 叶子节点为空列表，非叶子节点包含子节点 ID             |
| is\_leaf     | 根据 child\_ids 正确判断                 |

### 5.2 Mock 调用断言

| Mock 对象       | 方法                           | 断言内容                   |
| ------------- | ---------------------------- | ---------------------- |
| llm\_service  | extract\_knowledge           | 每个节点调用一次，参数正确          |
| llm\_service  | build\_capability\_questions | 每个节点调用一次，参数正确          |
| store         | delete\_document             | 摄入前调用，参数为 document\_id |
| store         | upsert\_reader               | 每个节点调用一次               |
| vector\_index | delete\_document             | 摄入前调用，参数为 document\_id |
| vector\_index | upsert\_reader               | 每个节点调用一次               |

***

## 6. 测试覆盖率目标

| 覆盖率类型 | 目标值   |
| ----- | ----- |
| 语句覆盖率 | ≥ 98% |
| 分支覆盖率 | ≥ 98% |
| 方法覆盖率 | 100%  |
| 类覆盖率  | 100%  |

***

## 7. 测试风险评估

| 风险等级 | 风险描述        | 关联测试用例                 | 缓解措施                  |
| ---- | ----------- | ---------------------- | --------------------- |
| 高    | LLM 服务调用失败  | 所有测试                   | 使用 Mock 对象隔离          |
| 高    | 递归深度过大导致栈溢出 | TC-RTB-012, TC-RTB-042 | 设置 max\_depth 限制，测试验证 |
| 中    | 文本分割结果不稳定   | TC-RTB-011, TC-RTB-045 | 使用确定性的分割器 Mock        |
| 中    | 复杂性评估结果不一致  | TC-RTB-022, TC-RTB-024 | 使用 Mock 控制评分结果        |
| 低    | UUID 生成冲突   | TC-RTB-004             | 验证格式而非具体值             |

