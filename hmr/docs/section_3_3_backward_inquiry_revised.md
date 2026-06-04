# 3.3 节：阅读阶段的"部分回答"与逆向求知机制 - 修订版实现说明

## 1. 概述

本文档是对原有实现说明的修订，包含两部分重要更新：
1. 字段命名优化
2. 分支节点知识构建逻辑重构

## 2. 字段命名优化

### 2.1 修改内容

将 `ReaderNode` 中的字段名进行优化，更清晰地表达语义：

| 原字段名 | 新字段名 | 说明 |
|---------|---------|------|
| backward_inquiries | emitted_inquiries | 强调这是本 Reader **发出**的逆向请求 |
| received_inquiries | received_inquiries | 保持不变，本 Reader 收到的请求 |
| partial_answers | emitted_partial_answers | 强调这是本 Reader **发出**的部分回答 |
| complete_answers | received_complete_answers | 强调这是本 Reader **收到**的完整回答 |

### 2.2 修改后的 ReaderNode 结构

```python
@dataclass(slots=True)
class ReaderNode:
    # ... 现有字段保持不变 ...
    
    # 新增/重命名字段
    emitted_inquiries: list[BackwardInquiry] = field(default_factory=list)  # 发出的逆向请求
    received_inquiries: list[BackwardInquiry] = field(default_factory=list)  # 收到的逆向请求
    emitted_partial_answers: list[PartialAnswer] = field(default_factory=list)  # 发出的部分回答
    received_complete_answers: list[CompleteAnswer] = field(default_factory=list)  # 收到的完整回答
```

### 2.3 数据库表字段同步更新

相应的数据库表字段也需要同步更新，保持一致性。

---

## 3. 分支节点知识构建逻辑重构

### 3.1 问题分析

**现有实现的问题**：
目前 `_make_reader` 在构建分支节点时，仍然使用该节点分配到的全量文档内容来提取知识：

```python
# reader_builder.py:118
knowledge = self.llm_service.extract_knowledge(text, title=title)
```

这存在以下不合理之处：
1. **重复计算**：分支节点和子节点都处理相同的文本
2. **语义层次不清晰**：分支节点应该是子节点知识的**抽象和聚合**，而不是重复提取
3. **能力建模不准确**：分支节点的回答能力应该基于子节点的能力来估计

### 3.2 设计目标

对于不同类型的 Reader 节点，采用不同的知识构建策略：

| 节点类型 | 知识构建方式 | 能力构建方式 |
|---------|-------------|-------------|
| 叶节点（无子节点） | 直接从分配的文本提取 | 基于自身知识生成 |
| 分支节点（有子节点） | 聚合子节点的知识 | 基于子节点的能力估计 |

### 3.3 接口设计

#### 3.3.1 LLM 服务接口扩展

在 `ReaderLLMService` 中新增知识聚合和能力估计方法：

```python
class ReaderLLMService(ABC):
    # ... 现有方法保持不变 ...
    
    @abstractmethod
    def aggregate_children_knowledge(
        self,
        children_knowledge: list[ReaderKnowledge],
        *,
        title: str
    ) -> ReaderKnowledge:
        """聚合多个子节点的知识，构建更高层次的知识表示"""
        
    @abstractmethod
    def estimate_capability_from_children(
        self,
        children_capabilities: list[list[str]],  # 每个子节点的 capability_questions
        *,
        title: str
    ) -> list[str]:
        """基于子节点的能力，估计父节点的回答能力"""
```

#### 3.3.2 新增 Prompt 模板

- `promptTemplates/aggregate_knowledge.prompt` - 聚合子节点知识的提示词
- `promptTemplates/estimate_capability.prompt` - 估计父节点能力的提示词

### 3.4 详细修改方案

#### 3.4.1 修改 ReaderTreeBuilder._make_reader 方法

**文件**：`reader_builder.py`

**修改前**：
```python
def _make_reader(...):
    knowledge = self.llm_service.extract_knowledge(text, title=title)
    knowledge.capability_questions = self.llm_service.build_capability_questions(knowledge, title=title)
    # ...
```

**修改后**：
```python
def _make_reader(...):
    if not child_ids:  # 叶节点
        # 叶节点：直接从文本提取知识
        knowledge = self.llm_service.extract_knowledge(text, title=title)
        knowledge.capability_questions = self.llm_service.build_capability_questions(knowledge, title=title)
    else:  # 分支节点
        # 分支节点：聚合子节点的知识
        children = [self.store.get_reader(child_id) for child_id in child_ids if self.store.get_reader(child_id)]
        children_knowledge = [child.knowledge for child in children]
        
        # 聚合知识
        knowledge = self.llm_service.aggregate_children_knowledge(children_knowledge, title=title)
        
        # 基于子节点能力估计父节点能力
        children_capabilities = [child.knowledge.capability_questions for child in children]
        knowledge.capability_questions = self.llm_service.estimate_capability_from_children(
            children_capabilities, title=title
        )
        
        # 分支节点的 source_excerpt 可以是子节点的摘要集合
        knowledge.source_excerpt = "\n---\n".join([
            f"[{child.title}]\n{child.knowledge.summary}" 
            for child in children
        ])
    
    # ... 后续逆向求知处理保持不变 ...
```

#### 3.4.2 构建流程调整

为了确保分支节点构建时子节点已经可用，需要调整构建顺序：

**现有流程**：
```
_build_node
├─ _build_children_if_needed (获取 child_ids)
└─ _make_reader (使用 child_ids)
```

**问题**：`_build_children` 返回的只是 child_ids，但此时子节点可能还没有被持久化到 store 中，无法在 `_make_reader` 中读取。

**修改方案**：
重构构建流程，确保子节点完全构建并持久化后再构建父节点：

```python
def _build_node(...):
    if len(text) == 0: 
        return None
    
    reader_id = self._new_reader_id(document_id, depth, ordinal)
    logger.debug("Building reader id=%s depth=%s ordinal=%s", reader_id, depth, ordinal)
    
    # 先构建并持久化子节点
    child_nodes = self._build_children_if_needed(document_id, title, text, reader_id, depth)
    child_ids = [child.reader_id for child in child_nodes if child]
    
    # 再构建当前节点（此时子节点已在 store 中）
    node = self._make_reader(reader_id, document_id, title, parent_id, depth, ordinal, text, child_ids, child_nodes)
    
    # 持久化当前节点
    self.store.upsert_reader(node)
    self.vector_index.upsert_reader(node)
    
    return node

def _build_children_if_needed(...):
    # 返回实际的 ReaderNode 列表，而不仅仅是 child_ids
    # ...
    return child_nodes

def _make_reader(..., child_nodes: list[ReaderNode] | None = None):
    # 直接使用传入的 child_nodes，避免再次从 store 读取
    # ...
```

**必要性说明**：
这是确保分支节点能够正确获取子节点知识的关键修改。虽然这改变了原有的构建顺序，但：
1. 逻辑上更合理：父节点依赖子节点，应该子节点先完成
2. 不影响现有功能：叶节点的行为保持不变
3. 为逆向求知机制奠定了更好的基础

### 3.5 数据流转图

```
文档文本
    ↓
根节点（分支节点）
    ↓
[分裂判断]
    ↓ 是
构建子节点 1 → 持久化 → 子节点 2 → 持久化 → ...
    ↓
聚合子节点知识 → 构建父节点知识
    ↓
估计父节点能力（基于子节点）
    ↓
持久化根节点
    ↓
完成
```

---

## 4. 完整的修改阶段规划

### 阶段 1：数据结构和命名优化
- 修改 `domain.py` 中的字段名
- 更新相关的 DTO 和存储接口

### 阶段 2：接口和 Prompt 扩展
- 扩展 `ReaderLLMService` 接口
- 新增聚合知识和估计能力的 Prompt 模板
- 实现 PromptedReaderLLMService 中的新方法

### 阶段 3：存储接口扩展
- 更新 `KnowledgeStore` 接口
- 更新 SQLite 实现

### 阶段 4：ReaderTreeBuilder 重构
- 重构构建流程，确保子节点先完成
- 修改 `_make_reader`，区分叶节点和分支节点的处理逻辑

### 阶段 5：逆向求知机制集成
- 实现 BackwardInquiryCoordinator
- 集成到 ReaderTreeBuilder 中

### 阶段 6：测试和验证
- 单元测试
- 集成测试
- E2E 测试

---

## 5. 总结

本次修订包含两个重要改进：

1. **命名优化**：使用 `emitted_*` 和 `received_*` 前缀更清晰地表达数据流向
2. **分支节点知识构建重构**：
   - 叶节点：直接从文本提取知识
   - 分支节点：聚合子节点知识，基于子节点估计能力
   - 调整构建顺序，确保依赖关系正确

这两个改进相互配合，为逆向求知机制奠定了更坚实的基础，同时保持了系统的可扩展性和向后兼容性。
