# 3.3 节：阅读阶段的"部分回答"与逆向求知机制 - 详细实现说明文档

## 1. 概述

本文档详细描述了如何实现设计文档 3.3 节的"部分回答"与逆向求知机制。该机制允许 Reader 在阅读阶段发现信息缺口时，向上游 subReader 请求缺失信息，从而构建完整的知识表示。

## 2. 核心概念与设计目标

### 2.1 核心机制
- **信息缺口检测**：Reader 在提炼知识时识别出文本中引用的未定义概念、缺失的上下文信息
- **逆向求知**：下游 Reader 向上游 Reader 发送问题请求
- **部分回答**：上游 Reader 回答其能够回答的部分，并将剩余问题继续上传
- **知识整合**：答案沿原路径回传并逐步整合，最终补全下游 Reader 的知识缺口

### 2.2 设计目标
- 保持现有架构的稳定性和兼容性
- 最小化对现有代码的侵入性修改
- 提供可配置的开关控制该功能
- 支持可追踪的调试日志
- 确保系统性能不受显著影响

## 3. 接口设计

### 3.1 数据结构扩展

#### 3.1.1 新增通信数据结构

```python
# 逆向求知请求包
@dataclass(slots=True)
class BackwardInquiry:
    inquiry_id: str
    source_reader_id: str
    target_reader_id: str
    question: str
    depth: int  # 逆向传递深度
    created_at: str = field(default_factory=utc_now_iso)

# 部分回答包
@dataclass(slots=True)
class PartialAnswer:
    inquiry_id: str
    answering_reader_id: str
    answered_content: str  # 已回答的部分
    remaining_question: str | None  # 剩余未回答的问题
    confidence: float
    created_at: str = field(default_factory=utc_now_iso)

# 完整回答包（最终回传）
@dataclass(slots=True)
class CompleteAnswer:
    inquiry_id: str
    full_answer: str
    answering_chain: list[str]  # 参与回答的 Reader ID 链
    created_at: str = field(default_factory=utc_now_iso)
```

#### 3.1.2 ReaderNode 扩展

在现有 `ReaderNode` 中添加通信通道和逆向求知记录：

```python
@dataclass(slots=True)
class ReaderNode:
    # ... 现有字段保持不变 ...
    
    # 新增字段
    backward_inquiries: list[BackwardInquiry] = field(default_factory=list)  # 发出的逆向请求
    received_inquiries: list[BackwardInquiry] = field(default_factory=list)  # 收到的逆向请求
    partial_answers: list[PartialAnswer] = field(default_factory=list)  # 发出的部分回答
    complete_answers: list[CompleteAnswer] = field(default_factory=list)  # 收到的完整回答
```

### 3.2 LLM 服务接口扩展

在 `ReaderLLMService` 中新增两个核心方法：

```python
class ReaderLLMService(ABC):
    # ... 现有方法保持不变 ...
    
    @abstractmethod
    def detect_information_gaps(
        self,
        text: str,
        knowledge: ReaderKnowledge,
        *,
        title: str
    ) -> list[str]:
        """检测文本中的信息缺口，返回需要向上游询问的问题列表"""
        
    @abstractmethod
    def integrate_knowledge(
        self,
        original_knowledge: ReaderKnowledge,
        complete_answers: list[CompleteAnswer],
        *,
        title: str
    ) -> ReaderKnowledge:
        """将获取的完整答案整合到原有知识中"""
        
    @abstractmethod
    def answer_backward_inquiry(
        self,
        knowledge: ReaderKnowledge,
        question: str,
        *,
        reader_id: str,
        title: str
    ) -> tuple[str, str | None, float]:
        """回答逆向求知问题，返回(已回答内容, 剩余问题, 置信度)"""
```

### 3.3 协调器接口

新增 `BackwardInquiryCoordinator` 类负责协调整个逆向求知流程：

```python
class BackwardInquiryCoordinator:
    def __init__(
        self,
        llm_service: ReaderLLMService,
        store: KnowledgeStore,
        config: BackwardInquiryConfig
    ):
        self.llm_service = llm_service
        self.store = store
        self.config = config
    
    def process_reader_with_backward_inquiry(
        self,
        reader: ReaderNode,
        all_siblings: list[ReaderNode]
    ) -> ReaderNode:
        """处理单个 Reader，包括检测缺口、发起逆向请求、整合答案"""
    
    def coordinate_inquiry_flow(
        self,
        inquiry: BackwardInquiry,
        upstream_readers: list[ReaderNode]
    ) -> CompleteAnswer:
        """协调逆向求知的完整流程"""
```

## 4. 数据设计

### 4.1 存储扩展

#### 4.1.1 新增数据库表

在 `sqlite_store.py` 中新增以下表：

```sql
-- 逆向求知请求表
CREATE TABLE IF NOT EXISTS backward_inquiries (
    inquiry_id TEXT PRIMARY KEY,
    source_reader_id TEXT NOT NULL,
    target_reader_id TEXT NOT NULL,
    question TEXT NOT NULL,
    depth INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_reader_id) REFERENCES readers(reader_id),
    FOREIGN KEY (target_reader_id) REFERENCES readers(reader_id)
);

-- 部分回答表
CREATE TABLE IF NOT EXISTS partial_answers (
    answer_id TEXT PRIMARY KEY,
    inquiry_id TEXT NOT NULL,
    answering_reader_id TEXT NOT NULL,
    answered_content TEXT NOT NULL,
    remaining_question TEXT,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (inquiry_id) REFERENCES backward_inquiries(inquiry_id),
    FOREIGN KEY (answering_reader_id) REFERENCES readers(reader_id)
);

-- 完整回答表
CREATE TABLE IF NOT EXISTS complete_answers (
    answer_id TEXT PRIMARY KEY,
    inquiry_id TEXT NOT NULL,
    full_answer TEXT NOT NULL,
    answering_chain TEXT NOT NULL,  -- JSON 数组
    created_at TEXT NOT NULL,
    FOREIGN KEY (inquiry_id) REFERENCES backward_inquiries(inquiry_id)
);
```

#### 4.1.2 KnowledgeStore 接口扩展

```python
class KnowledgeStore(ABC):
    # ... 现有方法保持不变 ...
    
    @abstractmethod
    def upsert_backward_inquiry(self, inquiry: BackwardInquiry) -> None:
        """保存逆向求知请求"""
        
    @abstractmethod
    def upsert_partial_answer(self, answer: PartialAnswer) -> None:
        """保存部分回答"""
        
    @abstractmethod
    def upsert_complete_answer(self, answer: CompleteAnswer) -> None:
        """保存完整回答"""
        
    @abstractmethod
    def get_inquiry_answers(self, inquiry_id: str) -> tuple[list[PartialAnswer], CompleteAnswer | None]:
        """获取某个请求的所有回答"""
        
    @abstractmethod
    def get_upstream_readers(self, reader_id: str) -> list[ReaderNode]:
        """获取某个 Reader 的上游 Readers（按顺序，从近到远）"""
```

### 4.2 配置扩展

在 `config.py` 中新增逆向求知配置：

```python
@dataclass(slots=True)
class BackwardInquiryConfig:
    """逆向求知机制的配置"""
    
    enabled: bool = True  # 是否启用该功能
    max_inquiry_depth: int = 3  # 最大逆向传递深度
    max_questions_per_reader: int = 5  # 每个 Reader 最多发起的问题数
    min_confidence_threshold: float = 0.3  # 回答的最小置信度
    max_retries_per_inquiry: int = 2  # 每个请求的最大重试次数

@dataclass(slots=True)
class IngestionConfig:
    # ... 现有字段保持不变 ...
    backward_inquiry: BackwardInquiryConfig = field(default_factory=BackwardInquiryConfig)
```

## 5. 与现有实现的兼容性设计

### 5.1 兼容性原则

1. **向后兼容**：现有代码无需修改即可继续运行
2. **可配置开关**：通过配置项控制是否启用该功能
3. **渐进式集成**：功能可以分阶段启用和测试
4. **最小侵入性**：仅在必要的地方进行修改

### 5.2 兼容实现方案

#### 5.2.1 ReaderTreeBuilder 的扩展

**现有代码位置**：`reader_builder.py:63-132`

**修改方案**：
在 `_make_reader` 方法中，在知识提炼后添加可选的逆向求知处理步骤：

```python
def _make_reader(...):
    knowledge = self.llm_service.extract_knowledge(text, title=title)
    knowledge.capability_questions = self.llm_service.build_capability_questions(knowledge, title=title)
    
    # 新增：逆向求知处理（仅在配置启用时执行）
    if self.config.backward_inquiry.enabled and parent_id is not None:
        knowledge = self._process_backward_inquiry(reader_id, knowledge, parent_id, title)
    
    return ReaderNode(...)
```

**必要性说明**：这是核心集成点，必须在此处插入逆向求知流程。但通过配置开关确保了默认行为不变。

#### 5.2.2 构建子节点的顺序调整

**现有代码位置**：`reader_builder.py:72-98`

**现有问题**：当前子节点是并行构建的，但逆向求知需要按顺序从上游到下游构建，以便下游可以访问上游已构建的知识。

**修改方案**：
添加一个新的构建策略，支持顺序构建：

```python
def _build_children_if_needed(...):
    if not self._should_split(text, depth): 
        return []
    
    chunks = [...]
    
    if self.config.backward_inquiry.enabled:
        # 启用逆向求知时，按顺序从前往后构建子节点
        return self._build_children_sequential(...)
    else:
        # 原有行为保持不变
        return self._build_children_parallel(...)
```

**必要性说明**：这是实现逆向求知的关键修改，确保上游 Reader 先于下游 Reader 完成知识提炼。但通过配置开关保持了原有行为的兼容性。

## 6. 详细修改方案

### 6.1 第一阶段：数据结构和接口扩展（非侵入性）

**文件修改清单**：
1. `domain.py` - 新增数据类
2. `dto.py` - 更新或替换为新的数据结构
3. `llm/base.py` - 新增抽象方法
4. `config.py` - 新增配置类
5. `storage/base.py` - 新增抽象方法

**修改说明**：
- 这些都是纯粹的扩展，不修改现有代码逻辑
- 所有新增的方法都有默认实现或通过配置控制
- 现有功能完全不受影响

### 6.2 第二阶段：Prompt 模板新增

**新增文件**：
- `promptTemplates/detect_gaps.prompt` - 检测信息缺口的提示词
- `promptTemplates/answer_backward.prompt` - 回答逆向问题的提示词
- `promptTemplates/integrate_knowledge.prompt` - 整合知识的提示词

**修改说明**：
- 纯新增文件，无侵入性

### 6.3 第三阶段：LLM 服务实现

**修改文件**：
- `llm/prompted_service.py` - 实现新增的 LLM 服务方法
- `llm/heuristic_service.py` - 实现启发式版本（用于测试）

**修改说明**：
- 仅新增方法，不修改现有方法
- 保持向后兼容

### 6.4 第四阶段：存储实现

**修改文件**：
- `storage/sqlite_store.py` - 实现新增的存储方法和表结构

**修改说明**：
- 表创建使用 `IF NOT EXISTS`，确保安全
- 新增方法不影响现有方法

### 6.5 第五阶段：核心协调器和 ReaderTreeBuilder 修改

**新增文件**：
- `backward_inquiry_coordinator.py` - 新增协调器类

**修改文件**：
- `reader_builder.py` - 集成逆向求知流程

**修改说明**：
- 这是唯一的侵入性修改
- 通过配置开关默认关闭，确保安全
- 提供清晰的日志记录，便于调试

## 7. 执行流程图

```
文档摄入
    ↓
构建根 Reader
    ↓
判断是否需要分裂
    ↓ 是
分裂为多个块
    ↓
[启用逆向求知?] → 否 → 原有并行构建
    ↓ 是
按顺序从前往后构建子 Reader
    ↓
对于每个子 Reader:
    ├─ 提取知识
    ├─ 检测信息缺口
    ├─ 如有缺口，创建 BackwardInquiry
    ├─ 向上游 Reader 传递（按顺序）
    │   ├─ 上游 Reader 尝试回答
    │   ├─ 如能部分回答，返回 PartialAnswer
    │   ├─ 如不能，继续传递给更上游
    │   └─ 最终返回 CompleteAnswer
    ├─ 整合答案到知识中
    └─ 保存 Reader
    ↓
完成文档摄入
```

## 8. 测试策略

### 8.1 单元测试
- 测试信息缺口检测
- 测试部分回答逻辑
- 测试知识整合

### 8.2 集成测试
- 测试完整的逆向求知流程
- 测试与现有流程的兼容性
- 测试配置开关功能

### 8.3 E2E 测试
- 使用设计文档中的"降压药 X-200"示例进行端到端测试
- 验证知识完整性是否得到提升

## 9. 风险与缓解措施

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 逆向求知增加摄入时间 | 中 | 中 | 配置开关、异步处理选项 |
| LLM 错误导致问题生成不准确 | 高 | 中 | 启发式后备方案、置信度过滤 |
| 存储结构变更导致兼容性问题 | 低 | 低 | 迁移脚本、版本控制 |
| 无限递归循环 | 高 | 低 | 最大深度限制、循环检测 |

## 10. 总结

本实现方案通过以下设计确保了与现有系统的和谐集成：

1. **可配置开关**：功能默认可关闭，不影响现有流程
2. **最小侵入性**：仅在关键节点进行可选修改
3. **渐进式集成**：可以分阶段实施和测试
4. **完整的可观测性**：详细的日志和存储记录

待审核通过后，将按照上述阶段计划逐步实施代码编写。
