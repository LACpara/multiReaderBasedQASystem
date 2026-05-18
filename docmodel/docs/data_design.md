# DocModel 模块数据设计文档

## 1. 概述

本文档详细描述 DocModel 模块的数据结构设计、数据流转和数据持久化方案。核心设计原则是**字符偏移作为唯一坐标系**和**惰性视图**。

## 2. 核心数据结构

### 2.1 Source - 底层数据源

Source 是唯一持有完整文本数据的对象，所有视图都引用它。

```
┌─────────────────────────────────────────────────────┐
│                      Source                         │
├─────────────────────────────────────────────────────┤
│ source_id: str          # 唯一标识符                 │
│ text: str               # 完整文本内容               │
│ meta: Mapping[str, Any] # 元数据（不可变）            │
├─────────────────────────────────────────────────────┤
│ 不变量：                                             │
│ - 创建后不可修改（frozen=True）                       │
│ - source_id 在 Book 内唯一                           │
│ - text 长度 = meta["char_count"]                     │
└─────────────────────────────────────────────────────┘
```

**字段说明**：

| 字段         | 类型                 | 必需 | 说明                                                     |
| ---------- | ------------------ | -- | ------------------------------------------------------ |
| source\_id | str                | 是  | 在 Book 内的唯一标识，通常是文件名或序号                                |
| text       | str                | 是  | 完整的 UTF-8 文本内容                                         |
| meta       | Mapping\[str, Any] | 否  | 不可变字典，包含 filename, char\_count, sha256, order\_index 等 |

**元数据字段规范**：

```json
{
  "filename": "001_chapter1.txt",
  "char_count": 50000,
  "sha256": "a1b2c3...",
  "order_index": 0,
  "origin": {
    "format": "pdf",
    "page_range": [1, 24]
  }
}
```

### 2.2 Span - 原子坐标

Span 定义文本的精确位置，使用半开区间 `[start, end)`。

```
┌─────────────────────────────────────────────────────┐
│                       Span                          │
├─────────────────────────────────────────────────────┤
│ source_id: str   # 所属 Source 的 ID               │
│ start: int       # 起始字符偏移（闭区间）          │
│ end: int         # 结束字符偏移（开区间）          │
├─────────────────────────────────────────────────────┤
│ 不变量：                                             │
│ - 0 <= start <= end <= source.text 长度            │
│ - length = end - start                              │
│ - 半开区间保证拼接无歧义                            │
└─────────────────────────────────────────────────────┘
```

**坐标系示例**：

```
文本: "Hello World"
索引:  0123456789...

Span(source_id="ch1", start=0, end=5)   → "Hello"
Span(source_id="ch1", start=6, end=11)  → "World"
```

### 2.3 DocView - 文档视图

DocView 是唯一的文档抽象，支持所有操作。

```
┌─────────────────────────────────────────────────────┐
│                     DocView                         │
├─────────────────────────────────────────────────────┤
│ spans: Sequence[Span]        # 有序 Span 列表        │
│ sources: Mapping[str, Source]# 共享 Source 池       │
│ parent: Optional[DocView]    # 父视图引用            │
│ tags: Mapping[str, Any]      # 业务标签              │
├─────────────────────────────────────────────────────┤
│ 方法：                                               │
│ - slice(start, end) → DocView                       │
│ - split(splitter) → List[DocView]                   │
│ - search(query) → List[DocView]                     │
│ - iter(granularity) → Iterator[DocView]             │
│ - fold(granularity, init, step) → S                 │
│ - project(child) → List[Span]                       │
│ - text() → str                                      │
│ - length() → int                                    │
└─────────────────────────────────────────────────────┘
```

**关键设计点**：

1. **多 Span 支持**：一个视图可以跨越多个 Source
   ```
   Book 根视图: [Span(ch1, 0, 1000), Span(ch2, 0, 800), Span(ch3, 0, 1200)]
   ```
2. **本地坐标**：视图操作使用本地坐标（从 0 开始）
   ```
   视图文本: "第1章内容...第2章内容..."
   本地坐标:  0123456...
   ```
3. **惰性计算**：`slice` 不复制文本，只创建新 Span

### 2.4 Granularity - 迭代粒度

```python
class Granularity(Enum):
    CHAR = "char"           # 单字符
    WORD = "word"           # 单词
    SENTENCE = "sentence"   # 句子
    LINE = "line"           # 行
    PARAGRAPH = "paragraph" # 段落
    SOURCE = "source"       # 整个 Source
```

## 3. Book 协议数据格式

### 3.1 目录结构

```
my_book/
├── manifest.json          # 元信息清单
├── 001_chapter1.txt       # Source 文件
├── 002_chapter2.txt
├── 003_appendix.txt
└── assets/                # 可选资源目录
    └── images/
```

### 3.2 manifest.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["schema_version", "book_id", "sources"],
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+$"
    },
    "book_id": {
      "type": "string"
    },
    "title": {
      "type": "string"
    },
    "language": {
      "type": "string"
    },
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["source_id", "path", "order"],
        "properties": {
          "source_id": {"type": "string"},
          "path": {"type": "string"},
          "order": {"type": "integer"},
          "encoding": {"type": "string", "default": "utf-8"},
          "sha256": {"type": "string"},
          "meta": {"type": "object"}
        }
      }
    },
    "tags": {
      "type": "object"
    }
  }
}
```

### 3.3 示例 manifest.json

```json
{
  "schema_version": "1.0",
  "book_id": "shiji-2024",
  "title": "史记",
  "language": "zh",
  "sources": [
    {
      "source_id": "ch1",
      "path": "001_chapter1.txt",
      "order": 0,
      "encoding": "utf-8",
      "sha256": "abc123...",
      "meta": {
        "chapter_no": 1,
        "title": "本纪一",
        "origin": {"format": "pdf", "page_range": [1, 24]}
      }
    },
    {
      "source_id": "ch2",
      "path": "002_chapter2.txt",
      "order": 1,
      "encoding": "utf-8",
      "sha256": "def456...",
      "meta": {
        "chapter_no": 2,
        "title": "本纪二"
      }
    }
  ],
  "tags": {
    "publisher": "中华书局",
    "year": 1959
  }
}
```

## 4. 数据流转

### 4.1 摄入流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PDF/DOCX/MD │ ──→ │   Ingester   │ ──→ │  Book 目录    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ manifest.json│
                     │ + .txt files │
                     └──────────────┘
```

**数据转换规则**：

| 输入格式     | 切分策略          | 输出 Source 数量 |
| -------- | ------------- | ------------ |
| PDF      | 按书签/大纲或按页     | 1-N 个        |
| DOCX     | 按 Heading 样式  | 1-N 个        |
| Markdown | 按 H1 标题       | 1-N 个        |
| TXT      | 整体作为一个 Source | 1 个          |

### 4.2 加载流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Book 目录   │ ──→ │  Book.load() │ ──→ │  DocView     │
└──────────────┘     └──────────────┘     │  (根视图)    │
                           │              └──────────────┘
                           ▼
                    ┌──────────────┐
                    │ 解析 manifest│
                    │ 加载 Sources │
                    │ 创建 Spans   │
                    └──────────────┘
```

### 4.3 切分流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  DocView     │ ──→ │   Splitter   │ ──→ │ List[Span]   │
│  (父视图)     │     │  (计算区间)   │     │  (本地坐标)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ List[DocView]│
                                          │  (子视图)     │
                                          └──────────────┘
```

**坐标转换示例**：

```
父视图: Span(ch1, 100, 500)  # 本地坐标 0-400
Splitter 返回: [(0, 50), (50, 150), (150, 400)]
转换后子视图 Spans:
  - 子视图1: Span(ch1, 100, 150)
  - 子视图2: Span(ch1, 150, 250)
  - 子视图3: Span(ch1, 250, 500)
```

## 5. 数据持久化

### 5.1 书本目录持久化

Book 通过文件系统持久化：

```
保存流程:
Book.save(path) → 写入 manifest.json → 写入各 .txt 文件

加载流程:
Book.load(path) → 读取 manifest.json → 加载各 .txt 文件 → 构建 DocView
```

### 5.2 视图引用持久化

DocView 的位置信息可以序列化为 Span 列表：

```python
# 序列化
spans = view.to_book_spans()
data = [(s.source_id, s.start, s.end) for s in spans]

# 反序列化
spans = [Span(sid, start, end) for sid, start, end in data]
view = DocView(spans, sources)
```

### 5.3 援引数据结构

用于 QA 系统的援引信息：

```python
@dataclass
class Citation:
    spans: List[Span]           # 精确位置
    text: str                   # 文本内容
    context: str                # 上下文（扩展后）
    source_meta: Dict[str, Any] # 来源元信息
```

## 6. Splitter 数据结构

### 6.1 Splitter 协议

```python
class Splitter(Protocol):
    def __call__(self, text: str) -> List[Tuple[int, int]]:
        """输入文本，返回本地坐标区间列表"""
```

### 6.2 内置 Splitter 配置

**RegexSplitter**:

```python
@dataclass
class RegexSplitterConfig:
    pattern: str              # 正则表达式
    include_match: bool = True  # 是否包含匹配内容
```

**RecursiveSplitter**:

```python
@dataclass
class RecursiveSplitterConfig:
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", "。", ".", " "]
    )
    max_size: int = 500
    overlap: int = 50
```

**WindowSplitter**:

```python
@dataclass
class WindowSplitterConfig:
    size: int = 100
    stride: int = 50
```

## 7. 内存管理

### 7.1 内存布局

```
┌─────────────────────────────────────────────────────┐
│                      Book                           │
│  ┌─────────────────────────────────────────────┐    │
│  │              sources: Dict                  │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │ Source1 │ │ Source2 │ │ Source3 │        │    │
│  │  │ (text)  │ │ (text)  │ │ (text)  │        │    │
│  │  └─────────┘ └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │              root_view: DocView             │    │
│  │  spans: [Span(ch1,0,1000), Span(ch2,0,800)] │    │
│  │  (不持有文本，只持有引用)                       │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 7.2 内存优化策略

1. **共享 Source 池**：所有视图共享同一份文本数据
2. **惰性切片**：slice 操作不复制文本
3. **迭代器模式**：iter 返回迭代器，避免一次性物化

### 7.3 大文件处理（未来扩展）

对于超大文件（> 1GB），可使用 mmap：

```python
class LazySource:
    def __init__(self, path: Path):
        self._mmap = mmap.mmap(open(path, 'rb').fileno(), 0)
    
    def __getitem__(self, key: slice) -> str:
        return self._mmap[key].decode('utf-8')
```

## 8. 错误处理数据结构

### 8.1 异常层次

```
DocModelError (基类)
├── SourceNotFoundError      # Source 不存在
├── InvalidSpanError         # Span 坐标无效
├── BookLoadError            # Book 加载失败
├── IngesterError            # 摄入失败
└── SplitterError            # 切分失败
```

### 8.2 错误信息结构

```python
@dataclass
class ErrorContext:
    operation: str           # 操作类型
    source_id: Optional[str] # 相关 Source
    span: Optional[Span]     # 相关 Span
    message: str             # 错误消息
```

## 9. 性能考量

### 9.1 时间复杂度

| 操作          | 复杂度     | 说明             |
| ----------- | ------- | -------------- |
| text()      | O(n)    | n = 视图长度       |
| s**lice()** | O(1)    | 只创建 Span       |
| search()    | O(n\*m) | n=文本长度, m=模式长度 |
| split()     | O(n)    | 取决于 Splitter   |
| iter()      | O(1)    | 返回迭代器          |

### 9.2 空间复杂度

| 对象      | 复杂度  | 说明          |
| ------- | ---- | ----------- |
| Source  | O(n) | n = 文本长度    |
| Span    | O(1) | 固定大小        |
| DocView | O(k) | k = Span 数量 |

## 10. 数据完整性约束

### 10.1 Source 约束

- `source_id` 非空且唯一
- `text` 非空
- `meta["char_count"]` == `len(text)`

### 10.2 Span 约束

- `0 <= start <= end`
- `end <= len(sources[source_id].text)`
- `source_id` 必须存在于 sources 中

### 10.3 DocView 约束

- `spans` 有序且不重叠（可选约束）
- 所有 `spans[i].source_id` 存在于 `sources` 中
- `parent` 形成无环链

## 11. 版本兼容性

### 11.1 manifest 版本规则

- 主版本号变更：不兼容的结构变化
- 次版本号变更：向后兼容的新增字段

### 11.2 迁移策略

```python
def migrate_manifest(data: dict) -> dict:
    version = data.get("schema_version", "1.0")
    if version == "1.0":
        return data  # 当前版本
    # 未来版本的迁移逻辑
    ...
```

