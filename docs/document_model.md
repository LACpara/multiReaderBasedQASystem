# 文档模型设计方案：面向"书本"级输入的统一抽象层

## 0. 摘要与设计原则

你当前的系统接收"书本"量级的输入，提取知识入库，对外提供 QA 与援引检索。瓶颈是**输入侧只支持单 txt**，下游又有**递归分块层**对内部结构有强依赖。本方案提出一个**单层抽象 + 三个 Mixin 接口**的极简文档模型：以"惰性视图 + 字符偏移坐标系"为核心，把 `split / search / slice / iter / fold` 统一为视图上的纯函数操作；异构格式（pdf/docx/pptx）通过**显式的"摄入器"管道**归一到一个轻量级"书本协议"（一个目录 + manifest.json + 一组 .txt），与文档模型解耦。

设计原则按优先级：

1. **一个抽象，不多不少**：不区分 `Document / Chunk / Node / Span`，全部是 `DocView`。子视图与父视图能力完全相同，递归分块层不需要任何特殊代码路径。
2. **字符偏移作为唯一坐标系**：所有切片、搜索、援引都用 `(source_id, start, end)` 三元组定位，与具体存储介质无关。这是 LangChain `add_start_index`、Haystack `offsets_in_document`、Unstructured 元素 ID 的最大公约数（见 §8 调研附录）。
3. **惰性计算**：`slice` / `split` 不复制底层字节，只返回新的偏移区间和回引；只有显式 `text()` 时才物化。这是支持"按字符迭代地跑 n-gram 熵函数"且不爆内存的前提。
4. **Fold 是计算函数的统一入口**：用户的"传入计算函数，按字符迭代"需求，本质就是 `reduce / fold`。把它做成一等公民比为每种统计单独开 API 强 N 倍。
5. **摄入与模型彻底解耦**：异构格式只负责"产出书本目录"，模型代码完全不感知 pdf/docx 的存在。这一刀切干净是后续维护成本最低的方案。

不做的事（避免过度设计）：

- 不做嵌入存储、不做向量索引——这是知识库层的事。
- 不做 LangChain 风格的 Document 元数据洪流；只保留 `source_id / offsets / tags`。
- 不引入 NodeRelationship、parent/child 双向链接这种"图状文档"——子视图通过 `parent` 单向回引即可，足够支持递归分块的所有合理需求。
- 不在文档层做语义切分（embedding-based）。语义切分是分块策略的事，传 `Splitter` 进来即可，文档模型只提供"按模式切"这一能力。

---

## 1. 核心数据模型

### 1.1 Source：底层数据源（不可变）

```python
@dataclass(frozen=True)
class Source:
    """
    一份原始字符流。'书本'的一个 .txt 文件对应一个 Source。
    一本书是一组 Source 的有序拼接（见 §3 Book Protocol）。
    """
    source_id: str          # 在 Book 内唯一，通常是相对路径或顺序号
    text: str               # 完整文本（UTF-8）。也可以是 mmap / lazy-load 的代理对象，见 §6
    meta: Mapping[str, Any] # 不可变 dict：filename, char_count, sha256, order_index
```

`Source` 是**唯一持有完整字节的对象**，所有视图都只持有引用。

### 1.2 Span：原子坐标

```python
@dataclass(frozen=True)
class Span:
    source_id: str
    start: int   # 字符偏移，闭区间起点
    end: int     # 字符偏移，开区间终点；半开区间 [start, end)
```

跨 Source 的视图由**一组 Span** 表示（见下）。坚持半开区间是为了让 `span.length == end - start` 和拼接 `Span(a,b) + Span(b,c) = Span(a,c)` 无歧义。

### 1.3 DocView：唯一的文档抽象

```python
class DocView:
    """
    一个文档视图。'整本书'、'第3章'、'某段落'、'某句话' 全部是 DocView。
    视图本身不持有文本，只持有对 Source 的引用 + 一组有序 Span。
    """
    def __init__(
        self,
        spans: Sequence[Span],
        sources: Mapping[str, Source],
        parent: Optional["DocView"] = None,
        tags: Mapping[str, Any] = MappingProxyType({}),
    ): ...
```

关键属性：

- `spans`：**有序**的 Span 列表。同一视图可以跨多个 Source（例如"前三章"），但 spans 之间通常不重叠（重叠由调用方负责，模型不强制）。
- `sources`：共享的 Source 池。所有子视图共享同一个池——无论怎么切，底层字节只有一份。
- `parent`：单向回引父视图。`None` 表示是 Book 根视图。这一个引用足够支撑"援引上下文"（向上爬到根，沿途收集 tags），不需要双向。
- `tags`：业务标签（章节号、扫描置信度、来源页码、是否包含表格 …）。**不可变**。需要修改时返回新视图。

核心不变量：**任意 DocView 都和 Book 根视图能力完全相同**。这是后续递归分块层零特殊代码的基础。

---

## 2. 接口设计

总共 **6 类操作**，每类都是 DocView 上的方法，全部返回新的 DocView（或视图序列、或纯标量结果）。原始需求里的 `split / search / slice` 是其中 3 类；剩下 3 类（`iter / fold / project`）是我深入分析后认为**必须补上**才能闭环的最小集。

### 2.1 slice — 按坐标取子视图

```python
def slice(self, start: int, end: int) -> "DocView": ...
def slice_by_span(self, span: Span) -> "DocView": ...
```

- `start / end` 是**视图本地坐标**（视图首字符为 0），不是 Source 坐标。这一点很重要：用户通常不知道、也不该关心底层 Source 拼接细节。模型内部负责把本地坐标翻译成一组 Span。
- 跨 Source 边界的切片自动产生多 Span 视图。

### 2.2 split — 按模式切分（这是递归分块层的接入点）

```python
def split(
    self,
    splitter: Splitter,
) -> List["DocView"]: ...
```

`Splitter` 是一个**协议**，不是一个类层次。原因：你下游有递归分块、未来可能加语义分块、token-based 分块、按章节标题切——它们形态各异，没有公共父类，但有公共契约：

```python
class Splitter(Protocol):
    def __call__(self, text: str) -> List[Tuple[int, int]]:
        """
        输入视图的完整文本，输出一组 (start, end) 字符区间。
        区间允许重叠（支持 chunk_overlap），允许不覆盖全文（支持过滤）。
        """
```

`split` 拿到的 `(start, end)` 是**视图本地坐标**，由 DocView 自己翻译回全局 Span 并构造子视图，自动设 `parent=self`。

内置 5 个开箱即用的 Splitter（覆盖 95% 场景，借鉴 LangChain `RecursiveCharacterTextSplitter` + Unstructured 的经验，详见 §8）：

| Splitter | 用途 | 关键参数 |
|---|---|---|
| `RegexSplitter(pattern)` | 按正则切（章节标题、空行段落） | pattern |
| `RecursiveSplitter(seps, max_size, overlap)` | LangChain 同名工具的等价物，递归优先大粒度分隔符 | seps=["\n\n","\n","。","."," "], max_size, overlap |
| `WindowSplitter(size, stride)` | 字符级滑窗，配合 `fold` 做迭代统计 | size, stride |
| `SentenceSplitter(lang)` | 按 PySBD/blingfire 切句子 | lang |
| `TagSplitter(tag_key)` | 按已有 tag 边界切（例如按目录预设章节切） | tag_key |

**为什么不内置语义分块（embedding-based）？** 因为它依赖外部模型，把强依赖塞进文档层会污染抽象。需要的话用户实现 `Splitter` 协议 30 行代码搞定（见 §7 示例）。

### 2.3 search — 文本检索

```python
def search(
    self,
    query: Union[str, re.Pattern, Callable[[str], Iterable[Tuple[int,int]]]],
    *,
    overlapping: bool = False,
    limit: Optional[int] = None,
) -> List["DocView"]: ...
```

三种 query 形态覆盖所有需求：

- `str`：字面量子串匹配（最常用，简单情况）
- `re.Pattern`：正则
- `Callable`：用户自定义匹配器，返回 `(start, end)` 序列——给未来"向量召回"、"BM25 召回"等留好接口

返回**子 DocView 列表**，而不是裸 `(start, end)` 对。这样下游可以继续对每个命中片段调 `slice / split / fold`，调用风格统一。

### 2.4 iter — 按粒度迭代

```python
def iter(self, granularity: Granularity) -> Iterator["DocView"]: ...
```

`Granularity` 是枚举：`CHAR | WORD | SENTENCE | LINE | PARAGRAPH | SOURCE`。本质是 `split(对应Splitter)` 的惰性版本——返回迭代器而不是列表，避免大书一次性物化所有句子。

> 为什么单独开 `iter` 而不是让用户调 `split`？两点：(1) `iter` 是惰性的，对一本 500 万字的书按字符迭代不会爆内存；(2) `iter(CHAR)` 是 fold 的天然搭档，语义清晰。

### 2.5 fold — 携带用户计算函数迭代（这是你"n-gram 熵函数"需求的归宿）

```python
def fold(
    self,
    granularity: Granularity,
    init: S,
    step: Callable[[S, "DocView"], S],
    *,
    progress: bool = False,
) -> S: ...
```

语义上等价于 `functools.reduce`，但接受的是 DocView 序列。你说的"传入统计 n-gram 词汇频率并计算信息熵的函数，按字符迭代地执行"——直接是：

```python
def step(state, ch_view):
    state.update_ngram(ch_view.text())   # 累计 n-gram 频次
    return state

final_state = book.fold(Granularity.CHAR, init=NGramState(n=3), step=step)
entropy = final_state.entropy()
```

为什么是 fold 而不是 `map + reduce` 两步？因为 n-gram 这种**有状态滑窗统计**天然需要把状态串起来；map 出独立结果再 reduce 不能优雅表达"累积"。fold 是这种场景的标准答案。

**进阶**：`fold` 内部可以加 `parallel=True` 走 `functools.reduce` 风格的二叉归并（要求 step 满足结合律时），把熵这种 monoid 类统计并行起来。这是可选功能，第一版可以不做。

### 2.6 project — 跨视图坐标映射（援引功能的核心）

```python
def project(self, child: "DocView") -> List[Span]: ...
def to_book_spans(self) -> List[Span]: ...
def excerpt_with_context(self, context_chars: int = 200) -> "DocView": ...
```

- `project(child)`：把子视图的位置映射回**当前视图的本地坐标**。用于"在第3章里这段话出现在哪里"。
- `to_book_spans()`：返回视图覆盖的全局 Span 列表（用于持久化、援引）。
- `excerpt_with_context(n)`：向左右各扩 n 个字符，返回带上下文的新视图。**这是 QA 援引功能的杀手锏**——召回片段后调一下就有了可读上下文。

为什么这一组叫 `project`？因为它们本质都是坐标系之间的投影。把"投影"作为一等概念，比散在各处的 helper 函数有结构得多。

---

## 3. 书本协议（Book Protocol）

异构输入归一到这个协议；下游任何代码不再关心 pdf / docx。

### 3.1 目录结构

```
my_book/
├── manifest.json          # 必需：元信息与文件顺序
├── 001_chapter1.txt       # 一个文件 = 一个 Source
├── 002_chapter2.txt
├── 003_appendix.txt
└── assets/                # 可选：原始扫描图、转换中间产物（不被模型读取）
    └── ...
```

### 3.2 manifest.json 字段

```json
{
  "schema_version": "1.0",
  "book_id": "shiji-2024-edition",
  "title": "史记",
  "language": "zh",
  "sources": [
    {
      "source_id": "001_chapter1",
      "path": "001_chapter1.txt",
      "order": 0,
      "encoding": "utf-8",
      "sha256": "ab12...",
      "meta": {
        "chapter_no": 1,
        "title": "本纪一",
        "origin": {"format": "pdf", "page_range": [1, 24]}
      }
    },
    ...
  ],
  "tags": {
    "publisher": "中华书局",
    "year": 1959
  }
}
```

设计说明：

- **是否需要这个协议？** 强烈推荐做。理由：(1) 顺序信息（章节顺序）必须显式记录，靠文件名排序脆且不可靠；(2) `origin` 字段保留转换前的来源信息，QA 时可以告诉用户"出自原 PDF 第 X 页"——这是产品差异化点；(3) `sha256` 让缓存与增量更新成为可能。
- **schema 版本号**：从一开始就放上，未来加字段不会破坏旧书。
- **保持平**：不要嵌套深结构。复杂元信息塞 `meta` 字段，模型层不解释，原样透传。

### 3.3 摄入器（Ingester）：异构格式 → 书本目录

```python
class Ingester(Protocol):
    def ingest(self, input_path: Path, output_dir: Path) -> Path:
        """读入异构文件，输出标准 book 目录路径"""
```

第一版实现 3 个就够覆盖 90% 场景：

- `PdfIngester`：基于 PyMuPDF 或 pdfplumber。按页或按章节产生 .txt（推荐按 PDF 书签/大纲切，没书签就退化为按页）。
- `DocxIngester`：基于 python-docx。按 `Heading 1` 切，没标题层级就整本一个 source。
- `MarkdownIngester`：按 `# H1` 切。

**关键决策**：摄入器**只做"格式转换 + 切分到 source 粒度"**，不做内容清洗、不做去页眉页脚——那是数据预处理的事，硬塞进摄入器会让它变成下水道。需要清洗的话，做一个 `BookTransform` 链（输入 book 目录、输出 book 目录），按需调用，与摄入器解耦。

---

## 4. 整体形态

```
┌─────────────────────────────────────────────────────────┐
│  异构输入：pdf / docx / pptx / md / 单 txt / 文件夹     │
└────────────────────┬────────────────────────────────────┘
                     ▼
            ┌────────────────────┐
            │     Ingester       │  ← PdfIngester / DocxIngester / ...
            └────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│   书本目录（manifest.json + 一组 .txt）                  │
└────────────────────┬────────────────────────────────────┘
                     ▼
            ┌────────────────────┐
            │   Book.load(path)  │
            └────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│   DocView（根视图）                                      │
│   ─────────────────────────────────────────────────     │
│   slice / split / search / iter / fold / project        │
│        │       │                                         │
│        └───────┴── 返回子 DocView，能力完全相同 ─────┐ │
│                                                       │ │
│   ◄───────────── 递归分块层在这里挂钩 ◄──────────────┘ │
│   ◄───────────── 知识抽取在这里挂钩                      │
│   ◄───────────── 援引检索拿 to_book_spans() 持久化       │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 模块/包结构

```
docmodel/
├── __init__.py            # 暴露 Book, DocView, Span, Source, Granularity
├── core.py                # Source, Span, DocView 数据类与核心方法
├── book.py                # Book.load / Book.save，处理 manifest.json
├── splitters.py           # 5 个内置 Splitter
├── ingest/
│   ├── __init__.py        # Ingester 协议
│   ├── pdf.py             # PdfIngester
│   ├── docx.py
│   └── markdown.py
├── transforms.py          # BookTransform 链：清洗、去重、合并等
└── tests/
    └── ...
```

**总代码量预估**：核心 `core.py` 约 400 行，splitters 约 200 行，book.py + ingest 共约 500 行。整个包 1500 行以内可以做到完备。这就是为什么我反对引入 LangChain / LlamaIndex 全家桶——你的需求很聚焦，自己写一个 1500 行的层比依赖 50000 行的框架更可控。

---

## 6. 几个值得说清楚的设计取舍

### 6.1 为什么是字符偏移而不是 token 偏移？

- **字符是稳定的**：tokenizer 一换（GPT-4 → Claude → 国产模型），token 边界全乱。字符不会。
- 你的下游需求是知识抽取、援引、QA——这些都对"在原文什么位置"敏感，对 token 边界不敏感。
- 真到了要喂 LLM 的环节再做 token 转换（每个 chunk 的 `tokenize(view.text())`）也不晚，且那是消费侧的事。

### 6.2 为什么 split 返回 List 而 iter 返回 Iterator？

- `split` 通常用于"把章节切成段落"这种你需要随机访问、需要知道总数的场景。
- `iter` 用于"按字符流过整本书做统计"这种内存敏感场景。
- 强行二选一会逼用户做不必要的转换。两个都给，命名上拉开距离。

### 6.3 Source 一定要在内存里整个加载吗？

**对小书（< 1GB）**：直接 `text: str` 全加载最简单，性能足够。
**对大书**：把 `Source.text` 换成一个支持切片的 lazy 代理对象（背后是 `mmap`），只要它实现 `__getitem__(slice)` 返回 `str`，DocView 的所有代码不用改。这是把"内存策略"留作正交维度的好处。

第一版用全加载就行，等真的遇到 5GB 单文件再上 mmap。**不要预优化**。

### 6.4 关于"开放给计算函数"的安全考量

`fold` 接收用户函数，函数里能干任何事——这是 Python 风格的"我们都是成年人"。如果你的系统会跑别人写的 step 函数，要考虑沙箱（resource limits、超时）。这是部署层的事，模型层不管。

### 6.5 子视图的能力对等是怎么实现的？

`DocView` 上所有方法都不依赖"我是不是根视图"的状态——`split` 一个子视图返回的还是普通 DocView，挂在新 parent 下。**递归分块层**只要写一个"对当前视图调 split，对结果再调 split"的循环就行，模型层零特殊代码。例如：

```python
def recursive_chunk(view: DocView, depth: int, splitters: List[Splitter]) -> List[DocView]:
    if depth == 0:
        return [view]
    children = view.split(splitters[0])
    return [leaf
            for child in children
            for leaf in recursive_chunk(child, depth-1, splitters[1:])]
```

---

## 7. 端到端示例

```python
from docmodel import Book, Granularity
from docmodel.splitters import RecursiveSplitter, RegexSplitter

# 1. 一次性把 PDF 转成书本目录（只做一次）
from docmodel.ingest import PdfIngester
PdfIngester().ingest(Path("史记.pdf"), Path("books/shiji/"))

# 2. 加载
book = Book.load("books/shiji/")

# 3. 按章节大标题切（搜索"第X章"）
chapters = book.split(RegexSplitter(r"^第[一二三四五六七八九十百]+章"))

# 4. 在某章里按段落进一步切，递归分块层在这里挂钩
para_splitter = RecursiveSplitter(
    seps=["\n\n", "。", " "], max_size=500, overlap=50
)
para_chunks = chapters[2].split(para_splitter)

# 5. 检索 + 援引上下文
hits = book.search("项羽本纪")
for hit in hits[:5]:
    print(hit.excerpt_with_context(100).text())

# 6. 计算函数：按字符迭代统计 3-gram 熵
class NGramState:
    def __init__(self, n): self.n = n; self.counts = {}; self.buf = ""
    def update(self, ch):
        self.buf = (self.buf + ch)[-self.n:]
        if len(self.buf) == self.n:
            self.counts[self.buf] = self.counts.get(self.buf, 0) + 1
    def entropy(self):
        import math
        total = sum(self.counts.values())
        return -sum((c/total) * math.log2(c/total) for c in self.counts.values())

def step(state, view):
    state.update(view.text())
    return state

state = book.fold(Granularity.CHAR, NGramState(n=3), step)
print(f"3-gram entropy: {state.entropy():.4f} bits")

# 7. 在子视图上同样能做（能力对等）
state_ch2 = chapters[2].fold(Granularity.CHAR, NGramState(n=3), step)

# 8. 用户自定义 Splitter（语义分块示例，仅 30 行）
class SemanticSplitter:
    def __init__(self, embedder, threshold=0.7):
        self.embedder = embedder; self.threshold = threshold
    def __call__(self, text):
        sents = pysbd_split(text)  # 返回 [(start,end), ...]
        # ... 用 embedder 相邻句子相似度，低于阈值处切开 ...
        return regions
```

---

## 8. 开源实践调研附录

本设计的灵感与对照（**这些都看过、看够了，挑能用的、不抄不能用的**）：

### LangChain
- `Document(page_content, metadata)` + `RecursiveCharacterTextSplitter`：递归分隔符 `["\n\n","\n"," ",""]` 配合 `chunk_size / chunk_overlap` 已是事实标准。**我们采纳**这个递归思路，做成 `RecursiveSplitter`。LangChain 的 RecursiveCharacterTextSplitter 尝试保持较大单元（如段落）的完整性，当单元超出 chunk 大小则降级到下一层（句子），直至必要时到词级。
- `add_start_index=True`：告诉我们字符偏移作为坐标系是被验证过的正确选择。LangChain 的 split_documents 输出 chunk 时可以通过 add_start_index=True 携带起始偏移。
- **我们不采纳**：庞大的 `metadata` 字典与一长串 LoaderClass。Loader 我们解耦到 Ingester；metadata 我们收敛为薄薄的 `tags`。

### LlamaIndex
- `Document` → `TextNode` 的两层结构，加上 `HierarchicalNodeParser` 把节点串成"小 chunk 指向大 chunk 的层级"，配合 auto-merging retrieval：Hierarchical Node Parser 从文档中接收一组节点，构造小 chunk 链接到更大 chunk 的层级结构（如叶子 512 字符、父节点 1024 字符），存储时仅 embed 叶子，其余按 ID 存储。**这个思路值得借鉴**——你的递归分块层应该实现成"父视图保留 → 叶子供检索 → 命中后向上合并"。我们的 `parent` 单向回引正是为此设计。
- **我们不采纳**：双向 `NodeRelationship`、PREV/NEXT 邻接链表。这些把简单的"列表 + parent 引用"复杂化了，递归分块完全用不上。

### Unstructured.io
- 元素类型化：`Title / NarrativeText / ListItem / Table` 等。Unstructured 的 DOCX 分区器把 Word 样式（如 Heading 1-9）映射到 Title 元素，把 Caption / Intense Quote 等映射到对应元素类型。**对我们的启示**：异构格式摄入时可以保留结构信号到 `Source.meta` 里（比如 PDF 的页码、DOCX 的 Heading 级别）。但**不把它们抬到 DocView 顶层 API**，因为不同格式信号差异大、且下游 QA 主要靠文本+偏移。
- **我们不采纳**：把每个段落都包装成独立 Element 对象。我们的 Span 更轻。

### Haystack
- `Document` + offsets_in_document：Haystack 的 ExtractedAnswer 携带 offsets_in_document 字段，标明答案在原文档中的起止位置。**佐证**：偏移坐标系是检索系统的通用语言。
- `DocumentSplitter / RecursiveDocumentSplitter`：Haystack 的 RecursiveDocumentSplitter 持续递归切分直至所有 chunk 小于 split_length。**我们对齐**这个语义。
- **我们不采纳**：Pipeline / Component 那一整套 DSL。你的需求里没有"图状数据流"，引入 Pipeline 是杀鸡用牛刀。

### 综合判断

你的需求处在一个"比简单 chunker 复杂、比 LangChain 简单"的甜点位上。直接抄 LangChain 会带进太多无关概念；自己从零写一个 1500 行的薄层、把 5 大框架里被验证有效的设计精选保留下来，是性价比最高的方案。

---

## 9. 落地步骤建议（不急于实现，按需推进）

1. **MVP（1-2 周）**：`Source` / `Span` / `DocView` + 5 个 Splitter + `Book.load/save`。能跑通"加载书本目录 → split → search → fold"全链路。
2. **接通递归分块层（2-3 天）**：把现有分块代码重构成调 DocView 的 split。这一步最考验对等性的设计——如果递归分块层有任何"父视图 vs 子视图"的特殊代码，说明 DocView API 还有问题。
3. **三个 Ingester（1 周）**：PDF / DOCX / Markdown。先做 happy path，错误处理后置。
4. **援引上下文落地（3-5 天）**：把 `to_book_spans` 输出接进现有的知识抽取与 QA 援引流程。
5. **第二版考虑**：mmap 大文件、并行 fold、增量更新（基于 sha256）。这些都是有需求才做。

---

**一句话总结**：一个数据结构（DocView） + 一个坐标系（字符偏移） + 六个方法（slice/split/search/iter/fold/project） + 一份外部协议（书本目录）。剩下的都是堆代码。
