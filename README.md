# HMR Demo：分层多智能体 Reader 检索系统

这是一个基于 `idea_pro.md` 设计的 demo 版本，重点展示：

- Tree of Readers 文档摄取与动态分裂；
- Reader 能力问题 `capability_questions` 建模；
- ChromaDB 粗召回；
- Reader 自评估激活；
- 多 Reader 局部回答整合；
- SQLite / Chroma / LLM 调用全部通过抽象层与核心逻辑解耦。

## 目录结构

```text
hmr_demo_project/
├── hmr/
│   ├── app.py                         # 应用门面，预组装各模块
│   ├── reader_builder.py              # Tree of Readers 构建
│   ├── retrieval_engine.py            # 两阶段检索与激活
│   ├── complexity.py                  # 文本复杂度估计
│   ├── text_splitter.py               # 语义连续性优先的简单切分器
│   ├── llm/
│   │   ├── base.py                    # LLMClient / ReaderLLMService 抽象
│   │   ├── heuristic_service.py       # 无 API key 的 demo LLM 替身
│   │   ├── prompted_service.py        # 可接真实 LLM 的 Reader 服务
│   │   └── openai_compatible.py       # OpenAI-compatible 低层客户端
│   ├── storage/
│   │   ├── base.py                    # KnowledgeStore 抽象
│   │   └── sqlite_store.py            # SQLite 实现
│   └── vector/
│       ├── base.py                    # VectorIndex 抽象
│       ├── embedding.py               # 本地 hash embedding，用于 demo
│       └── chroma_index.py            # ChromaDB 实现
├── sample_docs/idea_demo.md           # 示例知识文档
├── run_demo.py                        # 一键运行入口
├── quick_start.sh                     # macOS / Linux quick start
├── quick_start.bat                    # Windows quick start
└── requirements.txt
```

## 快速开始

macOS / Linux：

```bash
bash quick_start.sh
```

Windows：

```bat
quick_start.bat
```

或手动运行：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py --reset
```

自定义问题：

```bash
python run_demo.py --reset --query "Reader 自评估激活的作用是什么？"
```

开启详细日志：

```bash
python run_demo.py --reset --verbose
```

运行后会生成：

- `runtime/hmr_demo.sqlite3`：Reader 树、结构化知识、查询日志；
- `runtime/chroma/`：ChromaDB 向量索引；
- `runtime/hmr_demo.log`：详细运行日志。

## 替换真实 LLM

核心逻辑依赖的是 `ReaderLLMService`，而不是某个厂商 SDK。默认实现是 `HeuristicReaderLLMService`。

接真实 LLM 时可以这样组装：

```python
from hmr.app import ReaderRetrievalApp
from hmr.config import AppConfig
from hmr.llm.openai_compatible import OpenAICompatibleLLMClient
from hmr.llm.prompted_service import PromptedReaderLLMService

client = OpenAICompatibleLLMClient(
    api_key="YOUR_API_KEY",
    model="YOUR_MODEL",
    base_url="https://api.openai.com/v1",
)
llm_service = PromptedReaderLLMService(client)
app = ReaderRetrievalApp(AppConfig(), llm_service=llm_service)
```

替换数据库或向量库时，只需要实现 `KnowledgeStore` 或 `VectorIndex` 协议，不需要改动 `ReaderTreeBuilder` 和 `RetrievalEngine`。
