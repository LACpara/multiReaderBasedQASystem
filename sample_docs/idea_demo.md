# 分层多智能体检索系统 Demo 文档

本系统的核心目标是把静态文档库转化为一个可交互的分布式知识网络。传统 RAG 更像是在文档片段中寻找相似文本，而本系统更关注寻找能够回答问题的专家 Reader。每个 Reader 保存局部知识，知道自己能回答什么，也知道自己不能回答什么。

## Reader 树

系统整体组织为递归的 Tree of Readers。rootReader 对应完整文档，内部 Reader 对应较大的语义块，叶子 Reader 对应最小信息单元。每个 Reader 都包含结构化知识 K、子 Reader 集合 S、读取和查询策略 pi，以及通信接口 C。这样的组织让系统可以在局部回答和全局整合之间建立清晰边界。

## 阅读阶段

在 ingestion 阶段，系统先评估文本复杂度。复杂度可以由文本长度、信息熵、概念密度等因素估计。复杂度较低时，Reader 直接就地阅读并提炼知识。复杂度较高时，Reader 触发动态分裂，将文本按语义连续性拆分为多个子块，并递归创建 subReader。

每个 Reader 的知识提炼需要同时保留宏观概括和微观细节。宏观概括用于快速判断主题，微观细节包括实体、关系、例外、限制、阈值等低频但高价值的信息。

## 能力建模与索引

Reader 不只表示自己存储了什么文本，还表示自己有能力回答什么问题。因此每个 Reader 会生成一组 capability questions。这些问题与 summary、entities、relations 一起组成可索引的能力表示。

索引层采用两阶段激活架构。第一阶段是粗召回，系统把用户 Query 向量化，并在 Chroma 向量数据库中检索与 capability questions 相近的 Reader。第二阶段是精筛选，被召回的 Reader 根据自己的结构化知识进行自评估，只有相关性超过阈值的 Reader 才会响应。

## 查询阶段

查询阶段采用自下而上的 self-activated retrieval。一个复杂问题可以被多个 Reader 部分回答。比如用户问“请对比 A 框架的路由机制和 B 框架的状态管理，并总结它们对首屏加载的影响”，A 框架 Reader 只认领路由机制，B 框架 Reader 只认领状态管理，性能 Reader 认领首屏加载影响。无法回答的 Reader 保持静默。

父级或整合器会收集多个 Reader 的局部回答，按照专家优先于万事通的原则进行去重、冲突处理和最终整合。回答越聚焦的 Reader 权重越高，泛泛而谈的 Reader 权重越低。

## Demo 工程落地

本 demo 使用 SQLite 保存 Reader 元数据、树结构、结构化知识和查询日志。SQLite 被封装在 KnowledgeStore 抽象后面，后续迁移到 PostgreSQL 或 MySQL 时不需要改动核心检索逻辑。

本 demo 使用 ChromaDB 保存 Reader 的 capability questions 向量索引。Chroma 被封装在 VectorIndex 抽象后面，后续迁移到 Milvus、Qdrant、Weaviate 时只需要替换向量索引实现。

LLM 调用通过 ReaderLLMService 与 LLMClient 两层抽象隔离。核心逻辑只依赖 ReaderLLMService 的高层语义方法，例如 extract_knowledge、build_capability_questions、evaluate_activation、answer_question 和 merge_answers。quick start 默认使用 HeuristicReaderLLMService，不需要 API key；真实服务可以换成 PromptedReaderLLMService 加 OpenAICompatibleLLMClient。
