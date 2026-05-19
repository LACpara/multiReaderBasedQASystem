# DocModel 模块测试说明文档

## 1. 概述

本文档描述 DocModel 模块的测试策略、测试用例设计和测试执行方法。

## 2. 测试框架

### 2.1 使用的测试工具

| 工具         | 版本   | 用途     |
| ---------- | ---- | ------ |
| pytest     | 9.0+ | 测试框架   |
| pytest-cov | 7.0+ | 覆盖率统计  |
| unittest   | 内置   | 部分测试基类 |

### 2.2 测试文件结构

```
docmodel/tests/
├── __init__.py           # 测试包初始化
├── test_core.py          # 核心模块测试 (Source, Span, DocView, Granularity)
├── test_splitters.py     # Splitter 测试
├── test_book.py          # Book 模块测试
├── test_ingest.py        # 摄入器测试
└── test_integration.py   # 集成测试
```

## 3. 测试分类

### 3.1 单元测试

#### 3.1.1 核心模块测试 (test\_core.py)

**TestSource 类** - Source 数据类测试

| 测试方法                                   | 测试内容           |
| -------------------------------------- | -------------- |
| test\_source\_creation                 | Source 创建和属性访问 |
| test\_source\_length                   | length 属性计算    |
| test\_source\_empty\_text              | 空文本处理          |
| test\_source\_frozen                   | 不可变性验证         |
| test\_source\_meta\_frozen             | 元数据不可变性        |
| test\_source\_default\_meta            | 默认元数据          |
| test\_source\_meta\_dict\_\_conversion | 元数据字典转换        |
| test\_source\_unicode\_text            | Unicode 文本支持   |
| test\_source\_long\_text               | 大文本处理          |

**TestSpan 类** - Span 坐标类测试

| 测试方法                                       | 测试内容        |
| ------------------------------------------ | ----------- |
| test\_span\_creation                       | Span 创建和属性  |
| test\_span\_length                         | length 属性计算 |
| test\_span\_zero\_length                   | 零长度 Span    |
| test\_span\_contains                       | contains 方法 |
| test\_span\_overlaps                       | overlaps 方法 |
| test\_span\_negative\_start\_raises        | 负起始值异常      |
| test\_span\_end\_less\_than\_start\_raises | 结束小于起始异常    |
| test\_span\_frozen                         | 不可变性验证      |

**TestDocView 类** - DocView 视图类测试

| 测试方法                                         | 测试内容         |
| -------------------------------------------- | ------------ |
| test\_docview\_creation                      | DocView 创建   |
| test\_docview\_multiple\_spans               | 多 Span 视图    |
| test\_docview\_empty                         | 空视图          |
| test\_docview\_parent                        | 父视图引用        |
| test\_docview\_tags                          | 标签处理         |
| test\_docview\_invalid\_source\_raises       | 无效 Source 异常 |
| test\_docview\_span\_exceeds\_source\_raises | Span 越界异常    |

**TestDocViewSlice 类** - slice 方法测试

| 测试方法                                        | 测试内容      |
| ------------------------------------------- | --------- |
| test\_slice\_beginning                      | 开头切片      |
| test\_slice\_middle                         | 中间切片      |
| test\_slice\_end                            | 结尾切片      |
| test\_slice\_full                           | 完整切片      |
| test\_slice\_empty                          | 空切片       |
| test\_slice\_negative\_start\_raises        | 负起始异常     |
| test\_slice\_end\_less\_than\_start\_raises | 结束小于起始异常  |
| test\_slice\_exceeds\_length\_raises        | 越界异常      |
| test\_slice\_preserves\_tags                | 标签保留      |
| test\_slice\_cross_multi_spans\_spans       | 跨 Span 切片 |
| test\_slice\_by\_span                       | 按 Span 切片 |

**TestDocViewSplit 类** - split 方法测试

| 测试方法                                    | 测试内容   |
| ------------------------------------------ | -------- |
| test\_split\_with\_simple\_splitter        | 简单 Splitter 切片 |
| test\_split\_empty\_result                 | 空结果切片 |

**TestDocViewSearch 类** - search 方法测试

| 测试方法                                       | 测试内容     |
| ------------------------------------------ | -------- |
| test\_search\_string                       | 字符串搜索    |
| test\_search\_regex                        | 正则搜索     |
| test\_search\_callable                     | 可调用对象搜索  |
| test\_search\_no\_match                    | 无匹配结果    |
| test\_search\_limit                        | 结果限制     |
| test\_search\_overlapping                  | 重叠匹配     |
| test\_search\_invalid\_query\_type\_raises | 无效查询类型异常 |

**TestDocViewIter 类** - iter 方法测试

| 测试方法                  | 测试内容      |
| ----------------------- | ------------ |
| test\_iter\_chars       | 字符迭代      |
| test\_iter\_words       | 单词迭代      |
| test\_iter\_sentences   | 句子迭代      |
| test\_iter\_lines       | 行迭代       |
| test\_iter\_paragraphs  | 段落迭代      |
| test\_iter\_sources     | Source 迭代  |
| test\_iter\_empty\_view | 空视图迭代     |

**TestDocViewFold 类** - fold 方法测试

| 测试方法                    | 测试内容     |
| ----------------------- | -------- |
| test\_fold\_sum         | 求和计算     |
| test\_fold\_collect     | 收集元素     |
| test\_fold\_empty\_view | 空视图 fold |

**TestDocViewProject 类** - project 方法测试

| 测试方法                                    | 测试内容         |
| --------------------------------------- | ------------ |
| test\_project\_child                    | 子视图坐标投影      |
| test\_project\_child\_different\_source | 不同 Source 投影 |
| test\_to\_book\_spans                   | 转换为全局 Span   |

**TestDocViewExcerpt 类** - excerpt_with_context 方法测试

| 测试方法                          | 测试内容           |
| ------------------------------- | -------------- |
| test\_excerpt\_with\_context    | 上下文扩展功能      |
| test\_excerpt\_at\_beginning    | 文本开头位置扩展     |
| test\_excerpt\_at\_end          | 文本结尾位置扩展     |
| test\_excerpt\_with\_large\_context | 大上下文扩展        |
| test\_excerpt\_empty\_view      | 空视图处理         |
| test\_excerpt\_no\_parent       | 无父视图时返回自身    |

#### 3.1.2 Splitter 测试 (test\_splitters.py)

**TestRegexSplitter 类**

| 测试方法                               | 测试内容    |
| ---------------------------------- | ------- |
| test\_split\_by\_pattern           | 按正则切分   |
| test\_split\_no\_match             | 无匹配处理   |
| test\_split\_empty\_text           | 空文本处理   |
| test\_split\_chapter\_pattern      | 章节模式切分  |
| test\_split\_include\_match\_false | 不包含匹配内容 |
| test\_repr                         | 字符串表示   |

**TestRecursiveSplitter 类**

| 测试方法                              | 测试内容      |
| --------------------------------- | --------- |
| test\_split\_small\_text          | 小文本不切分    |
| test\_split\_by\_separator        | 按分隔符切分     |
| test\_recursive\_degradation      | 递归降级过程     |
| test\_hard\_split\_fallback       | 硬切分回退      |
| test\_single\_part\_over\_max\_size | 单个部分过大场景  |
| test\_split\_with\_overlap\_correctness | 重叠计算正确性 |
| test\_split\_empty\_text          | 空文本处理      |
| test\_split\_chinese\_text        | 中文文本切分     |
| test\_split\_no\_valid\_separator | 无有效分隔符回退   |
| test\_empty\_chunk\_filtering     | 空块过滤      |
| test\_split\_preserves\_content   | 内容完整性验证    |
| test\_zero\_overlap               | overlap=0  |
| test\_large\_overlap              | 大 overlap  |
| test\_separator\_preservation     | 分隔符保留      |
| test\_repr                        | __repr__  |

**TestWindowSplitter 类**

| 测试方法                                     | 测试内容    |
| ------------------------------------------- | --------- |
| test\_split\_with\_window                   | 滑窗切分   |
| test\_split\_text\_shorter\_than\_window    | 文本短于窗口 |
| test\_split\_empty\_text                    | 空文本处理  |
| test\_split\_with\_equal\_size\_and\_stride | 等大小和步长 |
| test\_invalid\_size\_raises                 | 无效大小异常 |
| test\_invalid\_stride\_raises               | 无效步长异常 |
| test\_repr                                  | 字符串表示  |

**TestSentenceSplitter 类**

| 测试方法                            | 测试内容   |
| ------------------------------- | ------ |
| test\_split\_chinese\_sentences | 中文句子切分 |
| test\_split\_english\_sentences | 英文句子切分 |
| test\_split\_empty\_text        | 空文本处理  |
| test\_split\_single\_sentence   | 单句处理   |
| test\_split\_preserves\_content | 内容保留   |
| test\_repr                      | 字符串表示  |

**TestTagSplitter 类**

| 测试方法                                 | 测试内容  |
| ------------------------------------ | ----- |
| test\_call\_raises\_not\_implemented | 未实现异常 |
| test\_split\_with\_tags              | 按标签切分 |
| test\_repr                           | 字符串表示 |

#### 3.1.3 Book 模块测试 (test\_book.py)

**TestSourceInfo 类** - SourceInfo 数据类测试

| 测试方法                         | 测试内容  |
| ---------------------------- | ----- |
| test\_source\_info\_creation | 创建和属性 |
| test\_source\_info\_defaults | 默认值   |

**TestBookCreation 类** - Book 创建测试

| 测试方法                       | 测试内容    |
| -------------------------- | ------- |
| test\_book\_creation       | Book 创建 |
| test\_book\_with\_metadata | 元数据处理   |
| test\_book\_iteration      | 迭代支持    |
| test\_book\_getitem        | 索引访问    |
| test\_book\_repr           | 字符串表示   |

**TestBookLoad 类** - Book.load 方法测试

| 测试方法                                       | 测试内容          |
| ------------------------------------------ | ------------- |
| test\_load\_simple\_book                   | 简单书本加载        |
| test\_load\_multiple\_sources              | 多 Source 加载   |
| test\_load\_nonexistent\_directory\_raises | 目录不存在异常       |
| test\_load\_missing\_manifest\_raises      | manifest 缺失异常 |
| test\_load\_missing\_source\_file\_raises  | Source 文件缺失异常 |
| test\_load\_with\_sha256\_validation       | SHA256 验证     |
| test\_load\_sha256\_mismatch\_raises       | SHA256 不匹配异常  |
| test\_load\_unsupported\_version\_raises   | 不支持版本异常       |
| test\_load\_with\_tags                     | 标签加载          |

**TestBookSave 类** - Book.save 方法测试

| 测试方法                             | 测试内容          |
| -------------------------------- | ------------- |
| test\_save\_simple\_book         | 简单书本保存        |
| test\_save\_multiple\_sources    | 多 Source 保存   |
| test\_save\_creates\_directory   | 目录创建          |
| test\_save\_manifest\_content    | manifest 内容验证 |
| test\_save\_and\_load\_roundtrip | 保存加载往返        |

**TestBookViews 类** - Book 视图创建测试

| 测试方法                                    | 测试内容          |
| --------------------------------------- | ------------- |
| test\_root\_view                        | 根视图创建         |
| test\_source\_view                      | Source 视图创建   |
| test\_source\_view\_nonexistent\_raises | 不存在 Source 异常 |

#### 3.1.4 摄入器测试 (test\_ingest.py)

**TestMarkdownIngester 类**

| 测试方法                                     | 测试内容            |
| ---------------------------------------- | --------------- |
| test\_ingest\_simple\_markdown           | 简单 Markdown 摄入  |
| test\_ingest\_no\_h1\_headings           | 无 H1 标题处理       |
| test\_ingest\_nonexistent\_file\_raises  | 文件不存在异常         |
| test\_ingest\_non\_markdown\_raises      | 非 Markdown 文件异常 |
| test\_ingest\_unicode\_content           | Unicode 内容处理    |
| test\_sanitize\_title                    | 标题清理            |
| test\_split\_by\_h1                      | H1 切分           |
| test\_split\_by\_h1\_empty\_content      | 空内容处理           |
| test\_create\_manifest                   | manifest 创建     |
| test\_ingest\_with\_string\_paths        | 字符串路径支持         |
| test\_ingest\_creates\_output\_directory | 输出目录创建          |
| test\_ingest\_preserves\_content         | 内容保留            |
| test\_ingest\_sha256\_in\_manifest       | SHA256 记录       |

### 3.2 集成测试 (test\_integration.py)

| 测试类                        | 测试方法                           | 测试内容                    |
| ---------------------------- | --------------------------------- | ------------------------- |
| TestEndToEndWorkflow         | test\_full\_workflow              | 完整工作流测试              |
| TestMarkdownToBookToView     | test\_markdown\_to\_view          | Markdown → Book → View 流程 |
| TestSplitAndSearch           | test\_split\_then\_search         | 切分后搜索                     |
| TestFoldWithSplit            | test\_fold\_after\_split          | 切分后 fold                  |
| TestNGramEntropy             | test\_ngram\_entropy              | N-gram 熵计算                |
| TestRecursiveChunking        | test\_recursive\_chunking         | 递归分块                      |
| TestMultiSourceBook          | test\_multi\_source\_operations   | 多 Source 操作               |
| TestExcerptWithContext       | test\_excerpt\_context            | 上下文摘录                     |
| TestViewHierarchy            | test\_parent\_child\_relationship | 视图层级关系                    |
| TestProjectCoordinates       | test\_project\_coordinates        | 坐标投影                      |
| TestIterGranularities        | test\_all\_granularities          | 所有粒度迭代                    |
| TestSaveLoadRoundtrip        | test\_roundtrip\_with\_operations | 保存加载往返                    |
| TestRegexSplitterIntegration | test\_chapter\_split              | 章节切分集成                    |

## 4. 测试执行

### 4.1 运行所有测试

```bash
python -m pytest docmodel/tests/ -v
```

### 4.2 运行特定测试文件

```bash
python -m pytest docmodel/tests/test_core.py -v
```

### 4.3 运行特定测试类

```bash
python -m pytest docmodel/tests/test_core.py::TestDocView -v
```

### 4.4 运行特定测试方法

```bash
python -m pytest docmodel/tests/test_core.py::TestDocView::test_slice_beginning -v
```

### 4.5 生成覆盖率报告

```bash
python -m pytest docmodel/tests/ --cov=docmodel --cov-report=term-missing
```

### 4.6 生成 HTML 覆盖率报告

```bash
python -m pytest docmodel/tests/ --cov=docmodel --cov-report=html
```

## 5. 测试数据

### 5.1 测试数据位置

测试使用临时目录，在测试完成后自动清理。

### 5.2 测试数据类型

- 简单文本（英文、中文）
- Unicode 文本
- 大文本（性能测试）
- 空文本
- Markdown 格式文本

## 6. 测试覆盖率目标

| 模块                 | 目标覆盖率     | 实际覆盖率   |
| ------------------ | --------- | ------- |
| core.py            | ≥ 98%     | 98%     |
| book.py            | ≥ 98%     | 100%    |
| splitters.py       | ≥ 98%     | 99%     |
| ingest/markdown.py | ≥ 95%     | 100%    |
| **总体**             | **≥ 98%** | **99%** |

## 7. 持续集成

建议在 CI/CD 流程中添加以下检查：

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: python -m pytest docmodel/tests/ -v --cov=docmodel --cov-report=xml

- name: Check coverage
  run: |
    coverage report --fail-under=98
```

## 8. 测试最佳实践

1. **测试独立性**：每个测试方法独立运行，不依赖其他测试
2. **清理资源**：使用 setUp/tearDown 确保资源清理
3. **边界测试**：覆盖边界条件和异常情况
4. **命名规范**：测试方法名清晰描述测试内容
5. **断言明确**：使用明确的断言消息

## 9. 附录

### 9.1 测试命令速查

| 命令                    | 说明      |
| --------------------- | ------- |
| `pytest -v`           | 详细输出    |
| `pytest -x`           | 首次失败后停止 |
| `pytest -k "pattern"` | 按名称模式筛选 |
| `pytest --tb=short`   | 简短错误回溯  |
| `pytest -q`           | 安静模式    |

