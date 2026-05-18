# DocModel 模块测试说明文档

## 1. 概述

本文档描述 DocModel 模块的测试策略、测试用例设计和测试执行方法。

## 2. 测试框架

### 2.1 使用的测试工具

| 工具 | 版本 | 用途 |
|------|------|------|
| pytest | 9.0+ | 测试框架 |
| pytest-cov | 7.0+ | 覆盖率统计 |
| unittest | 内置 | 部分测试基类 |

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

#### 3.1.1 核心模块测试 (test_core.py)

**TestSource 类** - Source 数据类测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_source_creation | Source 创建和属性访问 |
| test_source_length | length 属性计算 |
| test_source_empty_text | 空文本处理 |
| test_source_frozen | 不可变性验证 |
| test_source_meta_frozen | 元数据不可变性 |
| test_source_default_meta | 默认元数据 |
| test_source_unicode_text | Unicode 文本支持 |
| test_source_long_text | 大文本处理 |

**TestSpan 类** - Span 坐标类测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_span_creation | Span 创建和属性 |
| test_span_length | length 属性计算 |
| test_span_zero_length | 零长度 Span |
| test_span_contains | contains 方法 |
| test_span_overlaps | overlaps 方法 |
| test_span_negative_start_raises | 负起始值异常 |
| test_span_end_less_than_start_raises | 结束小于起始异常 |
| test_span_frozen | 不可变性验证 |

**TestDocView 类** - DocView 视图类测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_docview_creation | DocView 创建 |
| test_docview_multiple_spans | 多 Span 视图 |
| test_docview_empty | 空视图 |
| test_docview_parent | 父视图引用 |
| test_docview_tags | 标签处理 |
| test_docview_invalid_source_raises | 无效 Source 异常 |
| test_docview_span_exceeds_source_raises | Span 越界异常 |

**TestDocViewSlice 类** - slice 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_slice_beginning | 开头切片 |
| test_slice_middle | 中间切片 |
| test_slice_end | 结尾切片 |
| test_slice_full | 完整切片 |
| test_slice_empty | 空切片 |
| test_slice_negative_start_raises | 负起始异常 |
| test_slice_end_less_than_start_raises | 结束小于起始异常 |
| test_slice_exceeds_length_raises | 越界异常 |
| test_slice_preserves_tags | 标签保留 |
| test_slice_by_span | 按 Span 切片 |

**TestDocViewSearch 类** - search 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_search_string | 字符串搜索 |
| test_search_regex | 正则搜索 |
| test_search_callable | 可调用对象搜索 |
| test_search_no_match | 无匹配结果 |
| test_search_limit | 结果限制 |
| test_search_overlapping | 重叠匹配 |
| test_search_invalid_query_type_raises | 无效查询类型异常 |

**TestDocViewIter 类** - iter 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_iter_chars | 字符迭代 |
| test_iter_words | 单词迭代 |
| test_iter_sentences | 句子迭代 |
| test_iter_lines | 行迭代 |
| test_iter_paragraphs | 段落迭代 |
| test_iter_sources | Source 迭代 |
| test_iter_empty_view | 空视图迭代 |

**TestDocViewFold 类** - fold 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_fold_sum | 求和计算 |
| test_fold_collect | 收集元素 |
| test_fold_empty_view | 空视图 fold |

**TestDocViewProject 类** - project 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_project_child | 子视图坐标投影 |
| test_project_child_different_source | 不同 Source 投影 |
| test_to_book_spans | 转换为全局 Span |

#### 3.1.2 Splitter 测试 (test_splitters.py)

**TestRegexSplitter 类**
| 测试方法 | 测试内容 |
|----------|----------|
| test_split_by_pattern | 按正则切分 |
| test_split_no_match | 无匹配处理 |
| test_split_empty_text | 空文本处理 |
| test_split_chapter_pattern | 章节模式切分 |
| test_split_include_match_false | 不包含匹配内容 |
| test_repr | 字符串表示 |

**TestRecursiveSplitter 类**
| 测试方法 | 测试内容 |
|----------|----------|
| test_split_small_text | 小文本处理 |
| test_split_by_separator | 按分隔符切分 |
| test_split_with_overlap | 重叠切分 |
| test_split_empty_text | 空文本处理 |
| test_split_chinese_text | 中文文本切分 |
| test_split_no_valid_separator | 无有效分隔符 |
| test_split_preserves_content | 内容保留 |
| test_repr | 字符串表示 |

**TestWindowSplitter 类**
| 测试方法 | 测试内容 |
|----------|----------|
| test_split_with_window | 滑窗切分 |
| test_split_text_shorter_than_window | 文本短于窗口 |
| test_split_empty_text | 空文本处理 |
| test_split_with_equal_size_and_stride | 等大小和步长 |
| test_invalid_size_raises | 无效大小异常 |
| test_invalid_stride_raises | 无效步长异常 |
| test_repr | 字符串表示 |

**TestSentenceSplitter 类**
| 测试方法 | 测试内容 |
|----------|----------|
| test_split_chinese_sentences | 中文句子切分 |
| test_split_english_sentences | 英文句子切分 |
| test_split_empty_text | 空文本处理 |
| test_split_single_sentence | 单句处理 |
| test_split_preserves_content | 内容保留 |
| test_repr | 字符串表示 |

**TestTagSplitter 类**
| 测试方法 | 测试内容 |
|----------|----------|
| test_call_raises_not_implemented | 未实现异常 |
| test_split_with_tags | 按标签切分 |
| test_repr | 字符串表示 |

#### 3.1.3 Book 模块测试 (test_book.py)

**TestSourceInfo 类** - SourceInfo 数据类测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_source_info_creation | 创建和属性 |
| test_source_info_defaults | 默认值 |

**TestBookCreation 类** - Book 创建测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_book_creation | Book 创建 |
| test_book_with_metadata | 元数据处理 |
| test_book_iteration | 迭代支持 |
| test_book_getitem | 索引访问 |
| test_book_repr | 字符串表示 |

**TestBookLoad 类** - Book.load 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_load_simple_book | 简单书本加载 |
| test_load_multiple_sources | 多 Source 加载 |
| test_load_nonexistent_directory_raises | 目录不存在异常 |
| test_load_missing_manifest_raises | manifest 缺失异常 |
| test_load_missing_source_file_raises | Source 文件缺失异常 |
| test_load_with_sha256_validation | SHA256 验证 |
| test_load_sha256_mismatch_raises | SHA256 不匹配异常 |
| test_load_unsupported_version_raises | 不支持版本异常 |
| test_load_with_tags | 标签加载 |

**TestBookSave 类** - Book.save 方法测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_save_simple_book | 简单书本保存 |
| test_save_multiple_sources | 多 Source 保存 |
| test_save_creates_directory | 目录创建 |
| test_save_manifest_content | manifest 内容验证 |
| test_save_and_load_roundtrip | 保存加载往返 |

**TestBookViews 类** - Book 视图创建测试
| 测试方法 | 测试内容 |
|----------|----------|
| test_root_view | 根视图创建 |
| test_source_view | Source 视图创建 |
| test_source_view_nonexistent_raises | 不存在 Source 异常 |

#### 3.1.4 摄入器测试 (test_ingest.py)

**TestMarkdownIngester 类**
| 测试方法 | 测试内容 |
|----------|----------|
| test_ingest_simple_markdown | 简单 Markdown 摄入 |
| test_ingest_no_h1_headings | 无 H1 标题处理 |
| test_ingest_nonexistent_file_raises | 文件不存在异常 |
| test_ingest_non_markdown_raises | 非 Markdown 文件异常 |
| test_ingest_unicode_content | Unicode 内容处理 |
| test_sanitize_title | 标题清理 |
| test_split_by_h1 | H1 切分 |
| test_split_by_h1_empty_content | 空内容处理 |
| test_create_manifest | manifest 创建 |
| test_ingest_with_string_paths | 字符串路径支持 |
| test_ingest_creates_output_directory | 输出目录创建 |
| test_ingest_preserves_content | 内容保留 |
| test_ingest_sha256_in_manifest | SHA256 记录 |

### 3.2 集成测试 (test_integration.py)

| 测试类 | 测试方法 | 测试内容 |
|--------|----------|----------|
| TestEndToEndWorkflow | test_full_workflow | 完整工作流测试 |
| TestMarkdownToBookToView | test_markdown_to_view | Markdown → Book → View 流程 |
| TestSplitAndSearch | test_split_then_search | 切分后搜索 |
| TestFoldWithSplit | test_fold_after_split | 切分后 fold |
| TestNGramEntropy | test_ngram_entropy | N-gram 熵计算 |
| TestRecursiveChunking | test_recursive_chunking | 递归分块 |
| TestMultiSourceBook | test_multi_source_operations | 多 Source 操作 |
| TestExcerptWithContext | test_excerpt_context | 上下文摘录 |
| TestViewHierarchy | test_parent_child_relationship | 视图层级关系 |
| TestProjectCoordinates | test_project_coordinates | 坐标投影 |
| TestIterGranularities | test_all_granularities | 所有粒度迭代 |
| TestSaveLoadRoundtrip | test_roundtrip_with_operations | 保存加载往返 |
| TestRegexSplitterIntegration | test_chapter_split | 章节切分集成 |

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

| 模块 | 目标覆盖率 | 实际覆盖率 |
|------|-----------|-----------|
| core.py | ≥ 98% | 98% |
| book.py | ≥ 98% | 100% |
| splitters.py | ≥ 98% | 99% |
| ingest/markdown.py | ≥ 95% | 100% |
| **总体** | **≥ 98%** | **99%** |

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

| 命令 | 说明 |
|------|------|
| `pytest -v` | 详细输出 |
| `pytest -x` | 首次失败后停止 |
| `pytest -k "pattern"` | 按名称模式筛选 |
| `pytest --tb=short` | 简短错误回溯 |
| `pytest -q` | 安静模式 |
