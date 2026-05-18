# DocModel 模块过程设计文档

## 1. 概述

本文档描述 DocModel 模块的开发过程、实现策略和质量保障方法。DocModel 是一个面向"书本"级输入的统一文档抽象层，支持多种格式的文档摄入、切分、检索和迭代计算。

## 2. 开发流程

### 2.1 开发方法论

采用**测试驱动开发 (TDD)** 方法论：

1. **先写测试**：针对每个功能点先编写测试用例
2. **编写实现**：编写最小代码使测试通过
3. **重构优化**：在测试保护下优化代码结构

### 2.2 开发阶段划分

| 阶段   | 内容          | 产出                             |
| ---- | ----------- | ------------------------------ |
| 第一阶段 | 核心数据模型      | Source, Span, DocView 类及测试     |
| 第二阶段 | Book 协议实现   | Book 类、manifest 处理及测试          |
| 第三阶段 | Splitter 实现 | 5 个内置 Splitter 及测试             |
| 第四阶段 | 摄入器实现       | Pdf/Docx/Markdown Ingester 及测试 |
| 第五阶段 | 集成测试        | 端到端测试用例                        |
| 第六阶段 | 覆盖率优化       | 补充边界测试用例                       |

### 2.3 迭代周期

每个功能模块的开发遵循以下周期：

```
编写测试 → 运行测试(失败) → 编写实现 → 运行测试(通过) → 代码审查 → 提交
```

## 3. 模块依赖关系

### 3.1 依赖图

```
docmodel/
├── core.py          ← 无内部依赖（基础模块）
├── book.py          ← 依赖 core.py
├── splitters.py     ← 依赖 core.py
└── ingest/
    ├── __init__.py
    ├── base.py      ← 无内部依赖
    ├── pdf.py       ← 依赖 base.py, book.py
    ├── docx.py      ← 依赖 base.py, book.py
    └── markdown.py  ← 依赖 base.py, book.py
```

### 3.2 外部依赖

| 依赖             | 用途      | 是否必需                  |
| -------------- | ------- | --------------------- |
| pytest         | 测试框架    | 必需                    |
| pytest-cov     | 覆盖率统计   | 必需                    |
| PyMuPDF (fitz) | PDF 解析  | 可选（仅 PdfIngester 需要）  |
| python-docx    | DOCX 解析 | 可选（仅 DocxIngester 需要） |

## 4. 测试策略

### 4.1 测试层次

```
┌─────────────────────────────────────┐
│          集成测试 (E2E)              │  ← 完整工作流测试
├─────────────────────────────────────┤
│        单元测试 (模块间)              │  ← 模块交互测试
├─────────────────────────────────────┤
│        单元测试 (模块内)              │  ← 单个类/函数测试
└─────────────────────────────────────┘
```

### 4.2 测试分类

#### 4.2.1 单元测试

- **核心模块测试** (`test_core.py`)
  - Source 创建和不可变性
  - Span 坐标计算
  - DocView 切片、搜索、迭代
  - 边界条件处理
- **Book 模块测试** (`test_book.py`)
  - manifest 解析和生成
  - Book 加载和保存
  - 多 Source 拼接
- **Splitter 测试** (`test_splitters.py`)
  - 各 Splitter 切分逻辑
  - 边界情况处理
  - 重叠切分
- **摄入器测试** (`test_ingest.py`)
  - 格式转换正确性
  - 错误处理

#### 4.2.2 集成测试

- **端到端测试** (`test_integration.py`)
  - 完整工作流：摄入 → 加载 → 切分 → 检索 → 计算
  - 多格式混合处理

### 4.3 测试用例设计原则

1. **等价类划分**：对每个输入域识别有效/无效等价类
2. **边界值分析**：重点测试边界条件（空字符串、最大长度、零值等）
3. **错误猜测**：基于经验预测可能的错误
4. **状态覆盖**：覆盖对象的所有状态转换

### 4.4 测试覆盖率目标

| 模块           | 目标覆盖率     |
| ------------ | --------- |
| core.py      | ≥ 99%     |
| book.py      | ≥ 99%     |
| splitters.py | ≥ 98%     |
| ingest/\*.py | ≥ 95%     |
| **总体**       | **≥ 98%** |

## 5. 代码规范

### 5.1 命名约定

| 类型   | 约定                    | 示例                               |
| ---- | --------------------- | -------------------------------- |
| 类名   | PascalCase            | `DocView`, `RegexSplitter`       |
| 函数名  | snake\_case           | `slice_by_span`, `to_book_spans` |
| 常量   | UPPER\_SNAKE          | `DEFAULT_CHUNK_SIZE`             |
| 私有方法 | \_leading\_underscore | `_translate_spans`               |

### 5.2 类型注解

所有公开 API 必须包含类型注解：

```python
def slice(self, start: int, end: int) -> "DocView":
    ...
```

### 5.3 文档字符串

使用 Google 风格文档字符串：

```python
def search(self, query: str) -> List["DocView"]:
    """Search for text within the view.

    Args:
        query: The search string or pattern.

    Returns:
        A list of DocView objects matching the query.
    """
```

## 6. 质量保障措施

### 6.1 静态检查

- 使用 `ruff` 进行代码风格检查
- 使用 `mypy` 进行类型检查（可选）

### 6.2 持续验证

每次代码变更后执行：

```bash
pytest --cov=docmodel --cov-report=term-missing
```

### 6.3 代码审查清单

- [ ] 测试用例覆盖所有分支
- [ ] 边界条件已处理
- [ ] 类型注解完整
- [ ] 文档字符串清晰
- [ ] 无硬编码值
- [ ] 异常处理恰当

## 7. 发布检查清单

- [ ] 所有测试通过
- [ ] 覆盖率 ≥ 98%
- [ ] 文档与代码一致
- [ ] 无安全漏洞
- [ ] 性能测试通过（可选）

## 8. 风险与缓解

| 风险      | 影响 | 缓解措施               |
| ------- | -- | ------------------ |
| 大文件内存溢出 | 高  | 实现 lazy loading 机制 |
| 编码问题    | 中  | 强制 UTF-8，提供编码检测    |
| 第三方库兼容性 | 中  | 版本锁定，可选依赖处理        |

## 9. 附录

### 9.1 测试命令

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_core.py

# 运行并生成覆盖率报告
pytest --cov=docmodel --cov-report=html

# 运行详细输出
pytest -v
```

### 9.2 目录结构

```
docmodel/
├── __init__.py
├── core.py
├── book.py
├── splitters.py
├── ingest/
│   ├── __init__.py
│   ├── base.py
│   ├── pdf.py
│   ├── docx.py
│   └── markdown.py
└── tests/
    ├── __init__.py
    ├── test_core.py
    ├── test_book.py
    ├── test_splitters.py
    ├── test_ingest.py
    └── test_integration.py
```

