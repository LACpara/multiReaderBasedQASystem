"""
DocModel - 面向"书本"级输入的统一文档抽象层

提供 Source, Span, DocView, Book, Granularity 等核心组件。
"""

from docmodel.core import DocView, Granularity, Source, Span
from docmodel.book import Book

__all__ = [
    "Source",
    "Span",
    "DocView",
    "Granularity",
    "Book",
]
__version__ = "1.0.0"
