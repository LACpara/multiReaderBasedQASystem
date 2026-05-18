"""
摄入器模块。

提供异构格式到书本目录的转换。
"""

from docmodel.ingest.base import Ingester
from docmodel.ingest.markdown import MarkdownIngester

__all__ = [
    "Ingester",
    "MarkdownIngester",
]
