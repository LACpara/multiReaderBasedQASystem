"""
摄入器基类和协议定义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Ingester(Protocol):
    """
    摄入器协议。
    读入异构文件，输出标准 book 目录。
    """

    def ingest(self, input_path: Path | str, output_dir: Path | str) -> Path:
        """
        读入异构文件，输出标准 book 目录路径。

        Args:
            input_path: 输入文件路径
            output_dir: 输出目录路径

        Returns:
            生成的 book 目录路径
        """
        ...
