"""
Path utilities.

Supports resolving data roots from plain text manifest files used by legacy
VeOmni training configs, where each line points to a parquet/json/csv root.
"""

import os
from typing import List


def resolve_data_paths(data_path: str) -> List[str]:
    """
    Resolve dataset paths from a comma-separated string or a manifest file.

    Args:
        data_path: Either a comma-separated path list, or a `.txt`/`.paths`
            file whose non-comment lines are dataset roots.

    Returns:
        A flat list of concrete dataset roots.
    """
    if (data_path.endswith(".txt") or data_path.endswith(".paths")) and os.path.isfile(data_path):
        return read_paths_from_file(data_path)

    return [path.strip() for path in data_path.split(",") if path.strip()]


def read_paths_from_file(file_path: str) -> List[str]:
    """
    Read data roots from a manifest file.

    Lines starting with `#` are ignored. Inline comments are supported.
    """
    paths = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            comment_start = None
            in_quotes = False
            quote_char = None

            for i, char in enumerate(line):
                if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif char == "#" and not in_quotes:
                    comment_start = i
                    break

            if comment_start is not None:
                line = line[:comment_start].strip()

            if line:
                if not os.path.exists(line) and not line.startswith("hdfs://"):
                    print(f"警告: 第{line_num}行路径可能不存在: {line}")
                paths.append(line)

    if not paths:
        raise ValueError(f"文件 {file_path} 中没有找到有效的路径")

    return paths
