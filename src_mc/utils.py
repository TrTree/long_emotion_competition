"""Utility helpers for the MC pipeline.

提供 JSONL 读写、文本处理等基础工具，供 MC 流水线复用。"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Dict, Generator, List


def read_jsonl(path: str) -> Generator[Dict, None, None]:
    """Yield dictionaries from a UTF-8 encoded JSONL file.

    读取 JSON Lines 文件，每次返回一条记录的字典表示。"""

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    """Write an iterable of dictionaries to a JSONL file.

    将多条记录逐行写入目标文件，默认保留中文字符。"""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _basic_tokens(text: str) -> List[str]:
    """使用简单的正则表达式进行粗粒度分词。"""

    pattern = re.compile(r"[\u4e00-\u9fff]|\w+|[^\s\w]")
    return pattern.findall(text)


def tokenize_len(text: str) -> int:
    """Estimate the approximate token length of text.

    利用简单分词结果估算 token 数，供窗口控制使用。"""

    tokens = _basic_tokens(text)
    return len(tokens)


def last_client_turn(dialog: str) -> str:
    """Return the content of the last utterance prefixed with 'Client:'.

    遍历对话末尾，提取来访者最新一句话。"""

    if not dialog:
        return ""

    for line in reversed(dialog.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Client:"):
            return line[len("Client:") :].strip()
    return ""


def extract_language_hint(text: str) -> str:
    """Rudimentary language detector returning 'zh' or 'en'.

    基于字符出现次数的启发式判断文本主要语言。"""

    if not text:
        return "zh"

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    ascii_letters = re.findall(r"[A-Za-z]", text)
    return "zh" if len(chinese_chars) >= len(ascii_letters) else "en"


def strip_reasoning_prefix(text: str) -> str:
    """Remove any leading <think>...</think> reasoning blocks from a reply.

    清理模型输出中的显式思维链片段，确保回复干净。"""

    if not text:
        return text

    cleaned = text

    while True:
        start = cleaned.find("<think>")
        if start == -1:
            break

        end = cleaned.find("</think>", start)
        if end != -1:
            cleaned = cleaned[:start] + cleaned[end + len("</think>") :]
            cleaned = cleaned.lstrip()
            continue

        after = cleaned[start + len("<think>") :]
        double_newline = after.find("\n\n")
        if double_newline != -1:
            cleaned = cleaned[:start] + after[double_newline + 2 :]
        else:
            newline = after.find("\n")
            if newline != -1:
                cleaned = cleaned[:start] + after[newline + 1 :]
            else:
                cleaned = cleaned[:start]
        cleaned = cleaned.lstrip()
        break

    return cleaned.strip()

