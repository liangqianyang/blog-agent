"""结构感知切片：按 HTML 标题层级切 section，段内打包，代码块/表格原子。"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag

# 打包参数（字符数）
TARGET_SIZE = 500
HARD_CAP = 1200
ATOMIC_SPLIT_AT = 3000  # 超长代码块/表格的拆分阈值（embedding 输入 ≤3072 token）

_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
_CONTAINER_TAGS = {"div", "section", "article", "main", "figure"}
# 句末标点（中英），用于超长段落按句拆分
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|(?<=\.\s)|(?<=\n)")


@dataclass
class Chunk:
    text: str
    heading: str  # 最近一级标题（preamble 为 ""）
    heading_path: str  # 面包屑 "安装 > 配置环境变量"（preamble 为 ""）
    kind: str  # prose | code | table


def chunk_html(html: str) -> list[Chunk]:
    """把清洗后的 HTML 切成 Chunk 列表。

    - h1-h4 为 section 边界，标题栈维护面包屑路径
    - `<pre>` / `<table>` 原子成块，绝不与散文混合；超 ATOMIC_SPLIT_AT 再按内部结构拆
    - 散文段累积到 ~TARGET_SIZE 字，硬上限 HARD_CAP
    """
    soup = BeautifulSoup(html, "lxml")
    root = soup.body or soup

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    prose_buf: list[str] = []

    def _path() -> tuple[str, str]:
        heading = heading_stack[-1][1] if heading_stack else ""
        path = " > ".join(text for _, text in heading_stack)
        return heading, path

    def _flush_prose() -> None:
        if not prose_buf:
            return
        heading, path = _path()
        text = "\n\n".join(prose_buf).strip()
        prose_buf.clear()
        if not text:
            return
        for piece in _split_long(text, HARD_CAP):
            chunks.append(Chunk(text=piece, heading=heading, heading_path=path, kind="prose"))

    def _emit_atomic(tag: Tag, kind: str) -> None:
        _flush_prose()
        heading, path = _path()
        if kind == "code":
            text = tag.get_text("\n").strip("\n")
            groups = _split_by_blank_lines(text)
        else:
            text = tag.get_text(" ", strip=True)
            groups = [text]
        buf = ""
        for g in groups:
            if buf and len(buf) + len(g) + 2 > ATOMIC_SPLIT_AT:
                chunks.append(Chunk(text=buf, heading=heading, heading_path=path, kind=kind))
                buf = g
            else:
                buf = f"{buf}\n\n{g}" if buf else g
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), heading=heading, heading_path=path, kind=kind))

    def _walk(el: Tag) -> None:
        for child in el.find_all(recursive=False):
            name = child.name
            if name in _HEADING_LEVELS:
                _flush_prose()
                level = _HEADING_LEVELS[name]
                text = child.get_text(" ", strip=True)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
            elif name == "pre":
                _emit_atomic(child, "code")
            elif name == "table":
                _emit_atomic(child, "table")
            elif name in _CONTAINER_TAGS:
                _walk(child)
            elif name in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    t = li.get_text(" ", strip=True)
                    if t:
                        prose_buf.append(t)
                        _maybe_flush()
            elif name in ("img", "br", "hr"):
                continue
            else:
                t = child.get_text(" ", strip=True)
                if t:
                    prose_buf.append(t)
                    _maybe_flush()

    def _maybe_flush() -> None:
        """段间边界处：已够目标大小就收口，避免无上限累积。"""
        if sum(len(p) for p in prose_buf) >= TARGET_SIZE:
            _flush_prose()

    _walk(root)
    _flush_prose()
    return chunks


def _split_long(text: str, cap: int) -> list[str]:
    """超长散文按句拆到 cap 内；单句仍超长则硬切。"""
    if len(text) <= cap:
        return [text]
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        if len(s) > cap:  # 单句超长（如长 URL），硬切
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(s[i : i + cap] for i in range(0, len(s), cap))
        elif len(buf) + len(s) > cap:
            pieces.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        pieces.append(buf)
    return pieces


def _split_by_blank_lines(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]
