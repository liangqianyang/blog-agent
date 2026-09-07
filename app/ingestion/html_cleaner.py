"""HTML 清洗：去噪音标签与注释，输出规范化 HTML 供切片。"""

from bs4 import BeautifulSoup
from bs4.element import Comment

# 与正文无关的标签，直接删除（连同子树）
_NOISE_TAGS = ("script", "style", "iframe", "form", "ins", "noscript", "button", "svg")


def clean(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    root = soup.body or soup
    return root.decode_contents().strip()
