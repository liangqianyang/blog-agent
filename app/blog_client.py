"""博客（blog-admin-v2）公开 API 客户端。

对接约定（探索确认）：
- 包络 {code, message, data}，成功 iff code == 0
- 列表 GET /articles/public/list：items 在 data.data，分页在 data.meta（total/current_page/last_page）
- 详情 GET /articles/public/{加密id}：完整字段含 content(HTML)；每次调用 +1 view_count
- id 为确定性 AES 加密串，当不透明字符串用，拼路径一律 quote
"""

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import quote

import httpx

logger = logging.getLogger("blog_agent.blog_client")

_MAX_PAGES = 100  # 跑飞保护：62 篇 × per_page=50 只需 2 页


class BlogApiError(Exception):
    """博客 API 返回非成功包络或网络失败。code 携带上游业务码（如 404）。"""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class ArticleListItem:
    """列表条目：仅含增量哈希与展示所需字段。"""

    id: str
    title: str
    summary: str
    published_at: str | None
    category_id: int | None
    category_name: str | None
    label_ids: tuple[int, ...]

    @classmethod
    def from_api(cls, item: dict) -> "ArticleListItem":
        category = item.get("category") or {}
        labels = item.get("labels") or []
        return cls(
            id=str(item["id"]),
            title=item.get("title") or "",
            summary=item.get("summary") or "",
            published_at=item.get("published_at"),
            category_id=category.get("id"),
            category_name=category.get("name"),
            label_ids=tuple(sorted(int(lb["id"]) for lb in labels if lb.get("id") is not None)),
        )

    def list_hash(self) -> str:
        """增量同步判据：标题/摘要/发布时间/分类/标签任一变化即视为需要重建。

        局限：列表无 updated_at，正文改动但元数据未动时检测不到，需手动 --full。
        """
        import hashlib

        raw = "|".join(
            [
                self.title,
                self.summary,
                self.published_at or "",
                str(self.category_id),
                ",".join(map(str, self.label_ids)),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BlogClient:
    def __init__(self, base_url: str, timeout: float = 15.0):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": "blog-agent/0.1"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """GET + 包络解析。网络错误与 5xx 重试 2 次，退避 0.5s。"""
        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(0.5)
            try:
                resp = await self._client.get(path, params=params)
                if resp.status_code >= 500:
                    last_exc = BlogApiError(f"上游 {resp.status_code}: {path}")
                    continue
                body = resp.json()
                if body.get("code") != 0:
                    raise BlogApiError(
                        f"包络失败 code={body.get('code')} message={body.get('message')}: {path}",
                        code=body.get("code"),
                    )
                return body
            except httpx.TransportError as exc:
                last_exc = exc
                continue
        raise BlogApiError(f"请求失败（已重试）: {path}") from last_exc

    async def fetch_public_articles(self, per_page: int = 50) -> list[ArticleListItem]:
        """分页拉全量已发布文章列表（不触详情，不影响 view_count）。"""
        items: list[ArticleListItem] = []
        page = 1
        while True:
            body = await self._get(
                "/articles/public/list", params={"page": page, "per_page": per_page}
            )
            payload = body.get("data") or {}
            batch = payload.get("data") or []
            items.extend(ArticleListItem.from_api(it) for it in batch)
            meta = payload.get("meta") or {}
            last_page = int(meta.get("last_page") or 1)
            if page >= last_page or not batch:
                break
            page += 1
            if page > _MAX_PAGES:
                raise BlogApiError(f"分页超过 {_MAX_PAGES} 页保护上限，中止")
        return items

    async def fetch_article_detail(self, encrypted_id: str) -> dict:
        """拉单篇详情（完整 content HTML）。会 +1 view_count，调用方控制频率。"""
        body = await self._get(f"/articles/public/{quote(encrypted_id, safe='')}")
        return body.get("data") or {}
