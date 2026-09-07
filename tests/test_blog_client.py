"""blog_client 单测：包络解析、分页、加密 id 的 URL 编码。respx 模拟上游。"""

from urllib.parse import quote

import pytest
import respx
from httpx import Response

from app.blog_client import ArticleListItem, BlogApiError, BlogClient

BASE = "http://blog.test/api"


def _list_body(items: list[dict], page: int, last_page: int) -> dict:
    return {
        "code": 0,
        "message": "获取成功",
        "data": {
            "data": items,
            "links": {},
            "meta": {"current_page": page, "last_page": last_page, "total": len(items) * last_page},
        },
    }


def _item(id_: str, title: str = "标题") -> dict:
    return {
        "id": id_,
        "title": title,
        "summary": "摘要",
        "published_at": "2026-08-01 10:00:00",
        "category": {"id": 3, "name": "后端"},
        "labels": [{"id": 2, "title": "Laravel"}, {"id": 1, "title": "PHP"}],
    }


class TestFetchList:
    @respx.mock
    async def test_pagination_follows_last_page(self):
        page1 = _list_body([_item("aaa"), _item("bbb")], 1, 2)
        page2 = _list_body([_item("ccc")], 2, 2)
        respx.get(f"{BASE}/articles/public/list").mock(
            side_effect=[Response(200, json=page1), Response(200, json=page2)]
        )
        client = BlogClient(BASE)
        items = await client.fetch_public_articles()
        await client.aclose()
        assert [i.id for i in items] == ["aaa", "bbb", "ccc"]

    @respx.mock
    async def test_envelope_failure_raises(self):
        respx.get(f"{BASE}/articles/public/list").mock(
            return_value=Response(200, json={"code": 500, "message": "服务器错误"})
        )
        client = BlogClient(BASE)
        with pytest.raises(BlogApiError):
            await client.fetch_public_articles()
        await client.aclose()

    @respx.mock
    async def test_5xx_retries_then_succeeds(self):
        page = _list_body([_item("aaa")], 1, 1)
        route = respx.get(f"{BASE}/articles/public/list").mock(
            side_effect=[Response(502, text="bad gw"), Response(200, json=page)]
        )
        client = BlogClient(BASE)
        items = await client.fetch_public_articles()
        await client.aclose()
        assert len(items) == 1
        assert route.call_count == 2


class TestFetchDetail:
    @respx.mock
    async def test_detail_encodes_id_in_path(self):
        encrypted = "NDg5Tk9YSU9GWCtuUHcrQzBvdVZSZz09"  # 含 + / = 等需编码字符的形态
        route = respx.get(f"{BASE}/articles/public/{quote(encrypted, safe='')}").mock(
            return_value=Response(200, json={"code": 0, "message": "ok", "data": {"id": encrypted, "content": "<p>x</p>"}})
        )
        client = BlogClient(BASE)
        detail = await client.fetch_article_detail(encrypted)
        await client.aclose()
        assert detail["content"] == "<p>x</p>"
        assert route.called


class TestListItem:
    def test_list_hash_changes_on_metadata_change(self):
        a = ArticleListItem.from_api(_item("x", "旧标题"))
        b = ArticleListItem.from_api(_item("x", "新标题"))
        assert a.list_hash() != b.list_hash()

    def test_list_hash_stable_regardless_of_label_order(self):
        item1 = _item("x")
        item2 = _item("x")
        item2["labels"] = list(reversed(item2["labels"]))
        assert ArticleListItem.from_api(item1).list_hash() == ArticleListItem.from_api(item2).list_hash()
