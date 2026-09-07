"""同步状态库与单篇文章同步（CLI 增量与 HTTP 实时接口共用）。"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.blog_client import ArticleListItem, BlogClient, BlogApiError
from app.config import Settings
from app.ingestion.indexer import ArticleIndexer

logger = logging.getLogger("blog_agent.sync")

_STATE_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id  TEXT PRIMARY KEY,
    list_hash   TEXT NOT NULL,
    chunk_count INTEGER,
    synced_at   TEXT
)
"""


def open_state_db(settings: Settings) -> sqlite3.Connection:
    path = Path(settings.data_dir) / "sync_state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute(_STATE_DDL)
    db.commit()
    return db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def sync_single(article_id: str, indexer: ArticleIndexer, settings: Settings, *, delete: bool = False) -> dict:
    """实时同步单篇文章（后台 Observer 触发）。

    - delete=False：拉详情重建向量，并更新状态哈希（下次增量不再重复拉取）
    - 文章已下架/不存在（上游 404）时自动转为删除向量点
    - delete=True：直接删除该文章全部向量点与状态行
    """
    db = open_state_db(settings)
    blog = BlogClient(settings.blog_api_base)
    try:
        if delete:
            await indexer.delete_article(article_id)
            db.execute("DELETE FROM articles WHERE article_id = ?", (article_id,))
            db.commit()
            logger.info("single sync: deleted %s", article_id)
            return {"status": "deleted", "article_id": article_id}

        try:
            detail = await blog.fetch_article_detail(article_id)
        except BlogApiError as exc:
            if exc.code == 404:  # 已下架或删除：清掉向量
                await indexer.delete_article(article_id)
                db.execute("DELETE FROM articles WHERE article_id = ?", (article_id,))
                db.commit()
                logger.info("single sync: %s not found upstream, deleted points", article_id)
                return {"status": "deleted", "article_id": article_id, "reason": "not_found"}
            raise

        chunks = await indexer.index_article(detail)
        item = ArticleListItem.from_api(detail)
        db.execute(
            "INSERT OR REPLACE INTO articles (article_id, list_hash, chunk_count, synced_at) VALUES (?, ?, ?, ?)",
            (item.id, item.list_hash(), chunks, now_iso()),
        )
        db.commit()
        logger.info("single sync: indexed %s (%d chunks): %s", article_id, chunks, item.title)
        return {"status": "ok", "article_id": article_id, "title": item.title, "chunks": chunks}
    finally:
        await blog.aclose()
        db.close()
