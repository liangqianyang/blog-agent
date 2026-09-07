"""文章同步：增量为主（列表哈希对比），--full 全量重建。

用法：
    uv run python -m scripts.sync_articles           # 增量
    uv run python -m scripts.sync_articles --full    # 全量（改完文章正文后手动跑）

已知局限：列表接口无 updated_at，正文改动但标题/摘要/分类/标签未变时增量检测不到 → 跑 --full。
"""

import argparse
import asyncio
import json
import logging

from app.blog_client import BlogClient
from app.config import get_settings
from app.embeddings import build_embeddings
from app.ingestion.indexer import ArticleIndexer
from app.qdrant import build_qdrant, ensure_collection
from app.sync_service import now_iso, open_state_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("blog_agent.sync")


async def run(full: bool) -> dict:
    settings = get_settings()
    db = open_state_db(settings)
    stored: dict[str, str] = dict(db.execute("SELECT article_id, list_hash FROM articles").fetchall())

    blog = BlogClient(settings.blog_api_base)
    qdrant = build_qdrant(settings)
    await ensure_collection(qdrant, settings)
    indexer = ArticleIndexer(qdrant, build_embeddings(settings), settings)

    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "details_fetched": 0}
    try:
        items = await blog.fetch_public_articles(per_page=settings.sync_per_page)
        logger.info("fetched list: %d articles", len(items))
        current_ids = set()

        for item in items:
            current_ids.add(item.id)
            h = item.list_hash()
            old = stored.get(item.id)
            if old == h and not full:
                stats["unchanged"] += 1
                continue

            detail = await blog.fetch_article_detail(item.id)
            stats["details_fetched"] += 1
            n = await indexer.index_article(detail)
            stats["added" if old is None else "updated"] += 1
            db.execute(
                "INSERT OR REPLACE INTO articles (article_id, list_hash, chunk_count, synced_at) VALUES (?, ?, ?, ?)",
                (item.id, h, n, now_iso()),
            )
            db.commit()
            logger.info("%s: %s (%d chunks)", "added" if old is None else "updated", item.title, n)
            await asyncio.sleep(settings.sync_detail_delay)  # 守限流 + 少虚增 view_count

        for gone in sorted(set(stored) - current_ids):
            await indexer.delete_article(gone)
            db.execute("DELETE FROM articles WHERE article_id = ?", (gone,))
            stats["removed"] += 1
            logger.info("removed: %s", gone)
        db.commit()
    finally:
        await blog.aclose()
        await qdrant.close()
        db.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="全量重建（忽略哈希，重新拉详情与向量化）")
    args = parser.parse_args()
    stats = asyncio.run(run(args.full))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
