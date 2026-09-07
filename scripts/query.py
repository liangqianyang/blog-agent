"""检索 CLI：不经过 LLM，直接验证「问题 → 向量 → Qdrant → 命中」数据链路。

用法：uv run python -m scripts.query "Laravel 队列"
"""

import argparse
import asyncio

from app.config import get_settings
from app.embeddings import build_embeddings
from app.ingestion.indexer import ArticleIndexer
from app.qdrant import build_qdrant


async def run(query: str) -> None:
    settings = get_settings()
    qdrant = build_qdrant(settings)
    indexer = ArticleIndexer(qdrant, build_embeddings(settings), settings)
    try:
        points = await indexer.search(query)
        if not points:
            print("(no hits)")
            return
        for p in points:
            payload = p.payload or {}
            print(f"{p.score:.3f} | {payload.get('article_title')} | {payload.get('heading_path')}")
            snippet = (payload.get("text") or "").replace("\n", " ")[:120]
            print(f"       {snippet}")
    finally:
        await qdrant.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="向量检索调试")
    parser.add_argument("query", help="检索词")
    args = parser.parse_args()
    asyncio.run(run(args.query))


if __name__ == "__main__":
    main()
