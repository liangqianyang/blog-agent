"""文章索引器：切片 → embedding → Qdrant upsert，以及检索与删除。"""

import logging
import uuid
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config import Settings, get_settings
from app.embeddings import embed_query, embed_texts
from app.ingestion.chunker import Chunk, chunk_html
from app.ingestion.html_cleaner import clean

logger = logging.getLogger("blog_agent.indexer")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def point_id(article_id: str, chunk_index: int) -> str:
    """确定性点 ID：同一文章同一序号重复 upsert 幂等覆盖。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"blog:{article_id}:{chunk_index}"))


def build_embedding_text(title: str, heading_path: str, text: str) -> str:
    """喂给 embedding 的文本：文章标题 + 面包屑 + 正文，让向量带上文章语境。"""
    parts = [p for p in (title, heading_path, text) if p]
    return "\n".join(parts)


class ArticleIndexer:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embeddings: OpenAIEmbeddings,
        settings: Settings | None = None,
    ):
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.settings = settings or get_settings()

    async def index_article(self, detail: dict) -> int:
        """重建单篇文章的全部向量点，返回 chunk 数。

        先删后插：文章缩短时避免残留旧尾巴点。
        """
        s = self.settings
        article_id = str(detail["id"])
        title = detail.get("title") or ""
        chunks: list[Chunk] = chunk_html(clean(detail.get("content") or ""))

        await self.delete_article(article_id)
        if not chunks:  # 空文章也upsert不出点，但状态表会记录，避免每轮都重拉
            return 0

        texts = [build_embedding_text(title, c.heading_path, c.text) for c in chunks]
        vectors = await embed_texts(self.embeddings, texts)

        category = detail.get("category") or {}
        labels = [lb.get("title") for lb in (detail.get("labels") or []) if lb.get("title")]
        payload_common = {
            "article_id": article_id,
            "article_title": title,
            "article_url": f"/article/{article_id}",
            "published_at": detail.get("published_at"),
            "category": category.get("name"),
            "category_id": category.get("id"),
            "labels": labels,
            "synced_at": _now_iso(),
        }
        points = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            payload = {
                **payload_common,
                "heading": c.heading,
                "heading_path": c.heading_path,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "text": c.text,
                "kind": c.kind,
            }
            points.append(PointStruct(id=point_id(article_id, i), vector=v, payload=payload))
        await self.qdrant.upsert(collection_name=s.qdrant_collection, points=points, wait=True)
        logger.info("indexed %s (%d chunks): %s", article_id, len(chunks), title)
        return len(chunks)

    async def delete_article(self, article_id: str) -> None:
        await self.qdrant.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
            ),
            wait=True,
        )

    async def search(self, query: str, top_k: int | None = None) -> list:
        """向量检索 top_k，返回 ScoredPoint 列表。"""
        s = self.settings
        vector = await embed_query(self.embeddings, query)
        resp = await self.qdrant.query_points(
            collection_name=s.qdrant_collection,
            query=vector,
            limit=top_k or s.retrieval_top_k,
            with_payload=True,
        )
        return resp.points

    @staticmethod
    def to_documents(points: list) -> list[Document]:
        """ScoredPoint → LangChain Document（retrieve 节点用）。"""
        docs = []
        for p in points:
            payload = p.payload or {}
            docs.append(
                Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        "article_id": payload.get("article_id", ""),
                        "article_title": payload.get("article_title", ""),
                        "article_url": payload.get("article_url", ""),
                        "heading_path": payload.get("heading_path", ""),
                        "score": p.score,
                    },
                )
            )
        return docs
