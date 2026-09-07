"""Qdrant 客户端装配与 collection 保障。"""

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import Settings, get_settings

logger = logging.getLogger("blog_agent.qdrant")


def build_qdrant(settings: Settings | None = None) -> AsyncQdrantClient:
    settings = settings or get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=10)


async def ensure_collection(client: AsyncQdrantClient, settings: Settings | None = None) -> None:
    """不存在则建 collection（cosine + 配置维度），并给 article_id 建 keyword 索引。

    若已存在但维度与 EMBEDDING_DIMS 不符，直接抛错：换 embedding 模型必须重建 collection，
    旧向量与新模型不兼容。
    """
    settings = settings or get_settings()
    name = settings.qdrant_collection
    if not await client.collection_exists(name):
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.embedding_dims, distance=Distance.COSINE),
        )
        logger.info("created qdrant collection %s (dims=%d)", name, settings.embedding_dims)
    else:
        info = await client.get_collection(name)
        size = info.config.params.vectors.size
        if size != settings.embedding_dims:
            raise RuntimeError(
                f"qdrant collection {name} 向量维度为 {size}，与 EMBEDDING_DIMS={settings.embedding_dims} 不符；"
                f"换 embedding 模型后请删除重建：curl -X DELETE {settings.qdrant_url}/collections/{name}"
            )
    await client.create_payload_index(
        collection_name=name, field_name="article_id", field_schema="keyword"
    )
