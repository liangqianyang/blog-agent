"""Embedding 客户端：OpenAI 兼容接口，换供应商只改本文件。

智谱 embedding-3 约束（已核验官方文档）：/embeddings 端点，dimensions ∈ {256,512,1024,2048}，
单条输入 ≤3072 token，单请求 ≤64 条。批次取 32 留余量。
"""

import asyncio

from langchain_openai import OpenAIEmbeddings

from app.config import Settings, get_settings

_BATCH_SIZE = 32
_BATCH_PAUSE = 0.3


def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_api_base_url,
        api_key=settings.embedding_api_key,
        dimensions=settings.embedding_dims,
        # 关闭 tiktoken 分片逻辑：自定义端点（智谱等）不走 OpenAI tokenizer
        check_embedding_ctx_length=False,
    )


async def embed_texts(embeddings: OpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
    """分批 embedding，批间小睡以守供应商限速。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vectors.extend(await embeddings.aembed_documents(batch))
        if i + _BATCH_SIZE < len(texts):
            await asyncio.sleep(_BATCH_PAUSE)
    return vectors


async def embed_query(embeddings: OpenAIEmbeddings, query: str) -> list[float]:
    return await embeddings.aembed_query(query)
