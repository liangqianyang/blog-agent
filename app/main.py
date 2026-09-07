"""FastAPI 入口：lifespan 装配 Qdrant/embedding/LLM/图，SSE 聊天与历史接口。

SSE 协议（data 恒为单行 JSON）：
    event: status   data: {"phase":"start","thread_id":"…"}   # 会话开始/改写检索词
    event: sources  data: {"sources":[{title,url,heading,score}]}
    event: delta    data: {"text":"…"}
    event: error    data: {"message":"…"}
    event: done     data: {"thread_id":"…"}
每次请求以且仅以一个终止事件（done 或 error）收尾。
"""

import asyncio
import json
import logging
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.embeddings import build_embeddings
from app.graph.builder import build_graph
from app.ingestion.indexer import ArticleIndexer
from app.llm import build_chat_model
from app.qdrant import build_qdrant, ensure_collection
from app.sync_service import sync_single

logger = logging.getLogger("blog_agent")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _cors_origins() -> list[str]:
    """blog-vue dev 源 + .env 里额外配置的源。"""
    settings = get_settings()
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    if settings.chat_cors_origins:
        origins.extend(o.strip() for o in settings.chat_cors_origins.split(",") if o.strip())
    return origins


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)

    qdrant: AsyncQdrantClient = build_qdrant(settings)
    indexer = ArticleIndexer(qdrant, build_embeddings(settings), settings)
    grader_llm = build_chat_model(settings, streaming=False)
    generator_llm = build_chat_model(settings, streaming=True)

    saver_cm = AsyncSqliteSaver.from_conn_string(f"{settings.data_dir}/checkpoints.sqlite")
    saver = await saver_cm.__aenter__()

    app.state.settings = settings
    app.state.qdrant = qdrant
    app.state.indexer = indexer
    app.state.graph = build_graph(indexer, grader_llm, generator_llm, settings, checkpointer=saver)
    try:
        await ensure_collection(qdrant, settings)
        logger.info("startup ok: qdrant=%s collection=%s", settings.qdrant_url, settings.qdrant_collection)
        yield
    finally:
        await saver_cm.__aexit__(None, None, None)
        await qdrant.close()


app = FastAPI(title="blog-agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Sync-Token"],
)


@app.get("/api/health")
async def health() -> dict:
    """存活检查：qdrant 可达性与点数。"""
    settings = get_settings()
    qdrant: AsyncQdrantClient | None = getattr(app.state, "qdrant", None)
    reachable = False
    points: int | None = None
    if qdrant is not None:
        try:
            reachable = await qdrant.collection_exists(settings.qdrant_collection)
            if reachable:
                points = (await qdrant.count(settings.qdrant_collection, exact=True)).count
        except Exception:  # noqa: BLE001 —— 健康检查要吞掉一切下游错误
            reachable = False
    return {"status": "ok", "env": settings.env_name, "qdrant": reachable, "collection": settings.qdrant_collection, "points": points}


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# ---------- 管理：后台同步（X-Sync-Token 保护） ----------

_sync_status: dict = {"running": False, "last_result": None, "last_run_at": None}


async def _run_sync() -> None:
    from scripts.sync_articles import run as run_sync

    try:
        result = await run_sync(full=False)
        _sync_status.update(running=False, last_result=result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("background sync failed")
        _sync_status.update(running=False, last_result={"error": str(exc)})
    finally:
        _sync_status["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")


@app.post("/api/admin/sync", status_code=202)
async def admin_sync(request: Request) -> dict:
    settings = request.app.state.settings
    token = request.headers.get("X-Sync-Token", "")
    if not secrets.compare_digest(token, settings.admin_sync_token):
        raise HTTPException(status_code=401, detail="invalid sync token")
    if _sync_status["running"]:
        return {"status": "already_running"}
    _sync_status["running"] = True
    asyncio.create_task(_run_sync())
    return {"status": "started"}


@app.get("/api/admin/sync/status")
async def admin_sync_status() -> dict:
    return _sync_status


class SyncArticleRequest(BaseModel):
    article_id: str = Field(min_length=8, max_length=256)  # 加密串
    action: Literal["index", "delete"] = "index"


@app.post("/api/admin/sync-article")
async def admin_sync_article(req: SyncArticleRequest, request: Request) -> dict:
    """单篇实时同步（博客后台 Observer 在新增/修改/删除后调用）。

    action=index 时若文章已下架（上游 404），自动转为删除向量点。
    """
    settings = request.app.state.settings
    token = request.headers.get("X-Sync-Token", "")
    if not secrets.compare_digest(token, settings.admin_sync_token):
        raise HTTPException(status_code=401, detail="invalid sync token")
    try:
        result = await sync_single(
            req.article_id,
            request.app.state.indexer,
            settings,
            delete=req.action == "delete",
        )
    except Exception as exc:  # noqa: BLE001 —— 同步失败返回明确错误，后台 Job 会重试
        logger.exception("single sync failed: %s", req.article_id)
        raise HTTPException(status_code=502, detail=f"sync failed: {type(exc).__name__}") from exc
    return result


# ---------- 限流：公开问答端点按 IP 简单滑窗（生产挂反代后应改用 X-Forwarded-For） ----------

_RATE_LIMIT = 20  # 次/分钟
_RATE_WINDOW = 60.0
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(client_ip: str) -> bool:
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]
    while bucket and now - bucket[0] > _RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT:
        return True
    bucket.append(now)
    return False


def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    if "Authentication" in name or "401" in str(exc):
        return "模型服务认证失败，请检查 API Key 配置"
    if "RateLimit" in name or "429" in str(exc):
        return "模型服务限流，请稍后重试"
    if "Timeout" in name:
        return "模型服务响应超时，请稍后重试"
    return f"服务暂时不可用（{name}），请稍后重试"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
    if _rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    graph = request.app.state.graph
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def gen():
        yield _sse("status", {"phase": "start", "thread_id": thread_id})
        try:
            stream = graph.astream(
                {"messages": [HumanMessage(content=req.message)], "query": req.message},
                config=config,
                stream_mode=["messages", "custom"],
            )
            async for mode, payload in stream:
                if mode == "messages":
                    chunk, meta = payload
                    if (
                        meta.get("langgraph_node") == "generate"
                        and isinstance(chunk, BaseMessage)
                        and isinstance(chunk.content, str)
                        and chunk.content
                    ):
                        yield _sse("delta", {"text": chunk.content})
                elif mode == "custom":
                    data = payload[1] if isinstance(payload, tuple) else payload
                    if isinstance(data, dict):
                        if data.get("type") == "sources":
                            yield _sse("sources", {"sources": data.get("sources", [])})
                        elif data.get("type") == "status":
                            yield _sse("status", {"phase": data.get("phase"), "query": data.get("query")})
            yield _sse("done", {"thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001 —— SSE 通道内错误必须转成事件而非断流
            logger.exception("chat failed: thread=%s", thread_id)
            yield _sse("error", {"message": _friendly_error(exc)})

    return EventSourceResponse(gen(), ping=15)


@app.get("/api/search")
async def agent_search(
    request: Request,
    q: str = Query(min_length=1, max_length=400),
    top_k: int = Query(default=8, ge=1, le=20),
) -> dict:
    """语义检索：问题 → 向量 → Qdrant top_k，返回命中片段（不走 LLM，毫秒级）。"""
    if _rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    indexer: ArticleIndexer = request.app.state.indexer
    points = await indexer.search(q, top_k)
    results = []
    for p in points:
        payload = p.payload or {}
        text = payload.get("text", "")
        results.append(
            {
                "title": payload.get("article_title", ""),
                "url": payload.get("article_url", ""),
                "heading": payload.get("heading_path", ""),
                "score": round(p.score, 3),
                "text": text[:220] + ("…" if len(text) > 220 else ""),
                "category": payload.get("category"),
                "labels": payload.get("labels", []),
                "published_at": payload.get("published_at"),
            }
        )
    return {"query": q, "results": results}


@app.get("/api/history/{thread_id}")
async def history(thread_id: str, request: Request) -> dict:
    """恢复会话。sources 只标注到最后一条 assistant 消息（状态里仅存最近一轮来源）。"""
    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    messages: list[BaseMessage] = values.get("messages", [])
    sources: list[dict] = values.get("sources", [])

    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content, "sources": []})
        else:
            continue
    for item in reversed(out):
        if item["role"] == "assistant":
            item["sources"] = sources
            break
    return {"thread_id": thread_id, "messages": out}
