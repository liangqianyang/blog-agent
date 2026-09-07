"""应用配置：按 APP_ENV 加载对应 env 文件（OS 环境变量 > .env.{APP_ENV} > .env > 默认值）。

用法：
    开发（默认）：uv run uvicorn app.main:app --port 8000
    生产：        APP_ENV=production uv run uvicorn app.main:app --port 8000
"""

from functools import lru_cache
from os import getenv
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    # 当前环境标识（来自 APP_ENV，仅用于展示，不参与加载逻辑）
    env_name: str = "development"

    # 数据源
    blog_api_base: str = "http://www.blog.test/api"

    # 对话 LLM（OpenAI 兼容）
    chat_api_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    chat_api_key: str = ""
    chat_model: str = "glm-4-flash"
    chat_temperature: float = 0.3

    # Embedding（OpenAI 兼容 /embeddings）
    embedding_api_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    embedding_api_key: str = ""
    embedding_model: str = "embedding-3"
    embedding_dims: int = 1024

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "blog_articles"
    retrieval_top_k: int = 5

    # 问答图
    max_rewrites: int = 2
    min_relevant_docs: int = 1

    # 同步
    sync_per_page: int = 50
    sync_detail_delay: float = 0.6

    # 服务
    data_dir: str = "./data"
    admin_sync_token: str = "change-me"
    chat_cors_origins: str = ""


def _pick_env_file(env: str) -> str:
    """存在则用 .env.{env}，否则回退 .env（都没有则用纯默认值/OS 环境变量）。"""
    for candidate in (f".env.{env}", ".env"):
        if Path(candidate).is_file():
            return candidate
    return candidate  # .env 不存在时返回路径无害：pydantic 会忽略缺失文件


@lru_cache
def get_settings() -> Settings:
    env = getenv("APP_ENV", "development").strip().lower() or "development"
    return Settings(env_name=env, _env_file=_pick_env_file(env))
