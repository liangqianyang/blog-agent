# blog-agent

基于博客文章的 RAG 问答助手：LangChain + LangGraph + Qdrant + FastAPI。

**[→ 生产部署手册](docs/deploy.md)**（Docker Compose + nginx 反代 + Laravel 侧接入，含验收清单）

从博客后台（blog-admin-v2）的公开 API 同步文章，向量化后存入 Qdrant，
用 LangGraph 组装「检索 → 判分 → 生成 / 改写重检」的问答图，
通过 SSE 流式输出给博客前台（blog-vue）的聊天组件。

```
blog-vue（浮窗 + /chat 页面）
   │  SSE 流式（fetch POST）
   ▼
blog-agent（本服务，FastAPI）
   ├─ 同步管道：博客公开 API → HTML 清洗 → 结构切片 → embedding → Qdrant
   ├─ 问答图（LangGraph）：
   │     START → retrieve → grade_documents ─┬ 相关 → generate → END
   │                  ↑                       ├─ 不相关且未达上限 → rewrite_query → retrieve
   │                  └───────────────────────┴─ 兜底 → generate → END
   └─ 多轮记忆：AsyncSqliteSaver checkpointer（按 thread_id 持久化）
   ▼
Qdrant（向量库） + LLM API（智谱 GLM-4-Flash，OpenAI 兼容可切换）
```

## 快速开始

```bash
uv sync                                   # 安装依赖（pin Python 3.12）
cp .env.example .env.development          # 开发配置，填 CHAT_API_KEY / EMBEDDING_API_KEY
cp .env.example .env.production           # 生产配置，部署时填真实值

uv run python -m scripts.sync_articles        # 全量同步文章（首次）
uv run python -m scripts.sync_articles --full # 改完文章正文后手动全量重建

uv run uvicorn app.main:app --reload --port 8000                    # 开发（默认读 .env.development）
APP_ENV=production uv run uvicorn app.main:app --port 8000          # 生产（读 .env.production）
curl http://127.0.0.1:8000/api/health                               # 应返回 env 和 points 数
```

**环境配置规则**（优先级从高到低）：OS 环境变量 > `.env.{APP_ENV}` > `.env` > 内置默认值。
开发/生产的差异主要在：`BLOG_API_BASE`（本地/线上博客）、`QDRANT_URL`、`ADMIN_SYNC_TOKEN`（生产用强随机串）、`CHAT_CORS_ORIGINS`（生产填博客前台域名，走 nginx 同域反代则不用）。

## 常用命令

| 命令 | 说明 |
|---|---|
| `uv run python -m scripts.sync_articles` | 增量同步（列表哈希对比，无变化则零详情请求） |
| `uv run python -m scripts.sync_articles --full` | 全量重建（约 1 分钟） |
| `uv run python -m scripts.query "关键词"` | 检索调试（不经过 LLM） |
| `uv run pytest -q` | 单测（切片器 + 博客客户端） |

## 实时同步（推荐）

博客后台（blog-admin-v2）的 `ArticleObserver` 在文章新增/修改/删除后派发
`SyncArticleToBlogAgent` 队列任务，主动调用本服务的单篇同步接口：

```
后台保存文章 → Observer → 队列 Job → POST /api/admin/sync-article → 重建该篇向量
```

- 修改正文也能实时同步（不依赖增量哈希，绕过了列表接口无 updated_at 的盲区）
- 文章下架后详情接口返回 404，agent 自动转为删除该篇向量点
- 幂等：同一篇重复同步只是覆盖（uuid5 确定点 ID，先删后插）

Laravel 侧配置（blog-admin-v2 的 .env）：

```ini
BLOG_AGENT_ENABLED=true
BLOG_AGENT_URL=http://host.docker.internal:8000   # PHP 跑在 docker 里时用此地址访问宿主机服务
BLOG_AGENT_SYNC_TOKEN=change-me                    # 与本服务 .env 的 ADMIN_SYNC_TOKEN 一致
```

前提：Laravel 队列 worker 需监听专用队列（Job 固定跑在 `blog-agent` 队列，与 default 隔离——
本地开发直连生产库时，服务器上的 worker 也在消费 default 队列，共享会互相偷任务）：

```bash
php artisan queue:work --queue=blog-agent --tries=3 --sleep=1
# 部署到服务器后：php artisan queue:work --queue=default,blog-agent
```

注意：Observer 已做脏字段过滤（title/content/summary/status/分类等变化才同步），
纯 view_count 变化（公开页浏览）不触发——否则 agent 拉详情 → 浏览量+1 → 再触发同步，形成死循环。

## API

| 端点 | 说明 |
|---|---|
| `POST /api/chat` | SSE 流式问答，body `{thread_id?, message}`；事件序列 status → sources → delta* → done/error |
| `GET /api/search?q=&top_k=` | 语义检索（不走 LLM）：返回命中的文章片段/章节/相关度，供前端检索页 |
| `GET /api/history/{thread_id}` | 恢复会话历史 |
| `POST /api/admin/sync` | 触发后台全量增量同步，头 `X-Sync-Token`（.env 的 ADMIN_SYNC_TOKEN） |
| `POST /api/admin/sync-article` | 单篇实时同步，body `{article_id: 加密串, action: index\|delete}`，博客 Observer 调用 |
| `GET /api/admin/sync/status` | 同步任务状态 |
| `GET /api/health` | 健康检查（qdrant 可达性 + 点数） |

问答端点限流 20 次/分钟/IP。

## 配置要点（.env）

- **换 LLM 供应商**：改 `CHAT_API_BASE_URL / CHAT_API_KEY / CHAT_MODEL`（OpenAI 兼容均可）
- **换 embedding**：改 `EMBEDDING_*` 四项。**注意：换模型必须重建 collection**
  （维度与向量空间不兼容）：`curl -X DELETE localhost:6333/collections/blog_articles` 后重新 sync，
  启动时也会自动校验维度防止误用
- **数据源**：`BLOG_API_BASE` 默认本地 Laravel（详情接口每次调用会 +1 浏览量，污染留在本地）

## 已知局限与设计取舍

1. **增量同步盲区**：博客列表接口无 `updated_at`，正文改动但标题/摘要/分类/标签未变时检测不到
   —— 改完文章跑一次 `--full`（62 篇量级约 1 分钟）
2. **结构化输出容错**：glm-4-flash 的 function calling 参数易畸形（实测输出 `[1]\n[2]` 或裸数组），
   判分/改写用 `method="json_mode"` + pydantic `model_validator(mode="before")` 宽容解析，
   仍失败则降级（判分保守保留全部文档）
3. **多轮来源标注**：checkpointer 状态里只存最近一轮 sources，历史接口仅给最后一条 assistant 消息标注引用
4. **流式 markdown 半成品**：未闭合代码栅栏在流式过程中渲染会短暂异常，属全量重渲染方案的固有权衡

## 目录结构

```
app/
├── main.py             # FastAPI：lifespan、SSE chat、history、admin sync、限流
├── config.py           # pydantic-settings 配置
├── blog_client.py      # 博客公开 API 客户端（分页/详情/包络解析）
├── llm.py              # ChatOpenAI 工厂（OpenAI 兼容 base_url）
├── embeddings.py       # embedding 客户端（换供应商的唯一改动点）
├── qdrant.py           # Qdrant 装配 + collection 维度校验
├── ingestion/
│   ├── html_cleaner.py # HTML 去噪
│   ├── chunker.py      # 标题结构切片（代码块/表格原子）
│   └── indexer.py      # uuid5 幂等 upsert / 检索 / 删除
└── graph/
    ├── state.py        # AgentState（messages + query + documents + sources）
    ├── nodes.py        # retrieve / grade / rewrite / generate 节点
    └── builder.py      # 图组装 + 条件路由
scripts/
├── sync_articles.py    # 同步 CLI（增量/全量，sqlite 状态表）
└── query.py            # 检索调试 CLI
```

## 后续方向（未实施）

- generate 后加 groundedness 检查节点（幻觉拦截，图结构已支持）
- 混合检索：复用博客 ES 关键词接口 + RRF 融合向量结果
- 问题分类：闲聊跳过检索
- highlight.js 动态加载（前端代码高亮）
- 定时增量同步（launchd/cron 调 `POST /api/admin/sync`）
