# blog-agent 教学文档

> 面向读者的学习笔记：这个项目为什么这样做、每块代码在干什么、踩过哪些坑。
> 配合源码阅读，按顺序看完相当于完整走一遍「生产级 RAG 系统」的设计与实现。

---

## 0. 这个项目解决什么问题

你有一个写了 45 篇技术文章的博客。想让访客"问博客"而不是"搜博客"：

| 传统方案的问题 | RAG 的解法 |
|---|---|
| 关键词搜索：搜「数组去掉重复元素」找不到标题叫「PHP数组去重详解」的文章 | **语义检索**：问题被向量化，和内容的语义距离匹配，同义不同词也能命中 |
| 直接问大模型：它没读过你的博客，会一本正经地胡编（幻觉） | **检索增强**：先从你的文章里捞出相关片段，让模型「看着材料回答」并标注引用 |
| 把全站文章塞进上下文：太长、太贵 | **向量库**：只取最相关的 top-k 片段（本项目 k=5） |

这就是 **RAG（Retrieval-Augmented Generation，检索增强生成）**。

## 1. 原理速成：从一句话到一个回答

```
「怎么给 PHP 数组去重？」
   │ ① embedding 模型把问题变成 1024 维向量
   ▼
[0.021, -0.083, 0.156, ...]  ←—— 与库里每个文章片段的向量算余弦相似度
   │ ② 取最相似的 5 个片段                    （Qdrant 毫秒级完成）
   ▼
「PHP数组去重详解 / 二、PHP数组去重方法 / array_unique...」
   │ ③ 片段 + 对话历史 + 问题拼成提示词
   ▼
④ LLM 流式生成回答（带 [1][2] 引用标注）
```

三个核心概念：

- **Embedding（嵌入）**：把文本映射成高维空间里的一个点，语义相近的文本距离近。本项目用智谱 `embedding-3`（1024 维），输入 `<标题+章节路径+正文>` 而非裸正文——让每个片段都"自带"文章语境。
- **余弦相似度**：衡量两个向量方向是否一致（0~1），不关心长度。适合文本语义匹配。
- **Chunking（切片）**：整篇文章太长，embedding 会"稀释"重点，检索粒度也太粗。切成 500 字左右的小块，命中即命中到"章节"级别。

## 2. 技术栈总览

| 层 | 技术 | 角色 | 为什么选它 |
|---|---|---|---|
| 语言/运行时 | Python 3.12 + uv | 服务实现 + 包管理 | AI 生态标准；uv 秒级装依赖、锁文件可复现（pin 3.12 是因为当时 3.14 与 LangChain 生态有兼容警告） |
| Web 框架 | FastAPI + uvicorn | HTTP/SSE 服务 | 原生异步（和 httpx/Qdrant 客户端同一套 asyncio 体系）、类型驱动、自动文档 |
| 配置 | pydantic-settings | .env 加载与类型校验 | 带类型提示的配置（`embedding_dims: int`），错了启动就报 |
| HTTP 客户端 | httpx | 调博客 API | 异步版的 requests，超时/重试好控制 |
| HTML 解析 | BeautifulSoup + lxml | 清洗文章 HTML | 容错好、lxml 快；切片要按 DOM 结构走 |
| LLM 编排 | LangChain（core + openai） | 模型调用抽象 | `ChatOpenAI(base_url=...)` 一套代码对接所有 OpenAI 兼容服务商（智谱/DeepSeek/…） |
| Agent 图 | LangGraph | 问答流程编排 | 把「检索→判分→改写重检→生成」写成带条件边的状态图，天然支持循环、多轮记忆、流式事件 |
| 向量数据库 | Qdrant | 存向量 + 相似度检索 | 轻量单容器可跑、payload 过滤/索引完善；本项目量级（几百个点）其实是"杀鸡用牛刀"，但学习价值正在于见到完整能力 |
| 流式传输 | sse-starlette | SSE 协议实现 | 处理好帧格式、心跳、客户端断连、nginx 兼容头，这些细节自己写容易漏 |
| 状态存储 | SQLite ×2 | 同步状态 + 会话检查点 | 单文件零运维；量级完全够，"用对工具"而不是"用新工具" |
| 部署 | Docker 多阶段 + compose | 交付 | 与 Qdrant 一把起；镜像只含运行时 |
| 配套前端 | Vue3 + Pinia + fetch SSE + markdown-it + DOMPurify | 聊天界面 | （另两个项目，见 §8） |
| 配套后端 | Laravel Observer + Queue Job | 实时同步触发 | （见 §8） |

## 3. 目录结构导览

```
blog-agent/
├── pyproject.toml            # 依赖清单 + uv 清华镜像 + pytest/ruff 配置
├── .env.development          # 开发配置（本地博客、本地 Qdrant）
├── .env.production           # 生产配置（线上博客、强 token）
├── Dockerfile                # 多阶段构建：builder 装依赖 → 运行时只拷产物
├── docker-compose.yml        # qdrant + agent 两个服务，双网络
├── app/
│   ├── main.py               # ★ FastAPI 入口：装配、路由、SSE、限流
│   ├── config.py             # ★ 多环境配置加载
│   ├── blog_client.py        # ★ 博客 API 客户端
│   ├── llm.py                # 对话模型工厂
│   ├── embeddings.py         # ★ embedding 客户端（换供应商唯一改动点）
│   ├── qdrant.py             # 连接 + collection 维度校验
│   ├── sync_service.py       # ★ 单篇实时同步 + 状态库
│   ├── ingestion/            # ★ 摄取管道（数据 → 向量）
│   │   ├── html_cleaner.py
│   │   ├── chunker.py        # ★★ 全项目对检索质量影响最大的文件
│   │   └── indexer.py        # ★★ 向量入库/检索/删除
│   └── graph/                # ★ LangGraph 问答图
│       ├── state.py
│       ├── nodes.py          # ★★ 四个节点 + 提示词 + 结构化输出容错
│       └── builder.py        # 图的接线
├── scripts/
│   ├── sync_articles.py      # 增量/全量同步 CLI
│   └── query.py              # 检索调试 CLI（不经过 LLM）
├── tests/                    # 切片器与客户端单测
└── docs/
    ├── deploy.md             # 部署手册
    └── tutorial.md           # 本文
```

（★ 越多越值得精读）

---

## 4. 核心模块精讲

### 4.1 config.py —— 多环境配置

**问题**：本地开发连本地博客/Qdrant，生产连线上，密钥不能进 git。

**方案**：优先级链 `OS 环境变量 > .env.{APP_ENV} > .env > 代码默认值`：

```python
@lru_cache
def get_settings() -> Settings:
    env = getenv("APP_ENV", "development").strip().lower()
    return Settings(env_name=env, _env_file=_pick_env_file(env))
```

知识点：
- `lru_cache` 让配置只解析一次（读文件 IO 只发生一次，进程内单例）
- `@lru_cache` + `_env_file` 参数是 pydantic-settings 的官方玩法：实例化时指定加载哪个文件
- OS 变量优先于文件——容器平台（compose 的 `environment:`）注入的值能覆盖文件，生产换 Qdrant 地址不用改文件
- 副作用：`health` 接口返回 `env` 字段，运维一眼确认跑的是哪套配置

### 4.2 blog_client.py —— 对接外部 API 的标准姿势

**问题**：博客是 Laravel 写的，响应有自己的包络 `{code:0, message, data}`，文章 id 是 AES 加密串，还有 120 次/分钟限流。

四个值得学的点：

1. **包络判断**：成功 iff `code == 0`，否则抛携带业务码的异常——把"对方的约定"封装在一个类里，图和管道完全不用关心
2. **重试**：`TransportError`（网络层）和 5xx 才重试（2 次、退避 0.5s）；4xx 不重试——重试解决不了"文章不存在"
3. **加密 id 当不透明字符串**：`quote(encrypted_id, safe='')` 再拼路径，永远不解析它的内容，只把它当稳定主键用（确定性加密 = 同一篇文章永远同一个串）
4. **分页保护**：`while page <= last_page` 外加 100 页硬上限，防上游数据异常时无限翻页

### 4.3 ingestion/ —— 摄取管道（本项目的心脏）

数据流水线：**HTML → 清洗 → 切片 → 向量化 → 入库**。

#### html_cleaner.py（30 行，作用却关键）

删掉 `script/style/iframe/form` 等噪音标签和注释。为什么不直接 strip_tags？因为**切片需要 DOM 结构**（标题层级），先清洗再解析，后面 chunker 拿到的树是干净的。

#### chunker.py —— 结构感知切片 ★★

**问题**：朴素按字数切片会把代码块拦腰斩断、把"标题"和"内容"分家，检索命中一段没有上下文的碎肉。

**算法**（两个阶段）：

```
阶段一：按标题分节
  遍历 DOM 顶层元素，维护一个标题栈 [h1, h2, h3, h4]
  遇到 h2 → 收口当前节，弹栈到比 h2 浅的层级，压入 "安装"
  遇到 h3 → 压入 "配置环境变量" → heading_path = "安装 > 配置环境变量"
  第一个标题之前的内容 = preamble（heading 为空）

阶段二：节内打包
  段落累积到 ~500 字就收口；单段超 1200 字按句号硬拆
  <pre> 代码块、<table> 是原子块：永远不与散文混合、不被斩断
  代码块超 3000 字（embedding 输入上限 3072 token）才按空行拆
```

为什么这些数字：**500** 是"一个完整语义单元"的经验值；**1200** 防止极端长段撑爆 embedding；**3000** 对齐智谱接口的 token 限制。切片策略对检索质量的影响大于模型选择——这是 RAG 工程的第一课。

#### indexer.py —— 入库、检索、删除 ★★

三个设计点：

1. **幂等点 ID**：
   ```python
   uuid.uuid5(uuid.NAMESPACE_URL, f"blog:{article_id}:{chunk_index}")
   ```
   uuid5 是**确定性**哈希——同一篇文章的第 N 块永远同一个 ID，重复同步 = 覆盖而非追加，不产生重复数据。
2. **先删后插**：文章从 11 块改成 8 块时，只靠幂等 upsert 会残留旧的 9/10/11 三块尾巴，所以每篇先按 `article_id` 过滤器删干净再插。
3. **payload 设计**：向量旁边存完整元数据（标题/章节/分类/标签/原文/原文路径），检索结果直接可用，前端引用 chips 的数据就来自这里——**向量库不只是"存向量"，它同时是这块数据的文档存储**。

#### embeddings.py —— 供应商隔离层

整个项目唯一 import `OpenAIEmbeddings` 的文件。换 embedding 供应商（智谱 → SiliconFlow bge-m3）只改这一个文件 + 重建 collection。两个防坑参数：
- `dimensions=1024`：显式指定维度（默认 2048，减半省内存）
- `check_embedding_ctx_length=False`：关闭 langchain 用 OpenAI tokenizer 预分片的逻辑——智谱不走这套，开了会请求两次

批处理：32 条/批（接口上限 64，留余量）+ 批间 sleep 0.3s（守限速）。

### 4.4 graph/ —— LangGraph 问答图 ★★

**为什么用"图"而不是顺序执行的函数**？因为流程**有条件分支和循环**：

```
START → retrieve → grade_documents ─┬ 相关≥1 → generate → END
              ↑                      ├─ 全不相关 且 改写<2次 → rewrite_query → 回 retrieve
              └──────────────────────┴─ 兜底 → generate（声明"博客没有相关内容"）→ END
```

如果用 if/else 写在函数里，多轮记忆、流式事件、节点级重试都要自己造；LangGraph 把这些变成声明式配置。

#### state.py —— 状态即数据库

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ★ 多轮记忆
    query: str            # 当前检索词（可能被改写过；生成回答用用户原话）
    documents: list[Document]
    sources: list[dict]
    rewrite_count: int    # 防死循环的保险丝
```

`add_messages` 是 reducer：新消息**合并**进列表而不是覆盖。配合 checkpointer，`messages` 就是持久化的对话历史——多轮对话在这个框架里不是"自己维护一个 list"，而是状态的一部分。

#### nodes.py —— 四个节点

| 节点 | 干什么 | 学习点 |
|---|---|---|
| `retrieve` | query 向量化 → Qdrant top-5 → 转 Document | 检索是"节点"，可被替换成混合检索而不动图 |
| `grade_documents` | 一次 LLM 调用批量判断 5 个文档哪些相关 | Agentic RAG 的核心：**对检索结果做质量把关**，防止垃圾上下文诱发幻觉 |
| `rewrite_query` | 检索落空时改写检索词（最多 2 次） | 自我纠错循环；`rewrite_count` 保险丝防死循环 |
| `generate` | 系统提示（含编号参考材料）+ 历史 → 流式生成 | 上下文只进本次调用的 system，**不污染持久化历史** |

两个最有价值的实战技巧（都是被 glm-4-flash 教做人的）：

**① 结构化输出的容错**。小模型的 function calling 会输出畸形参数（实测返回过 `'[1]\n[2]\n[3]'` 这种），解法是三层防御：

```python
structured = grader_llm.with_structured_output(GradeDocuments, method="json_mode")  # ① 换 json 模式

class GradeDocuments(BaseModel):
    relevant_indices: list[int] = Field(default_factory=list)
    @model_validator(mode="before")        # ② 宽容解析：裸数组 [1,2] 也包装成对象
    @classmethod
    def _coerce(cls, data): ...

try: ... except Exception: return {"documents": docs}  # ③ 终极降级：判分失败就全保留
```

**② 节点内发事件**。`get_stream_writer()({"type": "sources", ...})` 把引用来源**在生成开始前**推给前端——用户先看到"参考了哪些文章"，再看到回答逐字出现。

#### builder.py —— 接线

条件路由是唯一有点绕的部分：

```python
graph.add_conditional_edges(
    "grade_documents",
    lambda state: _route_after_grade(state, min_relevant, max_rewrites),
    {"generate": "generate", "rewrite": "rewrite_query"},
)
```

路由函数返回字符串 → 映射表决定去哪个节点。`graph.get_graph().draw_ascii()` 能打印 ASCII 图（装 grandalf），调试时很直观。

### 4.5 main.py —— 服务入口

#### lifespan：装配车间

所有重对象（httpx 客户端、Qdrant、embedding、两个 LLM、checkpointer、图）在**启动时**建好挂到 `app.state`，请求里零构建成本；关闭时统一释放。FastAPI 的 lifespan 是"应用级单例"的正确位置。

#### SSE：为什么是这个协议

| 方案 | 否决理由 |
|---|---|
| WebSocket | 双向能力用不上，握手/心跳/反代配置都更重 |
| 轮询 | 流式打字机效果做不了 |
| `EventSource` | 只能 GET、不能带 body 和自定义头——聊天要 POST JSON |
| **fetch + POST SSE** ✓ | 单向流式恰好匹配"生成型回答"的形状 |

事件协议（前端解析的依据）：

```
event: status   data: {"phase":"start","thread_id":...}   ← 会话开始/改写检索词
event: sources  data: {"sources":[{title,url,heading,score}]}  ← 生成前的引用
event: delta    data: {"text":"在 PHP 中"}                 ← token 增量（几百次）
event: done|error  data: {...}                            ← 有且仅有一个终止事件
```

流式机制：`graph.astream(..., stream_mode=["messages","custom"])` 同时订阅两种流——`messages` 模式吐模型 token（按 `metadata.langgraph_node=="generate"` 过滤，判分节点的内部调用不会漏出来），`custom` 模式吐我们自己发的事件。

#### 限流：10 行的滑动窗口

`deque` 存时间戳，进一个删过期的一个，超 20 个/分钟就 429。公开无认证端点的最低配置。生产记得让 uvicorn 开 `--proxy-headers`（镜像 CMD 已带），否则 nginx 转发后所有访客共享同一个"IP"。

### 4.6 sync_service.py + scripts/ —— 数据新鲜度

两条同步路径解决不同问题：

| | 增量同步（CLI/接口） | 实时同步（单篇接口） |
|---|---|---|
| 触发 | 手动 / `POST /api/admin/sync` | Laravel Observer 在保存后自动调 |
| 覆盖 | 全站对账（含下架清理） | 单篇 |
| 盲区 | 正文改动但标题摘要没变（列表接口无 `updated_at`） | 无 |
| 判据 | `sha256(标题|摘要|发布时间|分类|标签)` 对比 sqlite 状态表 | 事件驱动，不依赖对比 |

单篇同步的幂等设计（uuid5 + 先删后插）让"重复通知"无害——分布式同步的黄金特性。

`scripts/query.py` 是被低估的学习工具：不经过 LLM 直接看检索命中，**把"检索质量"和"生成质量"两个变量分开调试**。

### 4.7 tests/ —— 测试什么

只测**纯逻辑**（chunker 的切分规则、client 的包络/分页/重试），不测需要外部服务的部分。`respx` 模拟 httpx 传输层，让"上游 502 后重试成功"这种场景可复现。

---

## 5. 两次完整数据流走读

### 5.1 访客问「怎么给 PHP 数组去重」（跨 4 个进程）

```
浏览器(vue) fetch POST /agent-api/api/chat {thread_id, message}
  → nginx（剥 /agent-api 前缀，proxy_buffering off）
    → uvicorn：限流检查 → graph.astream
       ① retrieve：embed(问题) → Qdrant top5
       ② grade_documents：LLM 判相关性（json_mode）
       ③ generate：发 sources 事件 → 流式生成
    ← SSE: status → sources → delta×N → done
  ← 浏览器逐字渲染 markdown，sources 画成可点击 chips
（thread_id 对应的 messages 已被 checkpointer 写进 sqlite，刷新页面可恢复）
```

### 5.2 作者在后台改了一篇文章（跨 3 个项目、5 个环节）

```
Laravel ArticleObserver.updated（脏字段白名单过滤：只认内容字段）
  → 队列 job（blog-agent 专用队列，afterCommit）入 MySQL jobs 表
  → supervisor 的 worker 消费 → POST agent /api/admin/sync-article（token 鉴权）
    → agent 拉详情（404=已下架→删向量）
    → 清洗/切片/向量化 → 删旧点 → 插新点（幂等）
几秒后，访客的下一个问题就能检索到新内容。
```

---

## 6. 关键设计决策回顾（问题 → 选择 → 理由 → 代价）

1. **Python 而不是 Node**（尽管前端是 Vue）：LangChain/LangGraph 生态在 Python，学习目标是它们
2. **qdrant-client 直连，不用 langchain-qdrant 封装**：学习项目要看 points/payload/filter 原生形态，封装反而挡视线
3. **SQLite 存会话，不引 Redis/PG**：量级摆在那，单文件可备份。"为简历堆技术"和"解决问题"选后者
4. **检索上下文放 system 而非 user 消息**：不污染多轮历史，下一轮不会被上一轮的参考材料撑爆
5. **SSE 终止事件有且仅有一个（done 或 error）**：前端状态机因此可以写成"收到终止事件前一直 streaming"，不会出现悬挂的加载态
6. **同步接口 token 鉴权 + 常数时间比较**（`secrets.compare_digest`）：防时序攻击，一行的事
7. **换 embedding 模型必须重建 collection**：不同模型的向量空间不兼容，启动时校验维度就是防呆——把隐式约定变成显式报错

## 7. 踩坑实录（全部真实发生，按疼痛程度排序）

1. **glm-4-flash 结构化输出畸形**：function calling 返回 `'[1]\n[2]\n[3]'` → json_mode 也返回裸数组 → pydantic `mode="before"` 宽容包装 + 失败降级。**教训：小模型的结构化输出要按"会坏"来设计。**
2. **view_count 死循环**：agent 同步拉公开详情 → 浏览量+1（Eloquent save 事件）→ Observer 又派同步任务 → 无限循环，每 1.3 秒重建一次向量。**教训：事件驱动系统里，任何"读操作有副作用"的接口都可能形成正反馈；Observer 必须做脏字段过滤。**
3. **共享数据库的队列互踩**：本地开发直连生产库，服务器上的旧代码 worker 把新 Job 类打成 incomplete class。**教训：环境共享要按"最坏情况"隔离（独立队列名），长期方案是环境彻底分离。**
4. **Laravel `$queue` 属性冲突**：Job 类里重声明 `public $queue` 与 Queueable trait 的定义不兼容（Fatal Error）。**教训：用 trait 提供的 `$this->onQueue()` 方法。**
5. **bind mount 权限**：宿主机目录 root 属主 vs 容器内 app 用户 → sqlite 打不开。**教训：named volume 首次挂载继承镜像内属主，是"容器写文件"的省心方案。**
6. **nginx server 级 rewrite 劫持**：Laravel 的 `if (!-e) rewrite /index.php` 在 location 匹配**之前**执行，`/agent-api` 从未到达自己的 location。**教训：server 级 rewrite 是全局的，加新路径要么负向前瞻排除，要么改用 location 内 try_files。**
7. **前端 `@keydown` 忘加 `.enter`**：处理器对每个按键 preventDefault，输入框完全打不了字——而 curl 测试发现不了。**教训：交互逻辑必须真实操作验证，接口测试覆盖不了 UI。**
8. **容器间 127.0.0.1 陷阱**：Laravel/nginx 都在容器里，`127.0.0.1:8000` 指向容器自己。**教训：跨容器通信用服务名 + 共享 docker 网络。**

## 8. 配套部分速览（另两个仓库的改动）

**blog-vue（前端）**：
- `api/chat.ts`：fetch + ReadableStream 手写 SSE 解析（`\r\n\r\n|\n\n` 帧边界兼容、多行 data 拼接、注释行忽略）
- `stores/chat.ts`：Pinia 会话状态（threadId 持久化到 localStorage，浮窗和 /chat 页共享）
- `utils/markdown.ts`：markdown-it（`html:false`）+ DOMPurify（钩子强制 `target=_blank`）——**流式渲染 LLM 输出前必须消毒**
- `ChatComposer.vue`：中文输入法适配（`e.isComposing || e.keyCode === 229` 时不触发发送）

**blog-admin-v2（Laravel）**：
- `Observers/ArticleObserver.php`：脏字段白名单（SYNC_DIRTY_FIELDS）+ 派发两个同步 Job（ES / agent）
- `Jobs/SyncArticleToBlogAgent.php`：afterCommit + 专用队列 + 重试退避 + 失败钩子——队列 Job 的完整形态
- `config/blog_agent.php`：env 驱动的开关/地址/token

## 9. 动手练习（按难度递增）

1. **改提示词**：调 `_RAG_SYSTEM`，让回答更简短/更详细，观察引用标注率变化
2. **调参**：`RETRIEVAL_TOP_K` 从 5 改成 3 / 8，用 `scripts/query.py` 和真实提问对比命中质量
3. **加 groundedness 节点**：generate 后加一个"答案是否忠于参考资料"的判分节点，不合格重生成一次（图结构已支持，照抄 grade_documents 的写法）
4. **混合检索**：把博客现成的 ES 关键词接口 `/api/articles/search` 和向量结果做 RRF 融合（专有名词查不准的痛点就解决了）
5. **缓存热门问题**：相同问题的 embedding + 检索结果缓存（sqlite 即可），省 API 调用
6. **升级向量库用法**：给 payload 加 `category` 过滤索引，实现"只在这个分类里检索"

---

## 附：快速回忆卡

```
RAG 三步曲    embed(问题) → cosine top-k → LLM 看材料作答
检索质量 > 模型 切片策略决定上限，模型只决定下限
幂等三件套    uuid5 确定ID + 先删后插 + 404转删除
多轮记忆      state.messages + add_messages reducer + checkpointer(thread_id)
流式两通道    messages(token流) + custom(自定义事件)
小模型结构化  json_mode + before-validator + 失败降级
事件系统铁律  读操作不能有副作用，Observer 要过滤脏字段
```
