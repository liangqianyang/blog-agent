# Qdrant 向量数据库实战指南：从 Docker 部署到 RAG 混合检索

> 本文以一个真实的「在线教育错题本」业务场景为主线，从 Docker 部署、核心概念讲起，覆盖 Python / REST / PHP 三种接入方式、Embedding 入库与语义检索、MySQL ↔ Qdrant 数据同步架构、文本切分（Chunking）策略，直到稀疏向量 + RRF 融合的混合检索架构——一篇讲透 Qdrant 在生产环境中的完整用法。文中还穿插了我自己博客 RAG 问答助手的真实踩坑记录。

## 引言：为什么 AI 应用需要向量数据库

传统数据库的检索是「精确匹配」：`WHERE title LIKE '%三次握手%'` 要求用户的搜索词和存储的文本**字面上重合**。但用户的真实提问是千变万化的：

- 用户问「网络连接的三次握手是怎么回事」，而你文章的标题叫《TCP 建立连接的状态机解析》——关键词匹配直接失效；
- 用户搜「数组去掉重复元素」，找不到标题叫《PHP 数组去重详解》的文章——同义不同词，命中不了。

向量数据库解决的就是这个问题：把文本交给 Embedding 模型，转成一串高维浮点数（向量），**语义相近的文本，向量在空间中的距离也相近**。检索不再是「字符串匹配」，而是「找语义最近的邻居」。

Qdrant 是目前最流行的开源向量数据库之一，用 Rust 编写，特点是：

- **高性能**：Rust 实现 + HNSW 索引，百万级向量毫秒级检索；
- **Payload 过滤**：向量相似度检索的同时支持 JSON 元数据的结构化过滤（这是它对标 Milvus/FAISS 最大的差异化能力）;
- **混合检索**：原生支持稠密向量（Dense）+ 稀疏向量（Sparse）多路召回与 RRF 融合；
- **部署极简**：单个 Docker 容器即可跑起来，自带 Web 控制台。

下面我们从零开始，完整走一遍 Qdrant 的实战之路。

## 一、五分钟部署：Docker 起 Qdrant

官方推荐使用 Docker 部署：

```bash
# 拉取镜像
docker pull qdrant/qdrant

# 启动容器
docker run -d \
  --name qdrant_server \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**端口说明：**

| 端口 | 用途 |
|---|---|
| 6333 | REST API + Web 控制台 |
| 6334 | gRPC 高性能接口（批量写入场景更快） |

**验证服务是否存活：**

```bash
curl http://localhost:6333
# 返回 {"title":"qdrant - vector search engine","version":"..."}
```

**可视化控制台**：浏览器打开 `http://localhost:6333/dashboard`，可以直观地查看 Collection、浏览 Point、甚至直接在页面上执行检索调试——开发阶段非常好用。

如果是 docker-compose 编排（比如和你的应用服务放在一起），推荐这样写：

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage   # 数据持久化，容器删了数据还在

volumes:
  qdrant_data:
```

> 💡 **小技巧**：如果只是写个 Demo 做实验，Python 客户端还支持**内存模式**，连 Docker 都不用装：
> ```python
> client = QdrantClient(":memory:")  # 数据存在进程内存里，重启即失
> ```

## 二、核心概念：Collection、Point、Payload

写代码之前，必须理解 Qdrant 的三个核心设计。拿 MySQL 类比一下就秒懂：

| Qdrant | MySQL 类比 | 说明 |
|---|---|---|
| **Collection（集合）** | 表 | 同一集合内所有向量**维度必须一致**，且使用同一种距离度量（如余弦相似度） |
| **Point（点）** | 一行数据 | 由 `id` + `vector` + `payload` 三部分组成 |
| **Payload（负载）** | 行中的列 | JSON 格式元数据，存业务字段，检索时可施加结构化过滤 |

一个 Point 长这样：

```json
{
  "id": 10024,                          // 唯一标识（整数或 UUID）
  "vector": [0.05, 0.61, 0.76, 0.74],   // 向量本体
  "payload": {                          // 业务元数据
    "title": "数据库索引原理",
    "category": "后端"
  }
}
```

**Payload 是 Qdrant 的灵魂**。它让你可以做到「向量相似度 × 结构化条件」的混合查询，比如：

> 「找出和这道错题**语义最相似**的 5 道题，但限定必须是**数学学科**且**未解决**的。」

这在纯向量检索引擎（如裸 FAISS）里要自己绕很大弯子才能实现。

## 三、Python 客户端实战：从建集合到过滤检索

Python 是 AI 数据管道最主流的选择。先安装官方客户端：

```bash
pip install qdrant-client
```

下面是一段完整的数据操作流（注意：文中向量故意用了 4 维，方便阅读；实际对接 Embedding 模型时通常是 768 / 1024 / 1536 维）：

```python
from qdrant_client import QdrantClient, models

# 1. 连接服务
client = QdrantClient("localhost", port=6333)

# 2. 创建集合
client.create_collection(
    collection_name="my_knowledge_base",
    vectors_config=models.VectorParams(
        size=4,                      # 向量维度，必须与 Embedding 模型输出一致
        distance=models.Distance.COSINE,  # 距离度量：余弦相似度
    ),
)

# 3. 插入数据（upsert：ID 存在则覆盖更新，不存在则插入）
points = [
    models.PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74],
                       payload={"title": "数据库索引原理", "category": "后端"}),
    models.PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11],
                       payload={"title": "K8s 集群部署指南", "category": "运维"}),
    models.PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94],
                       payload={"title": "微服务架构设计", "category": "后端"}),
]
client.upsert(collection_name="my_knowledge_base", points=points)

# 4. 基础相似度搜索
results = client.query_points(
    collection_name="my_knowledge_base",
    query=[0.2, 0.1, 0.9, 0.7],   # 查询向量
    limit=2,                        # 返回最相似的 2 条
    with_payload=True,
).points

print("--- 基础搜索结果 ---")
for hit in results:
    print(f"ID: {hit.id}, 分数: {hit.score:.4f}, 详情: {hit.payload}")

# 5. 混合搜索：向量相似度 + Payload 结构化过滤
filtered = client.query_points(
    collection_name="my_knowledge_base",
    query=[0.2, 0.1, 0.9, 0.7],
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="后端"),  # 必须精确匹配"后端"
            )
        ]
    ),
    limit=2,
    with_payload=True,
).points

print("\n--- 过滤搜索结果（仅限后端分类） ---")
for hit in filtered:
    print(f"ID: {hit.id}, 分数: {hit.score:.4f}, 详情: {hit.payload}")
```

> ⚠️ **API 版本提示**：很多旧教程用的 `client.search()` 在 qdrant-client 1.10+ 已标记为废弃，请统一使用 `client.query_points()`——它同时兼容纯向量检索、过滤检索和后文的混合检索，是官方钦定的统一入口。

## 四、REST API 交互：跨语言通用方案

如果你的项目要保持轻量（不想引入 SDK），或者所用语言没有官方客户端，直接调 REST API 是标准做法。任何 HTTP 客户端都能搞定。

**1. 创建集合**

```
PUT http://localhost:6333/collections/my_knowledge_base
```

```json
{
    "vectors": {
        "size": 4,
        "distance": "Cosine"
    }
}
```

**2. 插入数据**

```
PUT http://localhost:6333/collections/my_knowledge_base/points?wait=true
```

```json
{
    "points": [
        {
            "id": 1,
            "vector": [0.05, 0.61, 0.76, 0.74],
            "payload": { "title": "数据库索引原理", "category": "后端" }
        }
    ]
}
```

**3. 带过滤条件的搜索**

```
POST http://localhost:6333/collections/my_knowledge_base/points/query
```

```json
{
    "query": [0.2, 0.1, 0.9, 0.7],
    "limit": 3,
    "with_payload": true,
    "filter": {
        "must": [
            { "key": "category", "match": { "value": "后端" } }
        ]
    }
}
```

> `?wait=true` 表示同步写（等落盘后才返回），数据管道里建议带上；高频写入可以去掉换取吞吐。

## 五、接入真实文本：Embedding 入库实战

前面演示的都是「现成的向量」。真实世界里，数据流转是这样的：

```
纯文本 → 调用 Embedding API → 获得向量（如 [0.012, -0.053, ...]）→ 附加业务 Payload → 存入 Qdrant
```

### 方案一：Python 脚本批量向量化（数据初始化）

```python
from openai import OpenAI
from qdrant_client import QdrantClient, models

openai_client = OpenAI(api_key="sk-your-api-key")
qdrant = QdrantClient("localhost", port=6333)

documents = [
    {"id": 1, "text": "TCP/IP 协议中的三次握手过程是什么？", "subject": "计算机网络"},
    {"id": 2, "text": "如何优化 MySQL 的慢查询？", "subject": "数据库"},
]

points = []
for doc in documents:
    # 调用 Embedding API（text-embedding-3-small 输出 1536 维）
    resp = openai_client.embeddings.create(
        input=doc["text"],
        model="text-embedding-3-small",
    )
    vector = resp.data[0].embedding

    points.append(models.PointStruct(
        id=doc["id"],
        vector=vector,
        payload={"content": doc["text"], "subject": doc["subject"]},
    ))

# 批量写入（生产环境建议每批 100~1000 条，别在循环里逐条 upsert）
qdrant.upsert(collection_name="tech_knowledge", points=points)
print(f"成功写入 {len(points)} 条向量！")
```

> 国内项目也可以用智谱（`embedding-3`，1024 维）、通义、DeepSeek 等任何 **OpenAI 兼容**接口，换 `base_url` 和 `model` 即可。我博客的 RAG 助手用的就是智谱 embedding-3。

### 方案二：业务后端实时入库（PHP / Laravel）

很多向量化的触发时机在用户交互时。比如在线教育系统里，学生做错一道题、错题写入 `user_wrong_question_pool` 表的瞬间，就应该同步生成向量，供后续「相似错题推荐」使用：

```php
use Illuminate\Support\Facades\Http;

// 刚写入错题池表的一条数据
$wrongQuestionId = 10024;
$questionContent = "已知集合 A={1,2,3}, B={2,3,4}，求 A ∪ B";
$subjectId = 2; // 数学

// 1. 调用 Embedding API 获取向量
$embedResponse = Http::withToken(env('OPENAI_API_KEY'))
    ->post('https://api.openai.com/v1/embeddings', [
        'model' => 'text-embedding-3-small',
        'input' => $questionContent,
    ]);

if ($embedResponse->successful()) {
    $vector = $embedResponse->json('data.0.embedding'); // 1536 维浮点数组

    // 2. 向量 + 业务元数据一起推给 Qdrant
    Http::put(env('QDRANT_HOST') . '/collections/wrong_questions/points?wait=true', [
        'points' => [
            [
                'id' => $wrongQuestionId,  // ★ 用 MySQL 主键作为 Qdrant Point ID
                'vector' => $vector,
                'payload' => [
                    'question_content' => $questionContent,
                    'subject_id' => $subjectId,
                    'is_resolved' => false,
                ],
            ],
        ],
    ]);
}
```

### 入库时的三条铁律

1. **维度必须对齐**：建集合时的 `size` 必须和 Embedding 模型输出维度**完全一致**。`text-embedding-3-small` 是 1536 维，智谱 `embedding-3` 是 1024 维，BGE-M3 是 1024 维——换模型 = 重建集合并全量重新灌入。
2. **保持 ID 一致性**：把关系库的主键作为 Point 的 ID。这样更新/删除时能用同一个 ID 直接覆写或移除向量，业务侧无需维护任何映射表。
3. **文本要切分（Chunking）**：Embedding 模型有上下文长度限制（通常 8192 Token），且文本越长语义越稀释。长文章必须先切块再逐块向量化（第八节详细展开）。

## 六、语义检索：把用户的搜索词也变成向量

检索的逻辑和入库是**对称的**：入库用什么模型，查询就必须用**同一个模型**。完整流程：

```
用户搜索词 → Embedding API 提取查询向量 → 组装过滤条件 → Qdrant 执行检索 → 返回相似结果
```

### Python 版

```python
user_query = "网络连接的三次握手是怎么回事？"

# 1. 搜索词转向量（必须与入库时同模型）
resp = openai_client.embeddings.create(input=user_query, model="text-embedding-3-small")
query_vector = resp.data[0].embedding

# 2. 组装过滤条件：只在"计算机网络"分类下搜
query_filter = models.Filter(
    must=[models.FieldCondition(key="subject", match=models.MatchValue(value="计算机网络"))]
)

# 3. 执行检索
results = qdrant.query_points(
    collection_name="tech_knowledge",
    query=query_vector,
    query_filter=query_filter,
    limit=3,                  # Top 3
    score_threshold=0.5,      # 相似度阈值，低于 0.5 的直接丢弃
    with_payload=True,
).points

for hit in results:
    print(f"相似度 {hit.score:.4f} | {hit.payload['content']} (ID: {hit.id})")
```

`score_threshold` 很实用：余弦相似度低于阈值的结果大概率不相关，与其返回 3 条凑数的，不如返回空让上层走兜底话术。

### PHP 版：多条件混合检索

学生复习时，系统要从错题池里找「和当前这道题相似 + 数学学科 + 还没解决」的错题：

```php
$searchQuery = "集合A与集合B的交集怎么求？";

$embedResponse = Http::withToken(env('OPENAI_API_KEY'))
    ->post('https://api.openai.com/v1/embeddings', [
        'model' => 'text-embedding-3-small',
        'input' => $searchQuery,
    ]);

if ($embedResponse->successful()) {
    $queryVector = $embedResponse->json('data.0.embedding');

    $searchResponse = Http::post(env('QDRANT_HOST') . '/collections/wrong_questions/points/query', [
        'query' => $queryVector,
        'limit' => 3,
        'with_payload' => true,
        'score_threshold' => 0.65,   // 业务阈值：太低说明不相关
        'filter' => [
            'must' => [
                // 条件 1：必须是当前学科
                ['key' => 'subject_id', 'match' => ['value' => 2]],
                // 条件 2：题目状态为未解决（布尔过滤）
                ['key' => 'is_resolved', 'match' => ['value' => false]],
            ],
        ],
    ]);

    foreach ($searchResponse->json('result') as $item) {
        // $item['id']     → 错题表主键
        // $item['score']  → 相似度得分
        // $item['payload']['question_content'] → 题目原文
        echo "匹配 ID: {$item['id']}, 相似度: {$item['score']}\n";
    }
}
```

**业务对接提示**：拿到返回的 ID 列表后，回关系库执行一次 `WHERE id IN (...)` 就能关联出解析、答题记录等完整业务数据——这就是「向量库管召回、关系库管详情」的标准分工。

## 七、数据一致性：MySQL 与 Qdrant 的同步

关系库的记录会改、会删，向量库必须跟着动，否则会检出「幽灵数据」。核心原则还是那条：**主键 ID 即 Point ID**，两边天然精确映射。

### 场景一：记录被删除 → 删向量

```php
// 刚从 user_wrong_question_pool 物理删除了错题 10024
Http::post(env('QDRANT_HOST') . '/collections/wrong_questions/points/delete', [
    'points' => [10024],   // 支持批量传多个 ID
]);
```

### 场景二：核心文本被修改 → 重新向量化 + Upsert 覆盖

用户修正了题目内容，旧向量已经失效。重新调 Embedding API，然后 Upsert——ID 不变，Qdrant 直接覆盖旧向量：

```php
Http::put(env('QDRANT_HOST') . '/collections/wrong_questions/points?wait=true', [
    'points' => [
        [
            'id' => $wrongQuestionId,   // ID 保持不变 → 触发覆盖
            'vector' => $newVector,     // 重新生成的新向量
            'payload' => [
                'question_content' => $newContent,
                'subject_id' => $subjectId,
                'is_resolved' => false,
            ],
        ],
    ],
]);
```

### 场景三：只有业务属性变了 → Set Payload，不烧 Embedding 额度

只是 `is_resolved` 从 false 变 true？题目文本没变，向量完全不用重新算。用 Set Payload 接口只改元数据：

```php
Http::post(env('QDRANT_HOST') . '/collections/wrong_questions/points/payload', [
    'payload' => ['is_resolved' => true],
    'points' => [10024],
]);
```

（如果想彻底替换全部 JSON 属性，Qdrant 还提供了 Overwrite Payload 接口。）

### 架构建议：用队列解耦，保证最终一致性

**强烈不建议在同步请求里直接调 Qdrant**。网络一抖动，MySQL 更新成功但向量库没更新，数据就永久不一致了。正确姿势：

```
                ┌─ Model Observer 监听 saved / deleted 事件
MySQL 数据变更 ─┤
                └─→ 投递队列（Redis / RabbitMQ）─→ 后台消费者
                                                   ├─ 调 Embedding API（失败自动重试）
                                                   └─ 调 Qdrant Upsert / Delete
```

- **监听变更**：Laravel 的 Model Observers（或 ThinkPHP 的模型事件）；
- **投队列不直调**：利用队列的重试机制保证最终一致性；
- **专用队列名**：给同步任务单独的 queue，避免和普通任务互相阻塞。

### 🩸 真实踩坑：一次同步触发的死循环

我自己博客的 RAG 助手就是这么搭的：Laravel 的 `ArticleObserver` 在文章新增/修改/删除后派发一个队列 Job，POST 到 Python agent 的同步接口（幂等，单篇重建向量；文章被删则自动转删除向量）。

上线后发现一个诡异现象：**每 1.3 秒就重建一次向量，文章阅读量疯涨**。排查后真相是——

```
agent 同步文章 → 调博客公开详情接口 → 详情接口 view_count + 1（Eloquent 保存）
    ↑                                                    │
    └──────── Observer 监听到 updated，又派发同步 Job ────┘
```

一个完美的死循环。修复方式：Observer 的 `updated` 钩子里加**脏字段白名单**——只有 `title / content / summary / status / 分类` 这类「语义字段」变化才触发同步，纯统计字段（view_count）的变化直接忽略。

> 教训：**同步的触发条件应该基于『影响向量语义的字段』，而不是任意 UPDATE。** 顺带还修好了一个老毛病——以前每次有人浏览文章都会白白重建一次全文索引。

## 八、Chunking：切出「完美向量」的四种策略

向量检索圈有句名言：**Garbage in, garbage out**。Embedding 模型只是忠实地把文字变成数字，切分质量直接决定检索上限。「完美切分」的标准只有一条：**每个 Chunk 包含完整、独立的语义，且没有多余噪音。**

### 策略 1（底线）：滑动窗口切分

- **Chunk Size**：200~500 Token，刚好容纳 1~2 个完整概念；
- **Overlap（重叠）**：防止「一刀切断语义」。按 300 字切块时保留约 50 字重叠：

```
块 1：字符 0 ~ 300
块 2：字符 250 ~ 550   ← 250~300 是重叠区，保证上下文连贯
块 3：字符 500 ~ 800
```

### 策略 2（提精度）：基于结构的语义切分

别按字数硬切，要顺应写作结构：

- 优先按段落（`\n\n`）切，其次按句子（`。`）切；
- **Markdown 文档按标题层级（`#` / `##` / `###`）切是最佳方案**——同一标题下的内容天然语义聚合。

### 策略 3（业务数据）：不「切」，而是「组装」

对高度结构化的业务数据（题目、商品、工单），传统切分会把「题干、选项、解析」切成三块，题目语义就碎了。正确做法是用模板把关键字段**拼装**成一段高密度文本。一个可直接落地的 PHP Service：

```php
<?php

namespace App\Services;

class QuestionVectorBuilder
{
    /**
     * 将错题池的结构化数据组装成高质量向量文本
     */
    public function buildEmbeddingText(array $question): string
    {
        // 1. 前置元数据：强化模型对领域和考点的认知
        $subject = $question['subject_name'] ?? '综合';
        $knowledgePoints = isset($question['knowledge_points'])
            ? implode(', ', (array) $question['knowledge_points'])
            : '未分类考点';
        $metaHeader = sprintf("学科：%s | 考点：%s\n", $subject, $knowledgePoints);

        // 2. 清洗题干（剔除富文本编辑器带来的 HTML 噪音）
        $stem = $this->cleanRichText($question['stem'] ?? '');

        // 3. 组装选项
        $optionsText = '';
        if (!empty($question['options']) && is_array($question['options'])) {
            $optionsList = [];
            foreach ($question['options'] as $key => $val) {
                $optionsList[] = sprintf("%s.%s", strtoupper($key), $this->cleanRichText($val));
            }
            $optionsText = "\n选项：" . implode(' ', $optionsList);
        }

        // 4. 最终组装
        // ★ 刻意不加入 analysis（解析）和 wrong_answer（用户错答）：
        //   解析包含大量衍生知识，会带偏相似度；解析存 MySQL，靠 ID 回表查
        return $metaHeader . "题目：" . $stem . $optionsText;
    }

    /**
     * 清理富文本噪音，保留纯净语义
     */
    private function cleanRichText(string $text): string
    {
        if (empty($text)) {
            return '';
        }
        // 移除所有 HTML 标签（<p>、<span>、<br>…对模型是纯噪音）
        $text = strip_tags($text);
        // 连续空白（换行/制表符/全角空格）压缩为单个空格，省 Token 且聚焦实体
        $text = preg_replace('/[\s\x{3000}]+/u', ' ', $text);
        return trim($text);
    }
}
```

组装效果对比：

```
❌ "题目ID:1024, 题干:已知..., 选项A:1, 选项B:2... 解析:本题考查了...过程极其复杂... 用户错选了B..."
✅ "学科：初中数学 | 考点：二次函数, 图像平移
    题目：将抛物线 y=x² 向右平移 2 个单位，再向下平移 1 个单位，得到的解析式是？
    选项：A.y=(x-2)²-1 B.y=(x+2)²-1 C.y=(x-2)²+1 D.y=(x+2)²+1"
```

### 策略 4（长文档终极形态）：父子文档检索（Small-to-Big）

长教程类文档会遇到一个两难：切太细，向量匹配精准但 LLM 看不懂没头没尾的碎片；切太大，LLM 读着舒服但相似度匹配不准。解法是「**切小块存，拿大块用**」：

1. 原始长文档存主库，赋予 `doc_id`；
2. 切成极小的 Chunk（单句/单段落），向量化后存入 Qdrant；
3. Payload 里附上 `doc_id`；
4. 检索时精准命中小 Chunk → 拿到 `doc_id` → 回主库取整篇上下文 → 交给 LLM 生成。

### 📌 我博客的真实切法

我的博客 RAG 助手用的是「策略 2 + 策略 4」的组合：文章 HTML 按 DOM 结构（标题层级）切块，每块约 500 字；每块的 Embedding 输入不是裸正文，而是——

```
《文章标题》 > 一级章节 > 二级章节
+ 正文片段
```

给每个碎片「自带文章语境」，检索命中章节级片段后，把片段 + 文章信息一起交给 LLM 作答。上线后最直观的改善：搜「怎么去重」能直接命中《PHP 数组去重详解》里的具体某一节。

## 九、硬核场景：LaTeX 数学公式怎么检索

理科题库是向量检索公认的痛点。通用 Embedding 模型预训练以自然语言为主，遇到 `$\frac{\sqrt{x^2+1}}{2x}$` 这样的符号，Tokenizer 会把它切碎成毫无逻辑的离散字符，导致「形式相似但数学本质不同」的题目被错误召回。四个维度的优化：

### 1. 入库前清洗：剥离视觉噪音，保留数学本质

很多 LaTeX 是为「排版好看」存在的，对语义毫无贡献：

```php
private function cleanLatexNoise(string $text): string
{
    // 移除字体/排版指令，仅保留内部内容
    $text = preg_replace('/\\\\(?:mathbf|mathrm|text|boldsymbol)\{([^}]*)\}/', '$1', $text);
    // 移除空白控制符和显示控制符
    $text = str_replace(['\\,', '\\;', '\\quad', '\\qquad', '\\displaystyle'], '', $text);
    // 统一等价符号写法
    $text = str_replace(['\\leq', '\\geq'], ['\\le', '\\ge'], $text);
    return $text;
}
```

### 2. 语义锚点：用自然语言「翻译」公式意图

既然模型更懂自然语言，就人为把公式的数学意图补进题干。对 `求解方程：$\log_2(x^2-1)=3$`，组装时在开头注入：

```
[对数方程、一元二次方程] 求解方程：log_2(x^2-1)=3
```

即使模型对 LaTeX 解析弱，前置的自然语言标签也能瞬间拉高这道题在「对数方程」维度的相似度。

### 3. 架构升级：混合检索（下一节详解）

稠密向量懂语义，稀疏向量（BM25/SPLADE）擅长精确匹配 `\int`、`\sum`、`\infty` 这类稀有符号——两路召回融合是公式检索最有效的工程手段。

### 4. 换底座：数学微调的 Embedding 模型

| 模型 | 适用场景 | 优势 |
|---|---|---|
| **BGE-M3** | 综合理工科题库 | 多语言 + 长文本，对代码/公式兼容好，**一个模型同时输出稠密+稀疏向量**（天生适配 Qdrant 混合检索） |
| **MathBERT / SciBERT** | 高阶数学/学术场景 | 针对科研论文和 LaTeX 预训练，极其懂数学符号 |
| **Qwen2-Math** | 复杂公式提权（推理端） | 生成模型，但其 Tokenizer 对 LaTeX 的切分非常符合数学逻辑 |

**落地建议**：初期先做 1（正则降噪）+ 2（标签注入），在现有组装逻辑里加几十行代码就能立竿见影；规模上来、遇到准确率瓶颈后，再上稀疏向量混合检索。

## 十、终极架构：Dense + Sparse 混合检索（RRF 融合）

混合检索的核心思想：**多向量存储（Named Vectors）+ 倒数秩融合（RRF）**。同一道题同时存两种向量：

- **稠密向量（Dense）**：捕捉自然语言语义（1536 维浮点数组）；
- **稀疏向量（Sparse）**：捕捉符号、公式、关键词词频（`{索引: 权重}` 字典，BM25/SPLADE 生成）。

### 1. 创建支持多向量的 Collection

```
PUT /collections/wrong_questions
```

```json
{
    "vectors": {
        "dense_text": {
            "size": 1536,
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse_text": {
            "index": { "on_disk": true }
        }
    }
}
```

> `on_disk: true`：数据量大时稀疏索引落盘，省内存。

### 2. 插入双向量数据

架构提示：PHP 不擅长跑模型。标准做法是内网用 Docker 部署一个轻量 Python 微服务（跑 BGE-M3 或 BM25），PHP 把洗净的题目发过去，微服务同时返回 Dense + Sparse 两种向量。

```
PUT /collections/wrong_questions/points?wait=true
```

```json
{
    "points": [
        {
            "id": 10025,
            "vector": {
                "dense_text": [0.015, -0.022, 0.081],
                "sparse_text": {
                    "indices": [210, 4503, 10211],
                    "values": [1.2, 0.8, 2.5]
                }
            },
            "payload": {
                "subject_name": "初中数学",
                "question_content": "求解方程：\\log_2(x^2-1) = 3"
            }
        }
    ]
}
```

> 键名 `dense_text` / `sparse_text` 必须与建集合时的定义完全一致。

### 3. Query API：prefetch 预取 + RRF 融合

```
POST /collections/wrong_questions/points/query
```

```json
{
    "prefetch": [
        {
            "query": [0.015, -0.022, 0.081],
            "using": "dense_text",
            "limit": 20
        },
        {
            "query": {
                "indices": [210, 891],
                "values": [1.5, 1.1]
            },
            "using": "sparse_text",
            "limit": 20
        }
    ],
    "query": { "fusion": "rrf" },
    "filter": {
        "must": [
            { "key": "subject_name", "match": { "value": "初中数学" } }
        ]
    },
    "limit": 5,
    "with_payload": true
}
```

执行逻辑：稠密、稀疏两路**各自**先召回 20 条（prefetch），再由 RRF 算法按「两路排名的倒数」融合打分，输出最终 Top 5。这样既懂「语义」，又能精准命中 `\log_2` 这样的稀有符号。

Python 客户端版本：

```python
from qdrant_client import models

result = client.query_points(
    collection_name="wrong_questions",
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense_text", limit=20),
        models.Prefetch(
            query=models.SparseVector(indices=[210, 891], values=[1.5, 1.1]),
            using="sparse_text",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=models.Filter(
        must=[models.FieldCondition(key="subject_name",
                                    match=models.MatchValue(value="初中数学"))]
    ),
    limit=5,
    with_payload=True,
).points
```

> 注意：RRF 融合得分的数值区间和单纯余弦相似度不同，不能用 0.65 这类余弦阈值去卡——融合排序的分数需要单独标定，或干脆只用 `limit` 控制返回条数。

## 十一、生产环境 Checklist

最后把散落在全文的生产建议汇总成一张清单：

| # | 实践 | 说明 |
|---|---|---|
| 1 | **给过滤字段建 Payload 索引** | 频繁用于 `filter` 的字段（如 `category`）必须建索引，混合查询速度能提升几个数量级：`PUT /collections/{name}/index`，body `{"field_name": "category", "field_schema": "keyword"}` |
| 2 | **批量写入** | 永远不要在循环里逐条 upsert，几百条打包成一次批量请求 |
| 3 | **数据解耦** | 长正文别塞 Payload。Qdrant 存向量 + 基础属性，详情回关系库查 |
| 4 | **主键即 Point ID** | 天然映射，更新删除都简单 |
| 5 | **队列解耦同步** | Observer → 队列 → 消费者，靠重试保证最终一致性；触发条件按「语义字段」白名单过滤 |
| 6 | **专用队列名** | 同步任务单独一个 queue，避免和业务任务互相阻塞 |
| 7 | **换 Embedding 模型 = 全量重建** | 维度/语义空间都变了，建新集合 → 灌数据 → 切换别名，别原地改 |
| 8 | **快照备份** | `POST /snapshots` 定期备份 collection；大库可开标量量化（quantization）省内存 |

## 写在最后

一张图总结 Qdrant 的完整实战路径：

```
部署（Docker）
   → 建模（Collection / Point / Payload）
   → 入库（Chunking 组装 → Embedding → Upsert）
   → 检索（查询向量化 → Filter 过滤 → Top-K）
   → 同步（Observer → 队列 → Upsert/Delete/SetPayload）
   → 进阶（稀疏向量 → Prefetch → RRF 混合检索）
```

核心思想其实只有三条：

1. **向量库管召回，关系库管详情**——各司其职，用 ID 桥接；
2. **检索质量的上限在入库之前就决定了**——Chunking 和文本清洗比调参更值得投入；
3. **混合检索是精确匹配和语义理解的合体**——关键词能救符号/术语场景，语义能救同义改写场景，RRF 让你全都要。

这些内容全部来自我给自己的博客搭建 RAG 问答助手的真实实践（LangGraph 检索问答图 + Qdrant + Laravel 实时同步管道），每一节踩过的坑都真实存在。如果只想快速上手，把第三节的 Python 代码跑通，你就已经掌握了 80% 的日常用法。
