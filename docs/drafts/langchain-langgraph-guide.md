# LangChain 与 LangGraph 实战指南：多智能体协作、人工介入与时间旅行

> LangChain 和 LangGraph 到底是什么关系？本文从两个框架的定位与核心区别讲起，用一个可完整运行的「生成器 × 审核器」多智能体案例贯穿全文，逐步加上 Human-in-the-loop 人工断点、PostgreSQL 生产级持久化、状态时间旅行与 Checkpoint 清理策略——一篇讲透 LangGraph 从入门到上生产的完整路径。文中还穿插了我自己博客 RAG 问答助手的真实选型与踩坑记录。

## 引言：一个框架为什么裂变成了两个

如果你用 LLM 做过哪怕一个 Demo，大概率绕不开 LangChain：调模型、拼提示词、解析输出、挂工具，它把这套脏活全标准化了。但当你想做的事情稍微复杂一点——比如「AI 生成一版内容，另一个 AI 审核，不通过就打回重写，直到满意或超过 3 次」——你会发现 LangChain 的链式结构很别扭：链是**线性**的，A 到 B 到 C，走完就结束，天生不适合「循环」和「反悔」。

LangGraph 就是为解决这个问题而生的：把工作流抽象成一张**支持循环的状态图**，节点间可以来回跳，全局状态在节点之间传递，还能随时暂停、恢复、回滚。

我给自己的博客搭 RAG 问答助手时（LangChain + LangGraph + Qdrant + FastAPI），正是靠 LangGraph 的循环能力实现了「检索 → 判分 → 不合格就改写问题再检索」的闭环——这事儿用纯 LangChain 链根本做不了。本文就把这套东西从概念到生产完整讲一遍。

## 一、LangChain 是什么：砖块与水泥

LangChain 是一个用于开发 LLM 驱动应用的开源框架，核心目标是**标准化 LLM 与外部数据源、工具的交互**。

- **核心逻辑**：基于链（Chains）和有向无环图（DAG）的思想。你把提示词（Prompt）、模型（Model）、输出解析器（Output Parser）等组件像流水线一样串起来——官方的组装语法叫 LCEL（LangChain 表达式语言），用 `|` 管道符连接：

```python
chain = prompt | model | output_parser
result = chain.invoke({"question": "什么是余弦相似度"})
```

- **典型应用**：RAG（检索增强生成）、单任务问答、文档总结、简单的单智能体（ReAct Agent）调用工具。
- **局限**：链式结构本质是线性的，处理需要「反复思考」「自我修正」「多角色协作」的任务时力不从心；传统 AgentExecutor 又是个黑盒，Agent 为什么卡死、为什么死循环，你很难干预。

一句话：**LangChain 提供的是砖块和水泥**——模型接口、提示词模板、工具、文档加载器、向量库对接。

## 二、LangGraph 是什么：会循环的状态图

LangGraph 出身于 LangChain 生态（现在作为独立库维护，但深度集成），专门构建**状态化（Stateful）+ 支持循环（Cyclic）**的 LLM 应用，尤其是复杂的多智能体系统。

它把工作流抽象为图（Graph），三个核心构件：

| 构件 | 作用 |
|---|---|
| **节点（Node）** | 一个具体操作：调 LLM、执行脚本、调 API，甚至就是一个 if 判断 |
| **边（Edge）/ 条件边（Conditional Edge）** | 决定下一步走向哪个节点；条件边支持循环（生成 → 审核报错 → 回到生成） |
| **状态（State）** | 一个全局状态对象，在整个图执行过程中被所有节点读取和更新 |

典型应用：带规划/反思/重试能力的复杂 Agent、多智能体协作（一个写代码一个 Review）、需要长时间保持上下文的任务流、需要人工审批的关键节点。

一句话：**LangGraph 提供的是建筑蓝图**——让你用 LangChain 的砖块，盖出能自我循环、能暂停等人、能回滚重来的大楼。

## 三、核心区别：一张表 + 两张图

| 维度 | LangChain（基础框架 / LCEL） | LangGraph |
|---|---|---|
| 架构基础 | 链式结构（Chains）/ 有向无环图（DAG） | 循环图（Cyclic Graphs） |
| 执行流 | A → B → C，单向流动为主 | 支持循环流（A → B → C → A），可实现「反思-修正」闭环 |
| 状态管理 | 相对松散，通常靠 Memory 组件在外部维护 | 原生全局状态（State），每个节点都读写同一份状态，适合长时间任务 |
| Agent 支持 | 适合简单单智能体（ReAct） | 专为复杂、自定义的多智能体工作流设计 |
| 控制力 | 黑盒 Agent 内部流转难以干预（不知道为什么卡死） | 控制力极强：条件路由、中止逻辑、断点全部显式声明，不会莫名死循环 |

用两张 ASCII 图直观看：

```
LangChain（链 / DAG）：单向流水线，走完即结束

  prompt → model → parser → END


LangGraph（循环图）：可以打回重做，可以熔断退出

  START → generator → reviewer ──通过──→ publish → END
             ↑          │
             └──打回重写──┘  （循环 N 次直到通过 / 达到上限）
```

## 四、两者的联系：不是替代，是分层

LangGraph 并不是来「取代」LangChain 的，两者是**分层共生的关系**：

1. **生态继承**：LangGraph 构建在 LangChain 之上，同一个团队出品、同一套版本节奏；
2. **组件共用**：LangGraph 的「节点」内部，用的依然是 LangChain 的核心组件——`ChatOpenAI` 模型对象、各种 Tools、PromptTemplates，原封不动；
3. **演进关系**：LangChain 给砖块和水泥，LangGraph 给蓝图。实际项目里两者经常同时出现在依赖清单中：模型调用走 LangChain 抽象，流程编排走 LangGraph。

**选型口诀**：任务线性、一次跑完 → LangChain LCEL 足够；需要循环、多角色、人工介入、断点恢复、状态回滚 → 上 LangGraph。二者不冲突。

## 五、实战：Generator 与 Reviewer 的多智能体协作

场景：生成器（Generator）按任务写草稿，审核器（Reviewer）检查草稿；不通过就带修改建议打回重写，通过或达到最大重试次数则结束。这是多智能体协作里最经典的「生成-批判」模式。

### 环境准备

```bash
pip install langgraph langchain-openai pydantic
```

### 完整代码

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. 全局状态（State）：两个 Agent 共享的「上下文公文包」
# ---------------------------------------------------------
class AgentState(TypedDict):
    task: str           # 用户的初始任务
    content: str        # 生成器产出的草稿
    feedback: str       # 审核器给出的修改建议
    is_approved: bool   # 是否通过审核
    loop_count: int     # 循环次数，防止死循环

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# ---------------------------------------------------------
# 2. 生成器节点（Generator Agent）
# ---------------------------------------------------------
def generator_node(state: AgentState):
    task = state.get("task", "")
    feedback = state.get("feedback", "")
    loop_count = state.get("loop_count", 0)

    # 有 feedback 说明是被审核器打回重写的
    if feedback:
        prompt = f"任务: {task}\n\n前一版被拒绝。审核建议: {feedback}\n\n请根据建议修改并输出新版本。"
    else:
        prompt = f"请完成以下任务: {task}"

    response = llm.invoke(prompt)
    return {"content": response.content, "loop_count": loop_count + 1}

# ---------------------------------------------------------
# 3. 审核器节点（Reviewer Agent）
# ---------------------------------------------------------
class ReviewOutput(BaseModel):
    is_approved: bool = Field(description="草稿是否完美达标，没有任何错误？")
    feedback: str = Field(description="如果不达标，给出具体的修改建议；如果达标，回复'通过'。")

def reviewer_node(state: AgentState):
    reviewer_llm = llm.with_structured_output(ReviewOutput)
    prompt = (
        f"原始任务: {state['task']}\n\n待审内容:\n{state['content']}\n\n"
        f"请作为严苛的审核员，判断内容是否完全符合要求。"
    )
    result = reviewer_llm.invoke(prompt)
    return {"is_approved": result.is_approved, "feedback": result.feedback}

# ---------------------------------------------------------
# 4. 条件路由（Conditional Edge）+ 熔断
# ---------------------------------------------------------
def router(state: AgentState):
    # 防死循环：超过 3 次直接结束
    if state.get("loop_count", 0) >= 3:
        print("--- 达到最大重试次数，强制结束 ---")
        return END
    if state.get("is_approved"):
        print("--- 审核通过，流程结束 ---")
        return END
    print(f"--- 审核未通过（打回重写）: {state.get('feedback')} ---")
    return "generator"

# ---------------------------------------------------------
# 5. 编排并编译图
# ---------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("generator", generator_node)
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "generator")       # 起点 → 生成器
workflow.add_edge("generator", "reviewer")  # 生成器 → 审核器
workflow.add_conditional_edges(
    "reviewer",
    router,
    {"generator": "generator", END: END},   # 路由返回值 → 目标节点
)

app = workflow.compile()

# ---------------------------------------------------------
# 6. 运行
# ---------------------------------------------------------
if __name__ == "__main__":
    final_state = app.invoke({
        "task": "写一首关于程序员修 Bug 的打油诗，要求恰好四句，每句七个字。",
        "loop_count": 0,
    })
    print(final_state["content"])
```

### 三个关键设计

1. **TypedDict 状态 = 公文包**：Generator 把稿件放进去，Reviewer 取出来看、把批注写进 `feedback` 再传回去。所有节点只返回「增量」，LangGraph 自动合并进全局状态。
2. **结构化输出是路由的地基**：Reviewer 用 `with_structured_output(ReviewOutput)` 强制 LLM 返回严格的布尔值和字符串，代码里才能稳定地 `if result.is_approved`。让 LLM「自由发挥」输出一段话再正则去抠，是新手最常见的翻车点。
3. **熔断必须有**：LLM 偶尔会「死脑筋」改不出 Reviewer 满意的结果，`loop_count >= 3` 这个跳出机制保住的不只是程序，还有你的 API 账单。

### 🩸 真实踩坑：`with_structured_output` 在弱模型上并不稳

我博客的 RAG 问答图里，「检索质量判分」节点同样需要结构化输出（判断每篇检索结果相不相关）。实际跑智谱 glm-4-flash 时发现它的 function calling 参数经常畸形，`with_structured_output` 默认走 tool calling 模式，解析时不时炸。

解法：改用 `method="json_mode"` 让模型直接吐 JSON，再用 pydantic 的 `before-validator` 做宽容解析（容忍裸 JSON、markdown 代码块包裹、多余字段）。**结论：结构化输出不要默认信任任何模型，永远留一层宽容解析。**

另外，我这张图本身就是本文模式的翻版：

```
START → retrieve → grade_documents ─┬ 相关 → generate → END
             ↑                      ├─ 不相关且未达上限 → rewrite_query → retrieve
             └──────────────────────┴─ 兜底 → generate → END
```

「检索质量不够就改写问题再检索」这个循环，正是选 LangGraph 的全部理由。

> 💡 国内模型一行接入：`ChatOpenAI(base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash")`——智谱/DeepSeek/通义都是 OpenAI 兼容接口，换 `base_url` 即可，LangChain 的抽象在这里价值拉满。

## 六、人工介入：Human-in-the-loop

真实业务里，AI 审核通过不等于能发布——扣款、发文、删数据这类动作，必须人类点头。LangGraph 的实现机制是**持久化（Checkpointing）+ 断点（Interrupt）**：给图配一个检查点存储，运行到指定节点前把状态存下来并挂起，等人确认（或改完状态）后再恢复执行。

两个新概念：

- **Checkpointer**：状态存档器。示例用内存版 `MemorySaver`，生产换 Postgres/Redis（下一节）；
- **thread_id**：执行流的身份证。图可能暂停很久，靠它找回对应的执行流。

### 代码：审核通过后，等人点「同意」再发布

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver   # ← Checkpointer
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    task: str
    content: str
    is_approved: bool

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def generator_node(state: AgentState):
    print(">> [Generator] 正在生成内容...")
    return {"content": llm.invoke(state["task"]).content}

class ReviewOutput(BaseModel):
    is_approved: bool = Field(description="是否达标？")

def reviewer_node(state: AgentState):
    print(">> [Reviewer] 正在审核...")
    reviewer_llm = llm.with_structured_output(ReviewOutput)
    result = reviewer_llm.invoke(f"任务:{state['task']}\n内容:{state['content']}")
    return {"is_approved": result.is_approved}

def publish_node(state: AgentState):
    # 这个节点只有在人类同意后才会执行
    print("\n>> [Publish] 🚀 最终内容已发布上架！")
    return state

def router(state: AgentState):
    return "publish" if state.get("is_approved") else "generator"

workflow = StateGraph(AgentState)
workflow.add_node("generator", generator_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("publish", publish_node)
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "reviewer")
workflow.add_conditional_edges("reviewer", router,
                               {"publish": "publish", "generator": "generator"})
workflow.add_edge("publish", END)

# ================= 核心差异部分 =================
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["publish"],   # ★ 进入 publish 节点前暂停
)

if __name__ == "__main__":
    # 使用持久化必须指定 thread_id
    config = {"configurable": {"thread_id": "thread_demo_01"}}

    print("=== 第一阶段：AI 自动生成与审核 ===")
    for event in app.stream({"task": "写一句关于AI的简短名言"},
                            config=config, stream_mode="values"):
        pass   # 图会在 publish 之前停下

    current_state = app.get_state(config)
    print("\n=== 等待人工介入 ===")
    print(f"当前待发布的草稿内容：\n{current_state.values.get('content')}")
    print(f"下一节点（被挂起）: {current_state.next}")

    user_input = input("\n人类，你同意发布这段内容吗？(y/n): ")

    if user_input.lower() == "y":
        print("\n=== 第二阶段：人类同意，恢复执行 ===")
        # 传 None + 同一个 config → 从断点处继续
        for event in app.stream(None, config=config, stream_mode="values"):
            pass
    else:
        print("\n=== 流程终止 ===")
        # 拒绝时可以抛弃，或用 update_state 把状态打回 generator
```

### 进阶：人不仅能「同意」，还能「手动改」

断点处人类想润色一下再放行？用 `update_state` 直接改状态：

```python
# 人类手动修改了草稿
app.update_state(config, {"content": "这是人类手动润色后的最终版本内容。"})

# 再恢复执行，publish 拿到的就是人类修改后的 content
app.invoke(None, config=config)
```

这个模式非常适合「AI 批量生成 + 人工抽检微调 + 放行上架」的业务：AI 负责产量，人类守质量闸门。

> 💡 **版本提示**：`interrupt_before` 是编译期的静态断点；LangGraph 新版本还提供了在节点内部动态打断的 `interrupt()` 函数（配合 `Command(resume=...)` 恢复并携带人类输入），适合断点位置依赖运行时数据的场景。两种方式底层都依赖同一个 Checkpointer 机制。

## 七、生产级持久化：把状态存进数据库

`MemorySaver` 进程一重启状态就没了，人工介入挂起半天的执行流会直接丢失。生产环境必须换成真数据库，官方支持最完善的是 PostgreSQL。

### PostgreSQL Checkpointer（官方推荐）

```bash
pip install langgraph-checkpoint-postgres psycopg psycopg_pool
```

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

DB_URI = "postgresql://username:password@localhost:5432/langgraph_db"

# 连接池：避免每次请求重新建连（高并发必备）
connection_pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True},   # PostgresSaver 需要 autocommit 模式
)

checkpointer = PostgresSaver(connection_pool)
checkpointer.setup()              # ★ 首次运行自动建表（checkpoints 等）

app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_123_task_456"}}
app.invoke({"task": "写一段代码"}, config=config)
```

### 异步版：AsyncPostgresSaver（配合 FastAPI 等异步框架）

```python
import asyncio
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def main():
    async with AsyncConnectionPool(
        conninfo="postgresql://username:password@localhost:5432/langgraph_db",
        max_size=20,
        kwargs={"autocommit": True},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.asetup()          # 异步建表

        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "async_thread_001"}}
        await app.ainvoke({"task": "异步任务"}, config=config)  # 注意是 ainvoke

asyncio.run(main())
```

### Redis Checkpointer

状态不需要永久保留、追求极低读写延迟的场景，Redis 很合适（社区/官方扩展包 `langgraph-checkpoint-redis`）：

```bash
pip install langgraph-checkpoint-redis
```

```python
from langgraph.checkpoint.redis import RedisSaver

with RedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    checkpointer.setup()
    app = workflow.compile(checkpointer=checkpointer)
```

Redis 的杀手锏是**键过期（TTL）**：给 checkpoint 键设置 7 天过期，一周内没有交互的任务流自动回收，完全不用写清理脚本（是否支持初始化传 TTL 参数以所用版本文档为准）。

### 生产部署建议

- **状态清理策略**：图每走一步都会写一条 checkpoint（这是时间旅行的前提），不做干预表会迅速膨胀。定期清理已完结且久远的历史（第九节展开）；
- **容器化编排**：LangGraph 服务和数据库容器编排在一起时，务必确认数据库 Volume 正确挂载，否则重启丢状态；
- **并发控制**：同一个 `thread_id` 下 LangGraph 默认不允许并发执行（会抛错）。多个用户/协作者操作同一任务流时，要分配不同的 thread_id 或加分布式锁。

> 📌 **我的选型**：博客 RAG 这种量级（单实例、几百次对话），我用的是 `AsyncSqliteSaver`——单文件零运维，`thread_id` 按会话持久化天然实现多轮记忆。**用对工具，而不是用最重的工具**；哪天真要多副本部署，再换 Postgres 也就是改一行 checkpointer 的事。

## 八、时间旅行：回滚、分支与平行宇宙

有了持久化，每个 checkpoint 都成了可回到的历史节点。LangGraph 的「时间旅行」允许你：查看历史状态、回退到任意节点重新执行、甚至在历史状态上改数据开辟新分支——像 Git 的 checkout 和 branch。

三个核心 API：

- `app.get_state_history(config)`：列出该 thread 的所有历史状态（时间倒序）；
- `app.update_state(config, values, as_node=...)`：修改某个状态；
- `app.invoke(None, config)`：从指定状态继续执行。

```python
config = {"configurable": {"thread_id": "task_123"}}

# ---------------------------------------------------------
# 1. 获取所有历史状态（最新在最前）
# ---------------------------------------------------------
history = list(app.get_state_history(config))
for state in history:
    # StateSnapshot 包含：
    # - values:  当时的数据字典
    # - next:    下一个要执行的节点
    # - config:  唯一标识（含 thread_id 和 checkpoint_id）
    print(f"Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
    print(f"Next Node: {state.next}")
    print(f"Content: {state.values.get('content')}\n")

# ---------------------------------------------------------
# 2. 选定要回退到的历史状态（如倒数第 3 个）
# ---------------------------------------------------------
past_config = history[2].config

# ---------------------------------------------------------
# 3. 时间旅行（两种模式）
# ---------------------------------------------------------
# 模式 A：原样重放——从该历史状态继续往下执行
#         （会 fork 出新分支，不会覆盖原未来）
# new_state = app.invoke(None, config=past_config)

# 模式 B：篡改历史再执行（最常用）
forked_config = app.update_state(
    past_config,
    {"content": "这是我穿越回过去手动修改的新内容"},
    as_node="reviewer",   # 假装是 reviewer 节点更新的（决定状态合并视角）
)

print("=== 在平行宇宙中继续执行 ===")
final_state = app.invoke(None, config=forked_config)
print(final_state["content"])
```

### 分支管理逻辑（Git 隐喻）

理解时间旅行必须理解 LangGraph 状态的**不可变性（Immutability）**：

- **历史不可覆写**：`update_state` 修改过去的节点时，不是删掉重写，而是基于那个历史点创建一个**全新的 checkpoint_id**（thread_id 不变）；
- **多时间线共存**：此时该 thread 下存在多条时间线。用基础 config 取状态时，默认返回**最新创建**的那条分支。

```
checkpoint_1 → checkpoint_2 → checkpoint_3（原时间线）
                    │
                    └→ checkpoint_4 → checkpoint_5（你 fork 的新时间线）★最新
```

## 九、Checkpoint 清理：别让状态表撑爆磁盘

大量时间旅行 + 分支之后，数据库里会堆满无用状态。LangGraph 崇尚不可变，框架层面没有单条时间线的「垃圾回收」API，但有三条主流清理策略。

### 策略一：API 级——删除整个 Thread

任务已走到 END 且确认不再需要溯源，最安全的做法是级联删除整个 thread 的所有分支和检查点：

```python
# PostgresSaver 等官方后端已实现；删除 thread_id 下的全部状态
checkpointer.delete_thread("task_123")
```

### 策略二：SQL 级——定期裁剪历史（PostgreSQL）

只想保留每个 thread 最新 N 个 checkpoint、清掉废弃分支？直接在数据库层跑定时任务（pg_cron 或应用侧定时器）。Postgres 后端主要涉及三张表：`checkpoints`、`checkpoint_blobs`、`checkpoint_writes`。

```sql
-- 每个 thread 只保留最新 5 个 checkpoint
WITH RankedCheckpoints AS (
    SELECT
        thread_id,
        checkpoint_id,
        ROW_NUMBER() OVER(PARTITION BY thread_id ORDER BY checkpoint_id DESC) AS rn
    FROM checkpoints
)
DELETE FROM checkpoints
WHERE (thread_id, checkpoint_id) IN (
    SELECT thread_id, checkpoint_id
    FROM RankedCheckpoints
    WHERE rn > 5
);
```

> ⚠️ **务必先验证级联约束**：确认 `checkpoint_blobs`、`checkpoint_writes` 配置了 `ON DELETE CASCADE`，或在同一事务里同步删除——否则删了元数据、留下大体积 blob，磁盘根本没释放。上线前先在测试环境跑一遍。

也可以按时间做 TTL：删除 30 天前创建且已被覆盖的旧 checkpoint，只保留活跃数据。

### 策略三：Redis TTL——让过期机制自动回收

用 Redis 后端时给 checkpoint 键设置 TTL（如 7 天），一周内无交互的任务流自动消失，零脚本维护。适合「状态只在任务生命周期内有价值」的场景。

## 写在最后

一张图总结 LangGraph 的能力阶梯：

```
概念（LangChain vs LangGraph：链 vs 图）
   → 建图（State / Node / Edge / 条件路由 + 熔断）
   → 协作（Generator × Reviewer 生成-批判循环）
   → 介入（Checkpointer + interrupt 断点，人工审批/润色后恢复）
   → 持久（MemorySaver → PostgresSaver/RedisSaver，生产级）
   → 回滚（get_state_history + update_state，时间旅行与分支）
   → 治理（delete_thread / SQL 裁剪 / TTL，控制状态膨胀）
```

核心思想三条：

1. **LangChain 管组件，LangGraph 管编排**——砖块水泥和建筑蓝图的关系，不是替代而是分层；
2. **循环 + 熔断是智能体的灵魂**——能打回重做才有「自我修正」，有 loop_count 上限才不会烧钱死循环；
3. **状态是一等公民**——有了可持久化的全局状态，才有了人工介入、断点恢复、时间旅行这些传统链式架构做梦都做不到的能力。

这些内容全部来自我给自己的博客搭建 RAG 问答助手的真实实践：检索-判分-改写循环、json_mode 宽容解析、SQLite Checkpointer 多轮记忆，每一处选型都真实踩过。如果只想快速上手，把第五节的 Generator-Reviewer 代码跑通，你就已经理解了 LangGraph 80% 的核心。
