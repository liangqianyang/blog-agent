"""问答图节点：retrieve → grade_documents → generate / rewrite_query。

自定义流事件（stream_mode="custom"，单参数 dict 带 type 判别）：
- {"type": "status",  "phase": "rewrite", "query": ...}
- {"type": "sources", "sources": [{title, url, heading, score}]}
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.graph.state import AgentState

logger = logging.getLogger("blog_agent.graph")


class GradeDocuments(BaseModel):
    """文档相关性批量判分。小模型常输出裸数组 [1,2] 而非对象，before 钩子宽容包装。"""

    relevant_indices: list[int] = Field(default_factory=list, description="与问题相关的文档编号列表")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        if isinstance(data, list):
            return {"relevant_indices": data}
        if isinstance(data, dict) and "relevant_indices" not in data:
            for v in data.values():
                if isinstance(v, list):
                    return {"relevant_indices": v}
        return data


class RewrittenQuery(BaseModel):
    """检索词改写。同样宽容裸字符串输出。"""

    query: str = Field(description="改写后的检索词")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        if isinstance(data, str):
            return {"query": data}
        return data


_GRADE_SYSTEM = """你是检索结果相关性评估器。给定用户问题和一组编号文档，判断每个文档是否能帮助回答该问题。
只输出与问题实质相关的文档编号；全都不相关就给空列表。
只输出 JSON 对象 {"relevant_indices": [编号, ...]}，不要输出任何其他内容。"""

_REWRITE_SYSTEM = """你是检索查询优化器。用户的问题在知识库检索中未命中有效内容。
请改写检索词：保留核心意图，补充同义词、技术术语或更具体的表达。
只输出 JSON 对象 {"query": "改写后的检索词"}，不要输出任何其他内容。"""

_RAG_SYSTEM = """你是这个博客的 AI 助手，基于下面的参考文章回答用户问题。要求：
- 用与用户提问相同的语言回答（默认中文）
- 回答忠于参考资料，引用时用 [编号] 标注（如 [1]），编号对应参考资料序号
- 参考资料为空或与问题无关时，直接说明"博客现有文章中暂时没有找到相关内容"，可基于常识简要补充，但要明确区分哪些内容不是来自博客
- 涉及代码时用 markdown 代码块展示
- 组织清晰：适当使用小标题和列表，不要冗长"""


def make_nodes(indexer, grader_llm, generator_llm, settings):
    """闭包注入依赖，返回节点函数表。

    indexer: ArticleIndexer；grader_llm / generator_llm: ChatOpenAI（非流式 / 流式）。
    """

    async def retrieve(state: AgentState) -> dict:
        points = await indexer.search(state["query"], settings.retrieval_top_k)
        docs = indexer.to_documents(points)
        logger.info("retrieve %r -> %d docs", state["query"], len(docs))
        return {"documents": docs}

    async def grade_documents(state: AgentState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"documents": []}
        numbered = "\n\n".join(
            f"[{i}] {d.metadata.get('article_title', '')} | {d.metadata.get('heading_path', '')}\n{d.page_content[:600]}"
            for i, d in enumerate(docs, start=1)
        )
        # json_mode：glm-4-flash 的 function calling 参数易畸形（实测），json_object 响应稳定
        structured = grader_llm.with_structured_output(GradeDocuments, method="json_mode")
        try:
            grade: GradeDocuments = await structured.ainvoke(
                [
                    SystemMessage(content=_GRADE_SYSTEM),
                    HumanMessage(content=f"用户问题：{state['query']}\n\n文档：\n{numbered}"),
                ]
            )
            keep = [docs[i - 1] for i in (grade.relevant_indices or []) if 1 <= i <= len(docs)]
            logger.info("grade: %d/%d relevant", len(keep), len(docs))
            return {"documents": keep}
        except Exception:  # noqa: BLE001 —— 小模型结构化输出不稳：判分失败时保守保留全部文档
            logger.warning("grade failed, keeping all %d docs", len(docs), exc_info=True)
            return {"documents": docs}

    async def rewrite_query(state: AgentState) -> dict:
        structured = grader_llm.with_structured_output(RewrittenQuery, method="json_mode")
        try:
            result: RewrittenQuery = await structured.ainvoke(
                [
                    SystemMessage(content=_REWRITE_SYSTEM),
                    HumanMessage(content=f"原始问题：{state['query']}"),
                ]
            )
            new_query = result.query.strip() or state["query"]
        except Exception:  # noqa: BLE001 —— 结构化输出失败：沿用原检索词（计入次数，循环有界）
            logger.warning("rewrite failed, keeping original query", exc_info=True)
            new_query = state["query"]
        logger.info("rewrite %r -> %r", state["query"], new_query)
        _emit({"type": "status", "phase": "rewrite", "query": new_query})
        return {"query": new_query, "rewrite_count": state.get("rewrite_count", 0) + 1}

    async def generate(state: AgentState) -> dict:
        docs = state["documents"]
        # 来源按文章去重：同文章多块只保留得分最高的一块作为引用
        seen: dict[str, dict] = {}
        for d in docs:
            key = d.metadata.get("article_id", "")
            entry = {
                "title": d.metadata.get("article_title", ""),
                "url": d.metadata.get("article_url", ""),
                "heading": d.metadata.get("heading_path", ""),
                "score": round(float(d.metadata.get("score", 0.0)), 3),
            }
            if key not in seen or entry["score"] > seen[key]["score"]:
                seen[key] = entry
        sources = list(seen.values())
        _emit({"type": "sources", "sources": sources})

        context = (
            "\n\n".join(
                f"[{i}] {d.metadata.get('article_title', '')} | {d.metadata.get('heading_path', '')}\n{d.page_content}"
                for i, d in enumerate(docs, start=1)
            )
            if docs
            else "（无相关参考资料）"
        )
        # 检索上下文只进本次调用的 system，不污染持久化历史
        messages = [SystemMessage(content=f"{_RAG_SYSTEM}\n\n参考资料：\n{context}"), *state["messages"]]
        ai_msg = await generator_llm.ainvoke(messages)
        return {"messages": [ai_msg], "sources": sources}

    def _emit(payload: dict) -> None:
        from langgraph.config import get_stream_writer

        try:
            get_stream_writer()(payload)
        except RuntimeError:  # 非流式调用（CLI 调试）时无 writer，忽略
            pass

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "rewrite_query": rewrite_query,
        "generate": generate,
    }
