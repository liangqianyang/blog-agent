"""LangGraph 状态定义。"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """问答图状态。

    - messages：多轮对话历史（add_messages 合并 + checkpointer 持久化）
    - query：当前检索词（可能被 rewrite 改写；生成回答以用户原话为准）
    - documents：检索并过滤后的文档
    - sources：本次回答的引用来源（{title, url, heading, score}）
    - rewrite_count：已改写次数（防死循环）
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: list[Document]
    sources: list[dict]
    rewrite_count: int
