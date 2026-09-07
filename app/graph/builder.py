"""图的组装：节点接线 + 条件路由 + checkpointer。

结构：
START → retrieve → grade_documents ─┬ 相关 ≥ MIN_RELEVANT_DOCS → generate → END
              ↑                     ├─ 全不相关且改写次数未满 → rewrite_query → retrieve
              └─────────────────────┴─ 兜底 → generate（提示无资料，仍给出来处）→ END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.graph.nodes import make_nodes
from app.graph.state import AgentState


def _route_after_grade(state: AgentState, min_relevant: int, max_rewrites: int) -> str:
    if len(state.get("documents") or []) >= min_relevant:
        return "generate"
    if state.get("rewrite_count", 0) < max_rewrites:
        return "rewrite"
    return "generate"


def build_graph(indexer, grader_llm, generator_llm, settings, checkpointer: BaseCheckpointSaver | None = None):
    nodes = make_nodes(indexer, grader_llm, generator_llm, settings)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("grade_documents", nodes["grade_documents"])
    graph.add_node("rewrite_query", nodes["rewrite_query"])
    graph.add_node("generate", nodes["generate"])

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: _route_after_grade(state, settings.min_relevant_docs, settings.max_rewrites),
        {"generate": "generate", "rewrite": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":  # 调试用：打印图结构
    graph = build_graph(None, None, None, None)
    print(graph.get_graph().draw_ascii())
