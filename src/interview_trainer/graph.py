"""Сборка графа LangGraph для Interview Trainer."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    coach_node,
    evaluator_node,
    interviewer_node,
    memory_update_node,
    router_node,
    summarize_session_node,
)
from .state import InterviewTrainerState


def _route_after_router(state: InterviewTrainerState) -> str:
    """Определяет следующий узел после Router."""
    if state.get("route") == "summary":
        return "summary"
    if state.get("route") == "plan_only":
        return "coach"
    if state.get("route") == "evaluate_only":
        return "evaluator"
    return "interviewer"


def _route_after_evaluator(state: InterviewTrainerState) -> str:
    """Запускает повторный цикл при низком балле и доступном лимите retry."""
    return "retry" if state.get("should_retry", False) else "coach"


def build_graph():
    """Создает и компилирует LangGraph workflow Interview Trainer."""
    builder = StateGraph(InterviewTrainerState)

    builder.add_node("router", router_node)
    builder.add_node("interviewer", interviewer_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("coach", coach_node)
    builder.add_node("summary", summarize_session_node)
    builder.add_node("memory_update", memory_update_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "summary": "summary",
            "interviewer": "interviewer",
            "evaluator": "evaluator",
            "coach": "coach",
        },
    )
    builder.add_edge("interviewer", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {
            "retry": "interviewer",
            "coach": "coach",
        },
    )
    builder.add_edge("summary", END)
    builder.add_edge("coach", "memory_update")
    builder.add_edge("memory_update", END)

    return builder.compile()
