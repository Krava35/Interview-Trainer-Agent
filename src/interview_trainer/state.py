"""Определение состояния графа Interview Trainer."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

RouteType = Literal["interview", "evaluate_only", "plan_only", "summary"]


class InterviewTrainerState(TypedDict, total=False):
    """Состояние мультиагентного графа для тренировки собеседований."""

    query: str
    route: RouteType
    session_id: str

    position: str
    stack: str
    level: str
    goals: str

    interview_question: str
    candidate_answer: str
    current_question: str
    current_question_type: str
    current_code_snippet: str
    current_expected_points: list[str]
    asked_questions: list[dict[str, Any]]
    qa_history: list[dict[str, Any]]

    rubric: dict[str, Any]
    evaluation: dict[str, Any]
    score: int
    skill_scores: dict[str, int]
    weak_topics: list[str]
    weak_topic_counts: dict[str, int]
    recommendations: list[str]
    last_feedback: str
    summary_report: dict[str, Any]

    coach_plan: str
    final_response: str

    memory_snapshot: dict[str, Any]
    session_history: list[dict[str, Any]]

    tool_events: Annotated[list[dict[str, Any]], operator.add]
    activated_nodes: Annotated[list[str], operator.add]

    should_retry: bool
    loop_count: int
    max_retries: int
