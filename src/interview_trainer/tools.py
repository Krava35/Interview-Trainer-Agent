"""Инструменты (tools) для мультиагентного Interview Trainer."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import BaseTool, tool

_SESSION_STARTS: dict[str, float] = {}


def _utc_now_iso() -> str:
    """Возвращает текущее время в ISO-формате UTC."""
    return datetime.now(timezone.utc).isoformat()


@tool
def session_timer_logger_tool(
    action: str,
    session_id: str,
    payload: str = "",
) -> dict[str, Any]:
    """Управляет таймером сессии интервью и возвращает запись журнала."""
    normalized_action = action.strip().lower()
    event: dict[str, Any] = {
        "tool": "session_timer_logger_tool",
        "action": normalized_action,
        "session_id": session_id,
        "payload": payload,
        "timestamp": _utc_now_iso(),
    }

    if normalized_action == "start":
        if session_id not in _SESSION_STARTS:
            _SESSION_STARTS[session_id] = time.monotonic()
            status = "started"
        else:
            status = "already_started"
        event["status"] = status
        return {"status": status, "event": event}

    if normalized_action == "stop":
        started_at = _SESSION_STARTS.pop(session_id, None)
        if started_at is None:
            status = "not_started"
            elapsed_seconds = 0.0
        else:
            status = "stopped"
            elapsed_seconds = round(time.monotonic() - started_at, 3)
        event["status"] = status
        event["elapsed_seconds"] = elapsed_seconds
        return {"status": status, "elapsed_seconds": elapsed_seconds, "event": event}

    if normalized_action == "event":
        event["status"] = "logged"
        return {"status": "logged", "event": event}

    event["status"] = "error"
    event["error"] = "Неизвестное действие. Используйте start, stop или event."
    return {"status": "error", "event": event}


@tool
def rubric_generator_tool(position: str, stack: str, level: str) -> dict[str, Any]:
    """Генерирует рубрику оценивания ответа кандидата под профиль вакансии."""
    normalized_level = (level or "unknown").strip().lower()
    position_value = position.strip() if position else "не указана"
    stack_value = stack.strip() if stack else "не указан"

    by_level_depth_weight = {
        "intern": 15,
        "junior": 20,
        "middle": 30,
        "senior": 40,
        "unknown": 25,
    }
    depth_weight = by_level_depth_weight.get(normalized_level, 25)

    rubric = {
        "profile": {
            "position": position_value,
            "stack": stack_value,
            "level": normalized_level,
        },
        "criteria": [
            {
                "name": "Понимание концепций",
                "weight": 30,
                "description": "Насколько кандидат корректно объясняет ключевые принципы.",
            },
            {
                "name": "Практическое применение",
                "weight": 30,
                "description": "Умение применять подходы в реальных задачах.",
            },
            {
                "name": "Глубина по уровню",
                "weight": depth_weight,
                "description": "Соответствие глубины ответа целевому уровню позиции.",
            },
            {
                "name": "Ясность коммуникации",
                "weight": max(100 - (60 + depth_weight), 10),
                "description": "Структурированность, точность формулировок и логика ответа.",
            },
        ],
    }
    return rubric


@tool
def weak_topics_counter_tool(evaluation_result: str) -> dict[str, Any]:
    """Считает частоту слабых тем по JSON-результату оценки кандидата."""
    weak_topics: list[str] = []
    try:
        parsed = json.loads(evaluation_result)
        value = parsed.get("weak_topics", [])
        if isinstance(value, list):
            weak_topics = [str(item).strip() for item in value if str(item).strip()]
    except json.JSONDecodeError:
        candidates = [chunk.strip() for chunk in evaluation_result.split(",")]
        weak_topics = [item for item in candidates if item]

    counts: dict[str, int] = {}
    for topic in weak_topics:
        key = topic.lower()
        counts[key] = counts.get(key, 0) + 1
    return {"weak_topic_counts": counts}


def invoke_tool(tool_obj: BaseTool, **kwargs: Any) -> dict[str, Any]:
    """Вызывает tool через интерфейс LangChain и приводит ответ к словарю."""
    raw_result = tool_obj.invoke(kwargs)
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"result": raw_result}
    return {"result": raw_result}


__all__ = [
    "invoke_tool",
    "rubric_generator_tool",
    "session_timer_logger_tool",
    "weak_topics_counter_tool",
]
