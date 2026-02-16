"""Узлы LangGraph для мультиагентного Interview Trainer."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .memory_store import (
    append_session,
    apply_weak_topic_counts,
    load_memory,
    read_memory_snapshot,
    save_memory,
    update_profile,
)
from .prompts import (
    COACH_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    INTERVIEWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from .state import InterviewTrainerState, RouteType
from .tools import (
    invoke_tool,
    rubric_generator_tool,
    session_timer_logger_tool,
    weak_topics_counter_tool,
)

DEFAULT_MODEL_NAME = "qwen3-32b"
QUESTION_TYPE_ROTATION = [
    "conceptual",
    "design",
    "what_if",
    "debugging",
    "testing",
    "performance",
    "sql",
]
QUESTION_TYPE_ALIASES = {
    "architecture": "design",
    "code_review": "debugging",
    "scenario": "what_if",
}


def _memory_path() -> str:
    """Возвращает путь до файла памяти из переменной окружения или значение по умолчанию."""
    return os.getenv("INTERVIEW_MEMORY_PATH", "data/memory.json")


def _build_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Создает LLM-клиент для Qwen через OpenAI-совместимый vLLM endpoint."""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
        api_key=os.getenv("LITELLM_API_KEY", ""),
        base_url=os.getenv("LITELLM_BASE_URL"),
        temperature=temperature,
    )


def _content_to_text(content: Any) -> str:
    """Преобразует произвольный формат контента LangChain в строку."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    return str(content)


def _parse_json(text: str) -> dict[str, Any]:
    """Пытается извлечь JSON из текста модели и вернуть словарь."""
    cleaned = text.strip()
    if not cleaned:
        return {}

    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _invoke_json_agent(
    *,
    system_prompt: str,
    payload: dict[str, Any],
    fallback: dict[str, Any],
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Вызывает LLM-агента и гарантированно возвращает словарь JSON."""
    try:
        llm = _build_llm(temperature=temperature)
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        parsed = _parse_json(_content_to_text(response.content))
        if parsed:
            return parsed
    except Exception as error:  # noqa: BLE001
        fallback = {**fallback, "fallback_reason": str(error)}
    return fallback


def _heuristic_route(query: str) -> RouteType:
    """Определяет маршрут без модели, если LLM недоступна."""
    lowered = query.lower()
    if any(token in lowered for token in ("summary", "суммар", "итог", "сводка")):
        return "summary"
    if any(token in lowered for token in ("оцени", "оценка", "мой ответ")):
        return "evaluate_only"
    if any(token in lowered for token in ("план подготовки", "план", "roadmap")):
        return "plan_only"
    return "interview"


def _sanitize_route(value: str | None, fallback_query: str) -> RouteType:
    """Приводит route к допустимому значению."""
    if value in {"interview", "evaluate_only", "plan_only", "summary"}:
        return value
    return _heuristic_route(fallback_query)


def _int_score(value: Any, default: int = 50) -> int:
    """Преобразует значение score к целому числу в диапазоне 0..100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _as_string_list(value: Any) -> list[str]:
    """Преобразует произвольный объект в список непустых строк."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _question_fingerprint(text: str) -> str:
    """Нормализует вопрос для проверки на дубли."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"[^\w\s]", "", normalized)


def _is_question_duplicate(
    candidate: str,
    previous_questions: list[str],
    threshold: float = 0.88,
) -> bool:
    """Проверяет, что новый вопрос не совпадает и не слишком похож на предыдущие."""
    candidate_fp = _question_fingerprint(candidate)
    if not candidate_fp:
        return False
    for previous in previous_questions:
        previous_fp = _question_fingerprint(previous)
        if not previous_fp:
            continue
        if candidate_fp == previous_fp:
            return True
        if SequenceMatcher(None, candidate_fp, previous_fp).ratio() >= threshold:
            return True
    return False


def _sanitize_question_type(value: Any, default: str = "conceptual") -> str:
    """Приводит тип вопроса к поддерживаемому набору значений."""
    normalized = str(value or "").strip().lower()
    normalized = QUESTION_TYPE_ALIASES.get(normalized, normalized)
    if normalized in QUESTION_TYPE_ROTATION:
        return normalized
    return default


def _extract_previous_questions(state: InterviewTrainerState) -> list[str]:
    """Достает недавние вопросы из состояния сессии без дублей."""
    collected: list[str] = []

    asked_questions = state.get("asked_questions", [])
    if isinstance(asked_questions, list):
        for item in asked_questions[-10:]:
            if isinstance(item, dict):
                question = str(item.get("question", "")).strip()
            else:
                question = str(item).strip()
            if question:
                collected.append(question)

    qa_history = state.get("qa_history", [])
    if isinstance(qa_history, list):
        for item in qa_history[-10:]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if question:
                collected.append(question)

    deduplicated: list[str] = []
    seen_fingerprints: set[str] = set()
    for question in collected:
        fingerprint = _question_fingerprint(question)
        if not fingerprint or fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        deduplicated.append(question)
    return deduplicated[-10:]


def _extract_previous_question_types(state: InterviewTrainerState) -> list[str]:
    """Извлекает типы уже заданных вопросов из текущей сессии."""
    raw_items = state.get("asked_questions", [])
    if not isinstance(raw_items, list):
        return []
    question_types: list[str] = []
    for item in raw_items[-10:]:
        if not isinstance(item, dict):
            continue
        question_type = _sanitize_question_type(item.get("question_type", ""))
        if question_type:
            question_types.append(question_type)
    return question_types


def _pick_desired_question_type(loop_count: int, previous_types: list[str]) -> str:
    """Выбирает целевой тип вопроса с ротацией и без повтора прошлого типа."""
    base_index = max(loop_count - 1, 0) % len(QUESTION_TYPE_ROTATION)
    desired = QUESTION_TYPE_ROTATION[base_index]
    if previous_types:
        last_type = _sanitize_question_type(previous_types[-1], default=desired)
        if desired == last_type:
            desired = QUESTION_TYPE_ROTATION[(base_index + 1) % len(QUESTION_TYPE_ROTATION)]
    return desired


def _fallback_question_bundle(
    state: InterviewTrainerState,
    question_type: str,
) -> dict[str, Any]:
    """Готовит fallback-вопрос под выбранный тип, включая код для debugging."""
    position = state.get("position", "Software Engineer")
    stack = state.get("stack", "Python, SQL, Git")
    if question_type == "debugging":
        return {
            "interview_question": (
                "Найдите проблему в коде и объясните, почему он падает на части входов. "
                "Как бы вы исправили решение?"
            ),
            "expected_points": [
                "Точно указать место ошибки",
                "Объяснить, на каких входах воспроизводится сбой",
                "Предложить корректное исправление и тест-кейс",
            ],
            "difficulty": "medium",
            "question_type": "debugging",
            "code_snippet": (
                "def extract_user_id(payload: dict) -> int:\n"
                "    return int(payload[\"user\"][\"id\"])\n\n"
                "print(extract_user_id({\"profile\": {\"id\": \"42\"}}))"
            ),
        }
    if question_type == "what_if":
        return {
            "interview_question": (
                f"Что изменится в вашем решении для роли {position}, если нагрузка вырастет "
                "в 10 раз и SLA по latency станет в 2 раза жестче?"
            ),
            "expected_points": [
                "Узкие места и стратегия масштабирования",
                "Компромиссы по consistency/latency/cost",
                "Метрики и план проверки изменений",
            ],
            "difficulty": "hard",
            "question_type": "what_if",
            "code_snippet": "",
        }
    if question_type == "design":
        return {
            "interview_question": (
                f"Спроектируйте сервис под стек {stack}: опишите ключевые компоненты, "
                "границы ответственности и отказоустойчивость."
            ),
            "expected_points": [
                "Компоненты и их взаимодействия",
                "Хранение данных и кэширование",
                "Наблюдаемость и деградационные сценарии",
            ],
            "difficulty": "medium",
            "question_type": "design",
            "code_snippet": "",
        }
    if question_type == "testing":
        return {
            "interview_question": (
                "Как бы вы построили стратегию тестирования критичного API-эндпоинта: "
                "какие уровни тестов и какие риски они закрывают?"
            ),
            "expected_points": [
                "Пирамида тестирования и границы уровней",
                "Позитивные/негативные кейсы",
                "Интеграционные проверки и стабильность CI",
            ],
            "difficulty": "medium",
            "question_type": "testing",
            "code_snippet": "",
        }
    if question_type == "performance":
        return {
            "interview_question": (
                "Опишите, как вы бы провели performance-анализ медленного endpoint: "
                "какие метрики и шаги оптимизации примените?"
            ),
            "expected_points": [
                "Измерение baseline и профилирование",
                "Оптимизация БД/кода/сетевых вызовов",
                "Проверка результата под нагрузкой",
            ],
            "difficulty": "medium",
            "question_type": "performance",
            "code_snippet": "",
        }
    if question_type == "sql":
        return {
            "interview_question": (
                "В таблице заказов нужно быстро получать последние 20 заказов пользователя "
                "и фильтровать по статусу. Какие индексы и SQL-подход вы выберете?"
            ),
            "expected_points": [
                "Подходящий составной индекс",
                "Обоснование порядка полей в индексе",
                "Проверка плана запроса и риски деградации",
            ],
            "difficulty": "medium",
            "question_type": "sql",
            "code_snippet": "",
        }
    return {
        "interview_question": (
            f"Объясните, как вы бы спроектировали сервис для роли {position} "
            "с учетом отказоустойчивости и наблюдаемости."
        ),
        "expected_points": [
            "Четкая структура ответа",
            "Компромиссы по архитектуре",
            "Метрики и мониторинг",
        ],
        "difficulty": "medium",
        "question_type": "conceptual",
        "code_snippet": "",
    }


def _normalize_skill_scores(
    value: Any,
    rubric: dict[str, Any],
) -> dict[str, int]:
    """Нормализует оценки навыков в диапазон 0..100 для всех критериев рубрики."""
    criteria_names: list[str] = []
    for criterion in rubric.get("criteria", []):
        name = str(criterion.get("name", "")).strip()
        if name:
            criteria_names.append(name)

    normalized: dict[str, int] = {}
    if isinstance(value, dict):
        for key, raw_score in value.items():
            if not str(key).strip():
                continue
            normalized[str(key).strip()] = _int_score(raw_score, default=0)

    for name in criteria_names:
        normalized.setdefault(name, 0)
    return normalized


def router_node(state: InterviewTrainerState) -> InterviewTrainerState:
    """Классифицирует запрос и подготавливает профиль для остальных агентов."""
    query = state.get("query", "")
    memory_snapshot = read_memory_snapshot(_memory_path())
    profile = memory_snapshot.get("profile", {})
    fallback = {
        "route": state.get("route", _heuristic_route(query)),
        "position": state.get("position", "") or profile.get("position", ""),
        "stack": state.get("stack", "") or profile.get("stack", ""),
        "level": state.get("level", "") or profile.get("level", ""),
        "goals": state.get("goals", "") or profile.get("goals", ""),
        "candidate_answer": state.get("candidate_answer", ""),
    }

    router_data = _invoke_json_agent(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        payload={
            "query": query,
            "current_profile": profile,
            "known_topic_stats": memory_snapshot.get("topic_stats", {}),
        },
        fallback=fallback,
        temperature=0.0,
    )

    explicit_route = state.get("route")
    if explicit_route in {"interview", "evaluate_only", "plan_only", "summary"}:
        route = explicit_route
    else:
        route = _sanitize_route(str(router_data.get("route")), query)
    position = (
        state.get("position")
        or str(router_data.get("position", "")).strip()
        or str(profile.get("position", "")).strip()
        or "Software Engineer"
    )
    stack = (
        state.get("stack")
        or str(router_data.get("stack", "")).strip()
        or str(profile.get("stack", "")).strip()
        or "Python, SQL, Git"
    )
    level = (
        state.get("level")
        or str(router_data.get("level", "")).strip()
        or str(profile.get("level", "")).strip()
        or "junior"
    )
    goals = (
        state.get("goals")
        or str(router_data.get("goals", "")).strip()
        or str(profile.get("goals", "")).strip()
        or "Пройти собеседование уверенно и структурированно."
    )

    answer_from_router = str(router_data.get("candidate_answer", "")).strip()
    candidate_answer = state.get("candidate_answer", "") or answer_from_router
    session_id = state.get("session_id", "") or str(uuid.uuid4())

    return {
        "route": route,
        "position": position,
        "stack": stack,
        "level": level,
        "goals": goals,
        "candidate_answer": candidate_answer,
        "session_id": session_id,
        "memory_snapshot": memory_snapshot,
        "max_retries": state.get("max_retries", 1),
        "loop_count": state.get("loop_count", 0),
        "activated_nodes": ["router"],
    }


def interviewer_node(state: InterviewTrainerState) -> InterviewTrainerState:
    """Генерирует интервью-вопрос с учетом профиля и слабых тем."""
    loop_count = int(state.get("loop_count", 0)) + 1
    session_id = state.get("session_id", str(uuid.uuid4()))
    weak_topics = _as_string_list(state.get("weak_topics", []))
    previous_questions = _extract_previous_questions(state)
    previous_question_types = _extract_previous_question_types(state)
    desired_question_type = _pick_desired_question_type(loop_count, previous_question_types)

    action = "start" if loop_count == 1 else "event"
    timer_result = invoke_tool(
        session_timer_logger_tool,
        action=action,
        session_id=session_id,
        payload=f"interviewer_loop_{loop_count}",
    )

    fallback = _fallback_question_bundle(state, desired_question_type)
    interviewer_data: dict[str, Any] = fallback
    temperature_schedule = (0.4, 0.6, 0.8)

    for attempt, temperature in enumerate(temperature_schedule, start=1):
        interviewer_data = _invoke_json_agent(
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            payload={
                "route": state.get("route"),
                "position": state.get("position"),
                "stack": state.get("stack"),
                "level": state.get("level"),
                "goals": state.get("goals"),
                "weak_topics": weak_topics,
                "history": state.get("session_history", [])[-3:],
                "loop_count": loop_count,
                "previous_questions": previous_questions,
                "previous_question_types": previous_question_types,
                "desired_question_type": desired_question_type,
                "allow_code_snippet": True,
                "retry_attempt": attempt,
            },
            fallback=fallback,
            temperature=temperature,
        )
        candidate_question = str(
            interviewer_data.get("interview_question", fallback["interview_question"])
        ).strip()
        if not _is_question_duplicate(candidate_question, previous_questions):
            break

    interview_question = str(interviewer_data.get("interview_question", "")).strip()
    if not interview_question:
        interview_question = fallback["interview_question"]
    if _is_question_duplicate(interview_question, previous_questions):
        interview_question = (
            f"{interview_question}\n\n"
            f"Доп. условие: раскройте новый аспект решения для шага {loop_count}."
        )
    expected_points = _as_string_list(interviewer_data.get("expected_points"))
    if not expected_points:
        expected_points = fallback["expected_points"]
    question_type = _sanitize_question_type(
        interviewer_data.get("question_type"),
        default=desired_question_type,
    )
    code_snippet = str(interviewer_data.get("code_snippet", "")).strip()
    if question_type == "debugging" and not code_snippet:
        code_snippet = str(fallback.get("code_snippet", "")).strip()

    return {
        "interview_question": interview_question,
        "current_question": interview_question,
        "question_type": question_type,
        "current_question_type": question_type,
        "code_snippet": code_snippet,
        "current_code_snippet": code_snippet,
        "current_expected_points": expected_points,
        "loop_count": loop_count,
        "activated_nodes": ["interviewer"],
        "tool_events": [timer_result.get("event", {"tool": "session_timer_logger_tool"})],
    }


def evaluator_node(state: InterviewTrainerState) -> InterviewTrainerState:
    """Оценивает ответ по рубрике и определяет, нужен ли повторный цикл."""
    tool_events: list[dict[str, Any]] = []
    rubric = state.get("rubric")
    if not rubric:
        rubric = invoke_tool(
            rubric_generator_tool,
            position=state.get("position", ""),
            stack=state.get("stack", ""),
            level=state.get("level", ""),
        )
        tool_events.append(
            {
                "tool": "rubric_generator_tool",
                "status": "generated",
                "rubric_profile": rubric.get("profile", {}),
            }
        )

    fallback = {
        "score": 45,
        "skill_scores": {
            "Понимание концепций": 50,
            "Практическое применение": 40,
            "Глубина по уровню": 35,
            "Ясность коммуникации": 55,
        },
        "strengths": ["Ответ демонстрирует базовые знания по теме."],
        "weaknesses": ["Недостаточно деталей и практических примеров."],
        "weak_topics": ["архитектурные компромиссы", "проектирование API"],
        "recommendations": [
            "Добавьте пошаговый алгоритм решения задачи.",
            "Приведите короткий практический пример из опыта.",
        ],
        "feedback": (
            "Добавьте структуру: контекст, решение, компромиссы, проверка результата. "
            "Приведите 1-2 практических кейса."
        ),
        "should_retry": True,
        "follow_up_question": (
            "Опишите, как вы выбрали бы формат хранения данных и почему."
        ),
    }

    evaluator_data = _invoke_json_agent(
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        payload={
            "question": state.get("current_question", state.get("interview_question", "")),
            "question_type": state.get("current_question_type", ""),
            "code_snippet": state.get("current_code_snippet", ""),
            "candidate_answer": state.get("candidate_answer", ""),
            "rubric": rubric,
            "loop_count": state.get("loop_count", 1),
            "max_retries": state.get("max_retries", 1),
        },
        fallback=fallback,
        temperature=0.0,
    )

    score = _int_score(evaluator_data.get("score"), default=fallback["score"])
    skill_scores = _normalize_skill_scores(
        evaluator_data.get("skill_scores", fallback["skill_scores"]),
        rubric,
    )
    weak_topics = _as_string_list(evaluator_data.get("weak_topics"))
    if not weak_topics and score < 70:
        weak_topics = ["глубина технического объяснения"]
    recommendations = _as_string_list(evaluator_data.get("recommendations"))
    if not recommendations:
        recommendations = fallback["recommendations"]

    should_retry_raw = bool(evaluator_data.get("should_retry", score < 70))
    max_retries = int(state.get("max_retries", 1))
    loop_count = int(state.get("loop_count", 1))
    should_retry = (
        state.get("route") == "interview"
        and should_retry_raw
        and score < 70
        and loop_count <= max_retries
    )

    weak_topic_counts_result = invoke_tool(
        weak_topics_counter_tool,
        evaluation_result=json.dumps(
            {"weak_topics": weak_topics},
            ensure_ascii=False,
        ),
    )
    weak_topic_counts = weak_topic_counts_result.get("weak_topic_counts", {})
    tool_events.append(
        {
            "tool": "weak_topics_counter_tool",
            "status": "updated",
            "weak_topic_counts": weak_topic_counts,
        }
    )

    evaluation = {
        "score": score,
        "skill_scores": skill_scores,
        "strengths": _as_string_list(evaluator_data.get("strengths")),
        "weaknesses": _as_string_list(evaluator_data.get("weaknesses")),
        "weak_topics": weak_topics,
        "recommendations": recommendations,
        "feedback": str(evaluator_data.get("feedback", "")).strip(),
        "should_retry": should_retry,
        "follow_up_question": str(evaluator_data.get("follow_up_question", "")).strip(),
    }

    qa_history = list(state.get("qa_history", []))
    candidate_answer = str(state.get("candidate_answer", "")).strip()
    if candidate_answer:
        qa_history.append(
            {
                "question": state.get("current_question", state.get("interview_question", "")),
                "question_type": state.get("current_question_type", ""),
                "code_snippet": state.get("current_code_snippet", ""),
                "answer": candidate_answer,
                "score": score,
                "skill_scores": skill_scores,
                "strengths": evaluation["strengths"],
                "weaknesses": evaluation["weaknesses"],
                "weak_topics": weak_topics,
                "recommendations": recommendations,
                "feedback": evaluation["feedback"],
            }
        )

    return {
        "rubric": rubric,
        "evaluation": evaluation,
        "score": score,
        "skill_scores": skill_scores,
        "weak_topics": weak_topics,
        "weak_topic_counts": weak_topic_counts,
        "recommendations": recommendations,
        "last_feedback": evaluation["feedback"],
        "qa_history": qa_history,
        "should_retry": should_retry,
        "activated_nodes": ["evaluator"],
        "tool_events": tool_events,
    }


def coach_node(state: InterviewTrainerState) -> InterviewTrainerState:
    """Формирует персональный план прокачки на основе результата оценки."""
    fallback = {
        "coach_plan": (
            "1) Повторите слабые темы через короткие практические задачи.\n"
            "2) Каждый день тренируйте структурный ответ по схеме: контекст -> решение -> "
            "компромиссы -> проверка.\n"
            "3) Через 3 дня проведите повторный мини-опрос по тем же темам."
        ),
        "weekly_actions": [
            "Решать по 2 задачки в день по слабым темам.",
            "Раз в 2 дня проговаривать ответы вслух 15 минут.",
            "Вести заметки с ошибками и корректными формулировками.",
        ],
        "resources": [
            "Документация по ключевым технологиям стека.",
            "Список типовых interview-вопросов для выбранной позиции.",
        ],
        "next_check": "Через 5 дней повторить вопрос и сравнить score.",
    }

    coach_data = _invoke_json_agent(
        system_prompt=COACH_SYSTEM_PROMPT,
        payload={
            "route": state.get("route"),
            "position": state.get("position"),
            "stack": state.get("stack"),
            "level": state.get("level"),
            "goals": state.get("goals"),
            "score": state.get("score", 0),
            "weak_topics": state.get("weak_topics", []),
            "evaluation": state.get("evaluation", {}),
            "memory_snapshot": state.get("memory_snapshot", {}),
        },
        fallback=fallback,
        temperature=0.2,
    )

    coach_plan = str(coach_data.get("coach_plan", "")).strip() or fallback["coach_plan"]
    weekly_actions = _as_string_list(coach_data.get("weekly_actions")) or fallback["weekly_actions"]
    resources = _as_string_list(coach_data.get("resources")) or fallback["resources"]
    next_check = str(coach_data.get("next_check", "")).strip() or fallback["next_check"]

    timer_stop_result = invoke_tool(
        session_timer_logger_tool,
        action="stop",
        session_id=state.get("session_id", "unknown"),
        payload="coach_completed",
    )

    score = state.get("score", 0)
    question = state.get("interview_question", "Вопрос не сформирован.")
    feedback = state.get("evaluation", {}).get("feedback", "")
    weak_topics = state.get("weak_topics", [])

    final_response = (
        "=== Итог интервью-тренировки ===\n"
        f"Маршрут: {state.get('route', 'interview')}\n"
        f"Позиция: {state.get('position', '')}\n"
        f"Стек: {state.get('stack', '')}\n"
        f"Уровень: {state.get('level', '')}\n\n"
        f"Вопрос интервью:\n{question}\n\n"
        f"Оценка: {score}/100\n"
        f"Слабые темы: {', '.join(weak_topics) if weak_topics else 'не выявлены'}\n"
        f"Фидбек: {feedback or 'нет'}\n\n"
        "План прокачки:\n"
        f"{coach_plan}\n\n"
        "Действия на неделю:\n"
        + "\n".join(f"- {item}" for item in weekly_actions)
        + "\n\nРесурсы:\n"
        + "\n".join(f"- {item}" for item in resources)
        + f"\n\nСледующая проверка: {next_check}"
    )

    return {
        "coach_plan": coach_plan,
        "final_response": final_response,
        "activated_nodes": ["coach"],
        "tool_events": [timer_stop_result.get("event", {"tool": "session_timer_logger_tool"})],
    }


def memory_update_node(state: InterviewTrainerState) -> InterviewTrainerState:
    """Обновляет постоянную память профиля, слабых тем и истории сессий."""
    memory_path = _memory_path()
    memory = load_memory(memory_path)

    update_profile(
        memory,
        position=state.get("position", ""),
        stack=state.get("stack", ""),
        level=state.get("level", ""),
        goals=state.get("goals", ""),
    )
    apply_weak_topic_counts(memory, state.get("weak_topic_counts", {}))

    append_session(
        memory,
        {
            "session_id": state.get("session_id", ""),
            "query": state.get("query", ""),
            "route": state.get("route", ""),
            "score": state.get("score", 0),
            "weak_topics": state.get("weak_topics", []),
            "activated_nodes": state.get("activated_nodes", []),
            "tool_events": state.get("tool_events", []),
        },
    )
    save_memory(memory, memory_path)

    session_history = memory.get("sessions", [])[-5:]
    final_response = (
        state.get("final_response", "")
        + "\n\n[Память] Сессия сохранена. "
        + f"Всего сессий: {len(memory.get('sessions', []))}."
    )

    return {
        "session_history": session_history,
        "memory_snapshot": read_memory_snapshot(memory_path),
        "final_response": final_response,
        "activated_nodes": ["memory_update"],
    }


def summarize_session_node(state: InterviewTrainerState) -> InterviewTrainerState:
    """Строит summary только по ответам текущей сессии (qa_history)."""
    qa_history = list(state.get("qa_history", []))
    answered = [
        item
        for item in qa_history
        if str(item.get("answer", "")).strip() and str(item.get("question", "")).strip()
    ]
    if not answered:
        message = (
            "Вы еще не ответили на вопросы интервью. "
            "Сначала запросите вопрос и дайте хотя бы один ответ."
        )
        return {
            "summary_report": {
                "has_answers": False,
                "questions_answered": 0,
                "message": message,
            },
            "final_response": message,
            "activated_nodes": ["summary"],
        }

    score_sum = 0
    strengths_counter: Counter[str] = Counter()
    weaknesses_counter: Counter[str] = Counter()
    recommendations_counter: Counter[str] = Counter()
    skill_totals: dict[str, int] = {}
    skill_counts: dict[str, int] = {}

    for item in answered:
        score_sum += _int_score(item.get("score", 0), default=0)
        for value in _as_string_list(item.get("strengths", [])):
            strengths_counter[value] += 1
        for value in _as_string_list(item.get("weaknesses", [])):
            weaknesses_counter[value] += 1
        for value in _as_string_list(item.get("recommendations", [])):
            recommendations_counter[value] += 1

        raw_skill_scores = item.get("skill_scores", {})
        if isinstance(raw_skill_scores, dict):
            for skill_name, raw_value in raw_skill_scores.items():
                skill = str(skill_name).strip()
                if not skill:
                    continue
                skill_totals[skill] = skill_totals.get(skill, 0) + _int_score(raw_value, default=0)
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

    questions_answered = len(answered)
    average_score = round(score_sum / questions_answered, 1)
    average_skill_scores: dict[str, int] = {}
    for skill, total in skill_totals.items():
        average_skill_scores[skill] = int(round(total / max(skill_counts.get(skill, 1), 1)))

    top_strengths = [item for item, _ in strengths_counter.most_common(5)]
    top_weaknesses = [item for item, _ in weaknesses_counter.most_common(5)]
    top_recommendations = [item for item, _ in recommendations_counter.most_common(7)]

    summary_report = {
        "has_answers": True,
        "questions_answered": questions_answered,
        "average_score": average_score,
        "skill_scores": average_skill_scores,
        "strengths": top_strengths,
        "weaknesses": top_weaknesses,
        "recommendations": top_recommendations,
    }

    skills_block = "\n".join(
        f"- {skill}: {score}/100"
        for skill, score in sorted(average_skill_scores.items(), key=lambda item: item[0])
    )
    strengths_block = "\n".join(f"- {item}" for item in top_strengths) or "- Нет данных"
    weaknesses_block = "\n".join(f"- {item}" for item in top_weaknesses) or "- Нет данных"
    recommendations_block = (
        "\n".join(f"- {item}" for item in top_recommendations) or "- Нет данных"
    )

    final_response = (
        "=== Суммаризация текущего интервью ===\n"
        f"Отвечено на вопросов: {questions_answered}\n"
        f"Средний общий score: {average_score}/100\n\n"
        "Оценка по скиллам:\n"
        f"{skills_block if skills_block else '- Нет данных'}\n\n"
        "Сильные стороны:\n"
        f"{strengths_block}\n\n"
        "Слабые стороны:\n"
        f"{weaknesses_block}\n\n"
        "Что подучить и на что обратить внимание:\n"
        f"{recommendations_block}"
    )

    return {
        "summary_report": summary_report,
        "skill_scores": average_skill_scores,
        "final_response": final_response,
        "activated_nodes": ["summary"],
    }
