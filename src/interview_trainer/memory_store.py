"""Утилиты для чтения и обновления памяти Interview Trainer."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_PATH = Path("data/memory.json")
MAX_SESSIONS = 100

DEFAULT_MEMORY: dict[str, Any] = {
    "profile": {
        "position": "",
        "stack": "",
        "level": "",
        "goals": "",
    },
    "topic_stats": {},
    "sessions": [],
}


def _now_iso() -> str:
    """Возвращает текущий момент времени в ISO-формате UTC."""
    return datetime.now(timezone.utc).isoformat()


def ensure_memory_file(path: str | Path = DEFAULT_MEMORY_PATH) -> Path:
    """Создает файл памяти и родительскую директорию, если их еще нет."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(
            json.dumps(DEFAULT_MEMORY, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return target


def load_memory(path: str | Path = DEFAULT_MEMORY_PATH) -> dict[str, Any]:
    """Загружает память из JSON и нормализует обязательные поля."""
    target = ensure_memory_file(path)
    raw = target.read_text(encoding="utf-8").strip()
    if not raw:
        memory = deepcopy(DEFAULT_MEMORY)
        save_memory(memory, target)
        return memory

    data = json.loads(raw)
    memory = deepcopy(DEFAULT_MEMORY)
    memory.update(data)
    memory["profile"] = {**DEFAULT_MEMORY["profile"], **memory.get("profile", {})}
    if not isinstance(memory.get("topic_stats"), dict):
        memory["topic_stats"] = {}
    if not isinstance(memory.get("sessions"), list):
        memory["sessions"] = []
    return memory


def save_memory(memory: dict[str, Any], path: str | Path = DEFAULT_MEMORY_PATH) -> None:
    """Сохраняет память в JSON-файл с UTF-8 и читаемым форматированием."""
    target = ensure_memory_file(path)
    target.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def read_memory_snapshot(path: str | Path = DEFAULT_MEMORY_PATH) -> dict[str, Any]:
    """Возвращает компактный срез памяти для подстановки в state."""
    memory = load_memory(path)
    return {
        "profile": memory.get("profile", {}),
        "topic_stats": memory.get("topic_stats", {}),
        "recent_sessions": memory.get("sessions", [])[-3:],
    }


def update_profile(
    memory: dict[str, Any],
    *,
    position: str = "",
    stack: str = "",
    level: str = "",
    goals: str = "",
) -> None:
    """Обновляет профиль пользователя непустыми значениями из текущего запроса."""
    profile = memory.setdefault("profile", deepcopy(DEFAULT_MEMORY["profile"]))
    if position:
        profile["position"] = position
    if stack:
        profile["stack"] = stack
    if level:
        profile["level"] = level
    if goals:
        profile["goals"] = goals


def apply_weak_topic_counts(memory: dict[str, Any], weak_topic_counts: dict[str, int]) -> None:
    """Инкрементирует счетчики слабых тем в долговременной памяти."""
    topic_stats = memory.setdefault("topic_stats", {})
    for topic, count in weak_topic_counts.items():
        if not topic:
            continue
        topic_stats[topic] = int(topic_stats.get(topic, 0)) + int(count)


def append_session(memory: dict[str, Any], session_record: dict[str, Any]) -> None:
    """Добавляет запись о сессии в историю и ограничивает ее размер."""
    record = {"timestamp": _now_iso(), **session_record}
    sessions = memory.setdefault("sessions", [])
    sessions.append(record)
    if len(sessions) > MAX_SESSIONS:
        memory["sessions"] = sessions[-MAX_SESSIONS:]
