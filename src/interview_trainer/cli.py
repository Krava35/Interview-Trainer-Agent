"""CLI-запуск чатового Interview Trainer."""

from __future__ import annotations

import argparse

from .chatbot import InterviewChatbot


def parse_args() -> argparse.Namespace:
    """Парсит аргументы CLI для запуска чатового интервью."""
    parser = argparse.ArgumentParser(
        description="Мультиагентный Interview Trainer в чат-режиме.",
    )
    parser.add_argument(
        "--position",
        default="Python Backend Developer",
        help="Целевая позиция кандидата.",
    )
    parser.add_argument(
        "--stack",
        default="Python, FastAPI, PostgreSQL",
        help="Технологический стек кандидата.",
    )
    parser.add_argument(
        "--level",
        default="junior",
        help="Уровень кандидата (junior/middle/senior/intern).",
    )
    parser.add_argument(
        "--goals",
        default="Подготовиться к техническому интервью.",
        help="Цель подготовки.",
    )
    parser.add_argument(
        "--query",
        default="Начать тренировочное интервью",
        help="Стартовый запрос для Router-агента.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Лимит внутренних retry-циклов evaluator->interviewer.",
    )
    return parser.parse_args()


def main() -> None:
    """Запускает интерактивный чат Interview Trainer из CLI."""
    args = parse_args()
    chatbot = InterviewChatbot(
        position=args.position,
        stack=args.stack,
        level=args.level,
        goals=args.goals,
        query=args.query,
        max_retries=args.max_retries,
    )
    chatbot.run_interactive()


if __name__ == "__main__":
    main()
