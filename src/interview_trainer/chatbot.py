"""Чатовый интерфейс для мультиагентного Interview Trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .nodes import (
    evaluator_node,
    interviewer_node,
    memory_update_node,
    router_node,
    summarize_session_node,
)
from .state import InterviewTrainerState

EXIT_COMMANDS = {"stop", "exit", "quit", "выход"}
SUMMARY_COMMANDS = {"summary", "sum", "итог", "сводка", "суммаризация"}
NEXT_COMMANDS = {"next", "далее", "еще", "ещё", "вопрос", "следующий"}


@dataclass
class InterviewChatbot:
    """Управляет сессией чат-интервью поверх мультиагентных узлов."""

    position: str
    stack: str
    level: str
    goals: str
    query: str = "Начать тренировочное интервью"
    max_retries: int = 0
    state: InterviewTrainerState = field(default_factory=dict)

    @staticmethod
    def _question_fingerprint(question: str) -> str:
        """Строит простой отпечаток вопроса для дедупликации."""
        normalized = re.sub(r"\s+", " ", question.lower()).strip()
        return re.sub(r"[^\w\s]", "", normalized)

    def start_session(self) -> InterviewTrainerState:
        """Инициализирует сессию и профиль кандидата через Router."""
        base_state: InterviewTrainerState = {
            "query": self.query,
            "route": "interview",
            "position": self.position,
            "stack": self.stack,
            "level": self.level,
            "goals": self.goals,
            "candidate_answer": "",
            "current_question": "",
            "current_question_type": "",
            "current_code_snippet": "",
            "current_expected_points": [],
            "asked_questions": [],
            "max_retries": self.max_retries,
            "loop_count": 0,
            "qa_history": [],
            "activated_nodes": [],
            "tool_events": [],
        }
        routed_state = router_node(base_state)
        self.state = {
            **base_state,
            **routed_state,
            "route": "interview",
            "qa_history": [],
        }
        return self.state

    def _ensure_session(self) -> None:
        """Гарантирует, что сессия инициализирована перед шагами интервью."""
        if not self.state:
            self.start_session()

    def ask_next_question(self) -> str:
        """Запрашивает у Interviewer следующий вопрос."""
        self._ensure_session()
        step_state: InterviewTrainerState = {
            **self.state,
            "route": "interview",
            "candidate_answer": "",
        }
        result = interviewer_node(step_state)
        question = str(
            result.get("current_question", result.get("interview_question", ""))
        ).strip()
        question_type = str(
            result.get(
                "current_question_type",
                result.get("question_type", "conceptual"),
            )
        ).strip()
        code_snippet = str(
            result.get("current_code_snippet", result.get("code_snippet", ""))
        ).strip()
        asked_questions = list(self.state.get("asked_questions", []))
        if question:
            asked_questions.append(
                {
                    "question": question,
                    "question_type": question_type,
                    "code_snippet": code_snippet,
                    "fingerprint": self._question_fingerprint(question),
                }
            )
        self.state = {
            **self.state,
            **result,
            "current_question": question,
            "interview_question": question,
            "current_question_type": question_type,
            "current_code_snippet": code_snippet,
            "asked_questions": asked_questions,
        }
        return question

    def submit_answer(self, answer_text: str) -> dict[str, Any]:
        """Оценивает ответ кандидата и обновляет историю текущей сессии."""
        self._ensure_session()
        answer = answer_text.strip()
        if not answer:
            raise ValueError("Ответ не должен быть пустым.")
        if not str(self.state.get("current_question", "")).strip():
            raise RuntimeError(
                "Сейчас нет активного вопроса. Запросите `next`."
            )

        step_state: InterviewTrainerState = {
            **self.state,
            "route": "interview",
            "candidate_answer": answer,
        }
        eval_state = evaluator_node(step_state)
        self.state = {
            **self.state,
            **eval_state,
            "candidate_answer": answer,
            "current_question": "",
            "current_question_type": "",
            "current_code_snippet": "",
            "current_expected_points": [],
        }

        # Сохраняем долговременную память отдельно от summary
        # по текущей сессии.
        memory_state = memory_update_node(self.state)
        self.state["memory_snapshot"] = memory_state.get(
            "memory_snapshot",
            self.state.get("memory_snapshot", {}),
        )
        self.state["session_history"] = memory_state.get(
            "session_history",
            self.state.get("session_history", []),
        )

        evaluation = self.state.get("evaluation", {})
        return {
            "score": self.state.get("score", 0),
            "skill_scores": self.state.get("skill_scores", {}),
            "strengths": evaluation.get("strengths", []),
            "weaknesses": evaluation.get("weaknesses", []),
            "recommendations": evaluation.get("recommendations", []),
            "feedback": evaluation.get("feedback", ""),
            "follow_up_question": evaluation.get("follow_up_question", ""),
        }

    def get_summary(self) -> dict[str, Any]:
        """Возвращает summary только по ответам текущей сессии."""
        self._ensure_session()
        summary_state = summarize_session_node(self.state)
        self.state = {**self.state, **summary_state}
        return {
            "summary_report": self.state.get("summary_report", {}),
            "final_response": self.state.get("final_response", ""),
        }

    @staticmethod
    def format_question(
        question: str,
        expected_points: list[str] | None = None,
        code_snippet: str | None = None,
    ) -> str:
        """Форматирует вопрос и ожидаемые пункты ответа для вывода."""
        points = expected_points or []
        lines = ["=== ВОПРОС ===", question]
        snippet = (code_snippet or "").strip()
        if snippet:
            lines.extend(["", "Код:", "```python", snippet, "```"])
        if points:
            lines.append("")
            lines.append("Что желательно раскрыть в ответе:")
            lines.extend(f"- {point}" for point in points)
        return "\n".join(lines)

    @staticmethod
    def format_evaluation(result: dict[str, Any]) -> str:
        """Форматирует оценку ответа для консольного и notebook вывода."""
        lines = [f"=== ОЦЕНКА ОТВЕТА ===\nScore: {result.get('score', 0)}/100"]

        skill_scores = result.get("skill_scores", {})
        if isinstance(skill_scores, dict) and skill_scores:
            lines.append("\nОценка по скиллам:")
            for skill_name, skill_score in skill_scores.items():
                lines.append(f"- {skill_name}: {skill_score}/100")

        strengths = result.get("strengths", []) or ["Нет данных"]
        weaknesses = result.get("weaknesses", []) or ["Нет данных"]
        recommendations = result.get("recommendations", []) or ["Нет данных"]

        lines.append("\nСильные стороны:")
        lines.extend(f"- {item}" for item in strengths)
        lines.append("\nСлабые стороны:")
        lines.extend(f"- {item}" for item in weaknesses)
        lines.append("\nРекомендации:")
        lines.extend(f"- {item}" for item in recommendations)

        feedback = str(result.get("feedback", "")).strip()
        if feedback:
            lines.append("\nФидбек:")
            lines.append(feedback)

        follow_up = str(result.get("follow_up_question", "")).strip()
        if follow_up:
            lines.append("\nУточняющий вопрос от Evaluator:")
            lines.append(follow_up)

        return "\n".join(lines)

    def run_interactive(self) -> None:
        """Запускает интерактивный цикл интервью в формате чата."""
        self.start_session()
        print("Сессия инициализирована.")
        print(f"Позиция: {self.state.get('position')}")
        print(f"Стек: {self.state.get('stack')}")
        print(f"Уровень: {self.state.get('level')}")

        first_question = self.ask_next_question()
        print(
            self.format_question(
                first_question,
                self.state.get("current_expected_points", []),
                self.state.get("current_code_snippet", ""),
            )
        )
        print("\nКоманды: next, summary, stop")

        while True:
            user_input = input("\nВы: ").strip()
            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in EXIT_COMMANDS:
                print("Бот: Сессия завершена.")
                break

            if lowered in SUMMARY_COMMANDS:
                summary = self.get_summary()
                print("\nБот:")
                print(summary.get("final_response", "Нет данных по summary."))
                continue

            if lowered in NEXT_COMMANDS:
                question = self.ask_next_question()
                print(
                    "\nБот:\n"
                    + self.format_question(
                        question,
                        self.state.get("current_expected_points", []),
                        self.state.get("current_code_snippet", ""),
                    )
                )
                continue

            if not str(self.state.get("current_question", "")).strip():
                print(
                    "Бот: Сейчас нет активного вопроса. "
                    "Напишите `next` для нового вопроса."
                )
                continue

            try:
                evaluation = self.submit_answer(user_input)
            except (RuntimeError, ValueError) as error:
                print(f"Бот: Ошибка обработки ответа: {error}")
                continue

            print("\nБот:")
            print(self.format_evaluation(evaluation))
            print(
                "\nБот: Напишите `next` для следующего вопроса, "
                "`summary` для итогов или `stop`."
            )
