from types import SimpleNamespace
from uuid import uuid4

from src.services.context_manager import (
    count_context_tokens,
    select_old_messages_to_summarize,
)
import pytest
from src.core.exceptions import ContextBudgetExceeded
from src.services.context_manager import validate_current_message

class FakeLLM:
    """Deterministic tokenizer substitute for unit tests."""
    def get_num_tokens(self, text: str) -> int:
        return len(text.split())


def make_message(role: str, content: str):
    return SimpleNamespace(
        id=uuid4(),
        role=role,
        content=content,
    )


def test_selects_oldest_messages_by_token_budget():
    llm = FakeLLM()

    first = make_message("user", "old message one")
    second = make_message("assistant", "old response two")
    current = make_message("user", "current question")

    history = [first, second, current]

    selected = select_old_messages_to_summarize(
        llm=llm,
        history=history,
        current_message_id=current.id,
        target_tokens=7,
    )

    assert selected == [first, second]
    assert current not in selected


def test_does_not_select_current_message():
    llm = FakeLLM()

    current = make_message("user", "current question")

    selected = select_old_messages_to_summarize(
        llm=llm,
        history=[current],
        current_message_id=current.id,
        target_tokens=100,
    )

    assert selected == []


def test_context_token_count_includes_summary_and_history():
    llm = FakeLLM()
    message = make_message("user", "remember this fact")

    count = count_context_tokens(
        llm=llm,
        summary="previous important fact",
        history=[message],
    )

    assert count > 0


def test_rejects_oversized_current_message(monkeypatch):
    import src.services.context_manager as manager

    monkeypatch.setattr(manager, "CONTEXT_TOKEN_BUDGET", 10)
    monkeypatch.setattr(manager, "CONTEXT_SAFETY_MARGIN", 2)

    llm = FakeLLM()

    with pytest.raises(ContextBudgetExceeded):
        validate_current_message(
            llm,
            "one two three four five six seven eight",
        )


def test_accepts_message_within_budget(monkeypatch):
    import src.services.context_manager as manager

    monkeypatch.setattr(manager, "CONTEXT_TOKEN_BUDGET", 10)
    monkeypatch.setattr(manager, "CONTEXT_SAFETY_MARGIN", 2)

    llm = FakeLLM()

    validate_current_message(llm, "one two three")