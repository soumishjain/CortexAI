
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import select

from src.db.models import Conversation, Message
from src.services import chat_service


class FakeLLM:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    async def ainvoke(self, context):
        if self.should_fail:
            raise RuntimeError("Simulated Gemini failure")

        return SimpleNamespace(content="Hello from CortexAI!")


async def fake_build_context(**kwargs):
    return []


@pytest.mark.anyio
async def test_chat_persists_messages_in_sequence(
    db_session, monkeypatch
):
    monkeypatch.setattr(chat_service, "llm", FakeLLM())
    monkeypatch.setattr(
        chat_service, "build_context", fake_build_context
    )

    result = await chat_service.generate_reply(
        message="Hello",
        conversation_id=None,
        db=db_session,
    )

    conversation_id = result["conversation_id"]
    assert isinstance(conversation_id, UUID)
    assert result["reply"] == "Hello from CortexAI!"

    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.sequence)
    )
    messages = list(
        (await db_session.execute(query)).scalars().all()
    )

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[0].sequence == 1

    assert messages[1].role == "assistant"
    assert messages[1].content == "Hello from CortexAI!"
    assert messages[1].sequence == 2

    conversation = await db_session.get(
        Conversation, conversation_id
    )
    assert conversation.next_message_sequence == 3


@pytest.mark.anyio
async def test_user_message_survives_llm_failure(
    db_session, monkeypatch
):
    monkeypatch.setattr(
        chat_service, "llm", FakeLLM(should_fail=True)
    )
    monkeypatch.setattr(
        chat_service, "build_context", fake_build_context
    )

    with pytest.raises(
        RuntimeError, match="Simulated Gemini failure"
    ):
        await chat_service.generate_reply(
            message="Remember this input",
            conversation_id=None,
            db=db_session,
        )

    query = (
        select(Conversation)
        .where(Conversation.title == "Remember this input")
    )
    conversation = (
        await db_session.execute(query)
    ).scalar_one()

    messages_query = (
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.sequence)
    )
    messages = list(
        (
            await db_session.execute(messages_query)
        ).scalars().all()
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Remember this input"
    assert messages[0].sequence == 1
    assert conversation.next_message_sequence == 2
