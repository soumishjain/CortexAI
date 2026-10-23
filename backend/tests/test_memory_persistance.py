import pytest
from sqlalchemy import select

from src.db.models import Conversation, Message
from src.services.context_manager import load_unsummarized_messages


@pytest.mark.anyio
async def test_summary_and_cursor_persist_together(db_session):
    conversation = Conversation(
    title="Memory test",
    next_message_sequence=3,
    )
    db_session.add(conversation)
    await db_session.flush()

    first_message = Message(
        conversation_id=conversation.id,
        role="user",
        sequence=1,
        content="I am building CortexAI.",
    )
    second_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        sequence=2,
        content="We are using FastAPI.",
    )

    db_session.add_all([first_message, second_message])
    await db_session.flush()

    conversation.summary = (
        "The user is building CortexAI with FastAPI."
    )
    conversation.summary_through_message_id = first_message.id

    await db_session.commit()

    result = await db_session.execute(
        select(Conversation).where(
            Conversation.id == conversation.id
        )
    )
    saved = result.scalar_one()

    assert saved.summary == (
        "The user is building CortexAI with FastAPI."
    )
    assert saved.summary_through_message_id == first_message.id

@pytest.mark.anyio
async def test_cursor_returns_only_unsummarized_messages(db_session):
    conversation = Conversation(
    title="Cursor test",
    summary_through_sequence=2,
    next_message_sequence=4,
    )
    db_session.add(conversation)
    await db_session.flush()

    first = Message(
        conversation_id=conversation.id,
        role="user",
        sequence=1,
        content="I am building CortexAI.",
    )

    second = Message(
        conversation_id=conversation.id,
        role="assistant",
        sequence=2,
        content="We will use FastAPI.",
    )

    third = Message(
        conversation_id=conversation.id,
        role="user",
        sequence=3,
        content="Now add persistent memory.",
    )

    db_session.add_all([first, second, third])
    await db_session.flush()

    # The summary covers the first two messages.
    conversation.summary = (
        "The user is building CortexAI with FastAPI."
    )
    conversation.summary_through_message_id = second.id

    await db_session.commit()
    await db_session.refresh(conversation)

    remaining = await load_unsummarized_messages(
        conversation,
        db_session,
    )

    assert [message.id for message in remaining] == [third.id]