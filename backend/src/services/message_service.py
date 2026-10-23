
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Conversation, Message


async def add_message(
    db: AsyncSession,
    conversation_id: UUID,
    role: str,
    content: str,
) -> Message:
    """
    Allocate a sequence number while locking the conversation row.
    Caller owns the transaction and must commit or roll back.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .with_for_update()
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise ValueError("Conversation not found.")

    sequence = conversation.next_message_sequence
    conversation.next_message_sequence = sequence + 1

    message = Message(
        conversation_id=conversation.id,
        sequence=sequence,
        role=role,
        content=content,
    )

    db.add(message)
    await db.flush()

    return message
