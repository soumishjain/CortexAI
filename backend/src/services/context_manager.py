from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from src.db.models import Conversation, Message
from src.core.config import settings
from src.core.exceptions import ContextBudgetExceeded

CONTEXT_TOKEN_BUDGET = settings.CONTEXT_TOKEN_BUDGET
CONTEXT_SAFETY_MARGIN = settings.CONTEXT_SAFETY_MARGIN
SUMMARY_TARGET_TOKENS = 4000

SYSTEM_PROMPT = """
You are CortexAI, an agentic research and knowledge assistant.
Use the supplied conversation context when answering.
If required information is missing, ask a clarifying question.
"""
def validate_current_message(llm,message: str) -> None:
    available_budget = (
        CONTEXT_TOKEN_BUDGET - CONTEXT_SAFETY_MARGIN
    )

    if available_budget <= 0:
        raise RuntimeError("Invalid context budget configuration.")

    token_count = llm.get_num_tokens(message)

    if token_count >= available_budget:
        raise ContextBudgetExceeded(
            "Your message is too large for the available context budget."
        )

def to_langchain_message(message:Message):
    if message.role == "user":
        return HumanMessage(content=message.content)

    if message.role == "assistant":
        return AIMessage(content=message.content)

    return SystemMessage(content=message.content)

def count_context_tokens(llm, summary: str | None, history: list[Message]) -> int: 
    parts = [SYSTEM_PROMPT]

    if summary: 
        parts.append(
            "Summary of earlier conversation:\n" + summary
        )

    for message in history:
        parts.append(f"{message.role}:{message.content}")

    return llm.get_num_tokens("\n\n".join(parts))



async def load_unsummarized_messages(
    conversation: Conversation,
    db: AsyncSession,
) -> list[Message]:
    query = select(Message).where(
        Message.conversation_id == conversation.id
    )

    if conversation.summary_through_sequence is not None:
        query = query.where(
            Message.sequence > conversation.summary_through_sequence
        )

    query = query.order_by(Message.sequence)

    result = await db.execute(query)
    return list(result.scalars().all())


def select_old_messages_to_summarize(
        llm,
        history: list[Message],
        current_message_id,
        target_tokens: int,
) -> list[Message]:
    """
    Select the oldest prefix by token cost.
    Never include the current user message.
    """
    selected = []
    selected_tokens = 0

    for message in history:
        if message.id == current_message_id:
            break

        message_text = f"{message.role}: {message.content}"
        message_tokens = llm.get_num_tokens(message_text)

        selected.append(message)
        selected_tokens += message_tokens

        if selected_tokens >= target_tokens:
            break

    return selected

async def update_summary(
        llm,existing_summary: str | None,
        messages_to_summarize: list[Message],
) -> str:
    transcript = "\n".join(
        f"{message.role.upper()}: {message.content}"
        for message in messages_to_summarize
    )

    prompt = f"""
Update the persistent summary of this conversation.

Preserve important user preferences, project details, decisions,
constraints, unresolved questions, and facts needed in future turns.
Remove repetition and obsolete details.
Do not invent facts. Preserve uncertainty where appropriate.
Keep the summary concise, ideally under 700 tokens.

Existing summary:
{existing_summary or "(No previous summary)"}

New conversation messages:
{transcript}

Return only the updated summary.
"""

    response = await llm.ainvoke(prompt)
    content = response.content

    if isinstance(content, str):
        summary = content.strip()
    elif isinstance(content, list):
        summary = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        ).strip()
    else:
        summary = ""

    if not summary:
        raise RuntimeError("Summary generation returned empty content.")

    return summary

async def build_context(
        llm, 
        conversation: Conversation,
        db: AsyncSession,
        current_message_id,
) -> list:
    history = await load_unsummarized_messages(conversation, db)

    current_message = next(
        (m for m in history if m.id == current_message_id),
        None,
    )

    if current_message is None:
        raise RuntimeError(
            "Current message is missing from conversation history."
        )

    validate_current_message(llm,current_message.content)
    
    effective_budget= (
        CONTEXT_TOKEN_BUDGET - CONTEXT_SAFETY_MARGIN
    )

    while count_context_tokens(
        llm, conversation.summary, history
    ) > effective_budget:
        total_tokens = count_context_tokens(
            llm, conversation.summary, history
        )
        target_tokens = min(
            SUMMARY_TARGET_TOKENS,
            max(512, total_tokens - effective_budget + 1000),
        )

        old_messages = select_old_messages_to_summarize(
            llm=llm,
            history=history,
            current_message_id=current_message_id,
            target_tokens=target_tokens
        )

        if not old_messages:
            raise RuntimeError(
                "Context exceeds budget, but no older messages "
                "can be summarized. The current message must be "
                "shortened or rejected."
            )

        new_summary = await update_summary(
            llm=llm,
            existing_summary=conversation.summary,
            messages_to_summarize=old_messages,
        )

        previous_summary = conversation.summary
        previous_cursor = conversation.summary_through_sequence

        try:
            conversation.summary = new_summary
            conversation.summary_through_sequence = old_messages[-1].sequence
            await db.commit()
        except Exception:
            await db.rollback()
            conversation.summary = previous_summary
            conversation.summary_through_sequence = previous_cursor
            raise

        # Remove only the messages included in the new summary.
        history = history[len(old_messages):]

    context = [
        SystemMessage(
            content=(
                SYSTEM_PROMPT
                + (
                    "\n\nSummary of earlier conversation:\n"
                    + conversation.summary
                    if conversation.summary
                    else ""
                )
            )
        )
    ]

    context.extend(
        to_langchain_message(message)
        for message in history
    )

    return context