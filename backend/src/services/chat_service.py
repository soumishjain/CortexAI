
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.db.models import Conversation
from src.services.context_manager import build_context
from src.services.message_service import add_message
from src.agents.graph import agent_graph


llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
)


def extract_text(content) -> str:
    """Normalize the model response into plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()

    return str(content).strip()


async def generate_reply(
    message: str,
    conversation_id: UUID | None,
    db: AsyncSession,
    document_id: UUID | None = None,
) -> dict:
    try:
        # 1. Find an existing conversation or create a new one.
        if conversation_id is not None:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id
                )
            )
            conversation = result.scalar_one_or_none()

            if conversation is None:
                raise ValueError("Conversation not found.")

        else:
            conversation = Conversation(
                title=message[:50] or "New conversation"
            )
            db.add(conversation)
            await db.flush()

        # 2. Persist the user message with its sequence number.
        user_message = await add_message(
            db=db,
            conversation_id=conversation.id,
            role="user",
            content=message,
        )
        await db.commit()

        # 3. Build the context using the rolling summary and recent history.
        context = await build_context(
            llm=llm,
            conversation=conversation,
            db=db,
            current_message_id=user_message.id,
        )

        result = await agent_graph.ainvoke({
            "query": message,
            "context": context,
            "document_id": (
                str(document_id) if document_id else None
            ),
            "route": "direct",
            "reason": "",
            "answer": "",
            "sources": [],
        })

        reply = result["answer"]
        sources = result.get("sources", [])

        if not reply:
            raise RuntimeError("The agent returned an empty response.")

        if not reply:
            raise RuntimeError("The AI returned an empty response.")

        # 5. Persist the assistant message with the next sequence number.
        await add_message(
            db=db,
            conversation_id=conversation.id,
            role="assistant",
            content=reply,
        )
        await db.commit()

        return {
            "conversation_id": conversation.id,
            "reply": reply,
            "sources" : sources,
        }

    except ValueError:
        await db.rollback()
        raise

    except Exception:
        await db.rollback()
        raise
