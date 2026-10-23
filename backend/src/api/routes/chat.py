from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import ContextBudgetExceeded
from src.db.session import get_db
from src.services.chat_service import generate_reply

router = APIRouter(tags=["Chat"])

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    conversation_id: UUID | None = None
    document_id: UUID | None = None

class ChatResponse(BaseModel):
    conversation_id: UUID
    reply: str
    sources: list[dict[str,str | int | None]] = Field(
        default_factory=list
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await generate_reply(
            message=request.message,
            conversation_id=request.conversation_id,
            db=db,
            document_id=request.document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ContextBudgetExceeded as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        await db.rollback()
        print(f"Chat service error: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail="Chat request failed.",
        ) from exc