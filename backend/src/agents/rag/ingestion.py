
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.rag.pdf_loader import load_pdf
from src.agents.rag.vector_store import index_document
from src.db.models import Document


async def ingest_pdf(
    file_path: str | Path,
    db: AsyncSession,
    document_id: UUID,
) -> dict:
    """
    Extract, chunk, and index a PDF while tracking its status
    in PostgreSQL.
    """

    path = Path(file_path)

    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if document is None:
        raise ValueError("Document record not found.")

    document.status = "processing"
    await db.commit()

    try:
        chunks = load_pdf(path)

        for chunk in chunks:
            chunk.metadata["document_id"] = str(document_id)
            chunk.metadata["source"] = document.filename

        indexed_count = index_document(
            document_id=document_id,
            chunks=chunks,
        )

        document.status = "indexed"
        document.chunks_indexed = indexed_count

        await db.commit()

        return {
            "document_id": str(document.id),
            "filename": document.filename,
            "chunks_indexed": indexed_count,
            "status": document.status,
        }

    except Exception:
        await db.rollback()

        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if document is not None:
            document.status = "failed"
            await db.commit()

        raise
