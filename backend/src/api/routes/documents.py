
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.rag.ingestion import ingest_pdf
from src.db.models import Document
from src.db.session import get_db

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    temp_path = None
    document = None

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="PDF exceeds the 20 MB size limit.",
            )

        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Create a stable document ID before indexing.
        document = Document(
            id=uuid4(),
            filename=Path(file.filename).name,
            status="uploaded",
            chunks_indexed=0,
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        result = await ingest_pdf(
            file_path=temp_path,
            db=db,
            document_id=document.id,
        )

        return result

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        await db.rollback()
        print(f"Document ingestion error: {type(exc).__name__}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the PDF.",
        ) from exc

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

        await file.close()
