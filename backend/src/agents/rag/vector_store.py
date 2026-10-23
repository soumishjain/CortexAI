from uuid import UUID

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import settings

CHROMA_PERSIST_DIRECTORY = "data/chroma"
COLLECTION_NAME = "cortexai_documents"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GOOGLE_API_KEY,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)

def index_document(document_id: UUID, chunks: list) -> int:
    """Store document chunks and their embeddings in Chroma."""

    if not chunks:
        raise ValueError("Cannot index a document without text chunks.")

    # Assign a stable document ID to every chunk.
    for chunk in chunks:
        chunk.metadata["document_id"] = str(document_id)

    # Deterministic IDs help identify individual chunks.
    chunk_ids = [
        f"{document_id}:{chunk.metadata.get('page', 0)}:{chunk.metadata.get('start_index', i)}"
        for i, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    return len(chunks)
