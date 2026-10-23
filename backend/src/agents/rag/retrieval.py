from uuid import UUID

from src.agents.rag.vector_store import vector_store


def retrieve_document_chunks(
    query: str,
    document_id: UUID | None = None,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve relevant PDF chunks from Chroma.

    If document_id is supplied, search only that document.
    Otherwise, search across all indexed documents.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10.")

    search_kwargs = {"k": k}

    if document_id is not None:
        search_kwargs["filter"] = {
            "document_id": str(document_id)
        }

    results = vector_store.similarity_search(
        query,
        **search_kwargs,
    )

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "document_id": doc.metadata.get("document_id"),
        }
        for doc in results
    ]