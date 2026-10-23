from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.rag.retrieval import retrieve_document_chunks
from src.agents.router import answer_llm


async def answer_from_document(
    query: str,
    document_id: UUID | None = None,
) -> dict:
    chunks = retrieve_document_chunks(
        query=query,
        document_id=document_id,
        k=5,
    )

    if not chunks:
        return {
            "answer": "No relevant information was found in the indexed documents.",
            "sources": [],
        }

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content:\n{chunk['content']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(
            content=(
                "You are CortexAI, a document question-answering assistant. "
                "Answer using only the supplied document excerpts. "
                "Do not invent facts or citations. "
                "If the excerpts do not contain enough information, "
                "state that clearly. "
                "Cite supporting claims using the format [Source 1], "
                "[Source 2], etc. Use only source numbers provided below."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{query}\n\n"
                f"Document excerpts:\n{context}"
            )
        ),
    ]

    response = await answer_llm.ainvoke(messages)

    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item["text"]
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    sources = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "document_id": chunk["document_id"],
        }
        for chunk in chunks
    ]

    return {
        "answer": str(answer).strip(),
        "sources": sources,
    }