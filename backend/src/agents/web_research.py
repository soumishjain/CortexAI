
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.router import answer_llm
from src.core.config import settings


web_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=False,
)


async def research_web(query: str) -> dict[str, Any]:
    """Search the web and generate an answer grounded in search results."""

    if not settings.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is missing from configuration.")

    results = await web_search.ainvoke({"query": query})

    if not results:
        raise RuntimeError("Web search returned no results.")

    sources = []
    for result in results:
        if not isinstance(result, dict):
            continue

        url = result.get("url")
        content = result.get("content", "")

        if url and content:
            sources.append({
                "url": url,
                "content": content,
            })

    if not sources:
        raise RuntimeError("No usable web sources were returned.")

    research_context = "\n\n".join(
        f"Source URL: {source['url']}\n"
        f"Source content: {source['content']}"
        for source in sources
    )

    response = await answer_llm.ainvoke([
        SystemMessage(
            content=(
                "You are CortexAI's web research agent. "
                "Answer using the supplied search results. "
                "Do not invent facts or claim that a source supports "
                "something absent from its content. Cite relevant sources "
                "using their exact URLs in Markdown links. "
                "If evidence is insufficient or conflicting, say so."
            )
        ),
        HumanMessage(
            content=(
                f"Research question:\n{query}\n\n"
                f"Search results:\n{research_context}"
            )
        ),
    ])

    answer = response.content
    if isinstance(answer, str):
        answer = answer.strip()
    elif isinstance(answer, list):
        answer = "\n".join(
            block["text"]
            for block in answer
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()
    else:
        answer = str(answer).strip()

    if not answer:
        raise RuntimeError("The research model returned an empty answer.")

    return {
        "answer": answer,
        "sources": [
            {"url": source["url"]}
            for source in sources
        ],
    }
