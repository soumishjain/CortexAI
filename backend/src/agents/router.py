
from typing import Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings

answer_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
)

class RouteDecision(BaseModel):
    """Structured routing decision for CortexAI."""

    route: Literal[
        "direct",
        "web_research",
        "document_rag",
    ] = Field(
        description=(
            "Choose direct for general questions, "
            "web_research for current or externally researched "
            "information, and document_rag for questions about "
            "user-provided documents."
        )
    )

    reason: str = Field(
        description="Brief explanation for selecting this route."
    )


router_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
).with_structured_output(RouteDecision)


async def route_query(query: str) -> RouteDecision:
    """Classify a user query into one of CortexAI's routes."""

    prompt = f"""
You are the routing component of CortexAI.

Select exactly one route:
- direct: general questions answerable without external research
- web_research: requests requiring current information or web research
- document_rag: requests that require searching user-provided documents

Choose based on the user's actual request.
Do not claim that a document exists or that web research was performed.

User query:
{query}
"""

    return await router_llm.ainvoke(prompt)
