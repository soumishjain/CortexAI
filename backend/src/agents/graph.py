from typing import Literal
from uuid import UUID

from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agents.router import route_query, answer_llm
from src.agents.web_research import research_web
from src.agents.rag.answer import answer_from_document


class AgentState(TypedDict):
    query: str
    route: Literal["direct", "web_research", "document_rag"]
    reason: str
    answer: str
    context: list
    sources: list[dict[str, str]]
    document_id: str | None


async def router_node(state: AgentState) -> dict:
    """Classify the query and select the appropriate agent."""
    decision = await route_query(state["query"])

    return {
        "route": decision.route,
        "reason": decision.reason,
    }


async def direct_answer_node(state: AgentState) -> dict:
    """Answer general questions using conversation context."""
    messages = [
        SystemMessage(
            content=(
                "You are CortexAI, an AI research and knowledge assistant. "
                "Use the supplied conversation context to answer follow-up "
                "questions. Do not claim to have searched the web or read "
                "documents unless that actually happened."
            )
        )
    ]

    messages.extend(state.get("context", []))

    # Include the current query so the model can answer it.
    messages.append(HumanMessage(content=state["query"]))

    response = await answer_llm.ainvoke(messages)
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
        raise RuntimeError("The AI returned an empty response.")

    return {
        "answer": answer,
        "sources": [],
    }


async def web_research_node(state: AgentState) -> dict:
    """Research the web and generate a sourced answer."""
    result = await research_web(state["query"])

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }


async def document_rag_node(state: AgentState) -> dict:
    """Answer questions using retrieved PDF content."""
    raw_document_id = state.get("document_id")

    try:
        document_id = UUID(raw_document_id) if raw_document_id else None
    except ValueError as exc:
        raise ValueError("Invalid document_id format.") from exc

    result = await answer_from_document(
        query=state["query"],
        document_id=document_id,
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }


def select_route(
    state: AgentState,
) -> Literal["direct", "web_research", "document_rag"]:
    """Select the next graph node based on the router decision."""
    return state["route"]


builder = StateGraph(AgentState)

# Register nodes.
builder.add_node("router", router_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("web_research", web_research_node)
builder.add_node("document_rag", document_rag_node)

# Start with query classification.
builder.add_edge(START, "router")

# Route to the selected agent.
builder.add_conditional_edges(
    "router",
    select_route,
    {
        "direct": "direct_answer",
        "web_research": "web_research",
        "document_rag": "document_rag",
    },
)

# All agents terminate the graph.
builder.add_edge("direct_answer", END)
builder.add_edge("web_research", END)
builder.add_edge("document_rag", END)

# Compile the graph.
agent_graph = builder.compile()