import asyncio

from src.agents.graph import agent_graph


async def main():
    result = await agent_graph.ainvoke({
        "query": "Explain dependency injection in Spring Boot."
    })

    print("\nRoute:", result["route"])
    print("Reason:", result["reason"])
    print("\nAnswer:", result["answer"])


if __name__ == "__main__":
    asyncio.run(main())