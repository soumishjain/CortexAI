
import asyncio

from src.agents.router import route_query


async def main():
    queries = [
        "Explain dependency injection in Spring Boot.",
        "Research the latest developments in AI agents.",
        "Answer questions using my uploaded PDF.",
    ]

    for query in queries:
        decision = await route_query(query)

        print(f"\nQuery: {query}")
        print(f"Route: {decision.route}")
        print(f"Reason: {decision.reason}")


if __name__ == "__main__":
    asyncio.run(main())
