import os
from dataclasses import dataclass

import requests

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_ENDPOINT = "https://google.serper.dev/search"


@dataclass
class Tool:
    name: str
    description: str
    func: callable


def google_search(query: str, num_results: int = 10, **kwargs) -> str:
    """Find results for a query using Serper's API for Google

    Args:
        query (str): query to search for in Google
        num_results (int, optional): number of results. Defaults to 10.

    Returns:
        str: list of sources
    """
    url = SERPER_ENDPOINT
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results, **kwargs}

    response = requests.post(url, headers=headers, json=payload)

    # Check if it was successful
    response.raise_for_status()

    results = response.json()

    # Extract and clean up the results
    cleaned_results = []
    if "organic" in results:
        for result in results["organic"][:num_results]:
            cleaned_results.append(
                {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                }
            )

    result = "\n".join(
        [
            f"- [{i}] [{r['title']}]({r['link']}): {r['snippet']}"
            for i, r in enumerate(cleaned_results)
        ]
    )

    return result


tools = [
    Tool(
        name="Google search",
        description="useful to search in Google for information about a query",
        func=google_search,
    )
]
tool_names = [tool.name for tool in tools]