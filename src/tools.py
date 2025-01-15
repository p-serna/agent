import os
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

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


def fetch_website(url: str) -> str:
    """Fetch and extract main content from a website.

    Args:
        url (str): link url to request

    Returns:
        str: content of the website
    """
    try:
        # Add user agent to avoid some blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        # Truncate if too long (to avoid token limits)
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text

    except requests.RequestException as e:
        return (f"Failed to fetch website: {str(e)}",)


tools = [
    Tool(
        name="Google search",
        description="useful to search in Google for information about a query",
        func=google_search,
    ),
    Tool(
        name="Fetch website",
        description="useful to fetch website content of an url link",
        func=fetch_website,
    ),
]
tool_names = [tool.name for tool in tools]
