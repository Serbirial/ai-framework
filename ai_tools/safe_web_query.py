import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

def duckduckgo_query(query: str) -> dict:
    """
    Perform a DuckDuckGo Instant Answer query.
    Returns top summary if available, else first organic link.
    If blocked or rate-limited, returns an explanation.
    """
    try:
        # 1. Try Instant Answer API
        api_res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1},
            headers=headers,
            timeout=5
        )

        if api_res.status_code == 429:
            return {"error": "DuckDuckGo is rate-limiting the request. Try again later."}

        if not api_res.ok:
            return {"error": f"DuckDuckGo API request failed with status code {api_res.status_code}."}

        data = api_res.json()

        result_text = data.get("Answer") or data.get("AbstractText")
        if not result_text:
            related = data.get("RelatedTopics", [])
            if related and isinstance(related[0], dict):
                result_text = related[0].get("Text")

        if result_text:
            return {"result": result_text.strip()}

        # 2. Fallback: Try HTML search
        html_res = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=5
        )

        if html_res.status_code == 429:
            return {"error": "DuckDuckGo HTML search is rate-limiting you. Slow down requests."}

        if "unusual traffic" in html_res.text.lower() or "block" in html_res.url:
            return {"error": "DuckDuckGo has temporarily blocked or limited your access."}

        soup = BeautifulSoup(html_res.text, "html.parser")
        link = soup.select_one("a.result__a")
        if link and link.get("href"):
            return {
                "result": None,
                "top_link": link["href"]
            }

        return {"error": "No direct answer or valid search results found."}

    except requests.exceptions.Timeout:
        return {"error": "Request to DuckDuckGo timed out."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

EXPORT = {
    "duckduckgo_query": {
        "help": (
            "Perform a DuckDuckGo Instant Answer query and return the top answer or summary. "
            "If not found, returns the first real link. Handles rate limits and blocks gracefully."
        ),
        "callable": duckduckgo_query,
        "params": {
            "query": "string, required"
        }
    }
}
