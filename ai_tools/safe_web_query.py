import requests
from bs4 import BeautifulSoup
import urllib.parse
import time

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

# Define domain weights (higher = more important)
DOMAIN_WEIGHTS = {
    "wikipedia.org": 100,
    "britannica.com": 50,
    "nationalgeographic.com": 5,
    "history.com": 0,
    "britannica.co.uk": 0,
    # Add more domains as needed
}

def fetch_wikipedia_summary(url: str) -> dict:
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if not res.ok:
            return {"error": f"Wikipedia page fetch failed with status {res.status_code}"}
        soup = BeautifulSoup(res.text, "html.parser")
        content_div = soup.find("div", id="mw-content-text")
        if not content_div:
            return {"error": "Wikipedia content section not found."}
        first_p = content_div.find("p", recursive=True)
        if first_p and first_p.text.strip():
            summary = first_p.text.strip()
            return {"result": summary}
        return {"error": "No summary paragraph found on Wikipedia page."}
    except Exception as e:
        return {"error": f"Failed to fetch Wikipedia summary: {str(e)}"}

def duckduckgo_query(query: str) -> dict:
    try:
        # 1. Instant Answer API
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
        time.sleep(0.3)
        # 2. HTML Search - get top 10 links
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
        results = soup.select("a.result__a")[:10]  # first 10 links

        # Collect domain + link tuples with weights
        scored_links = []
        for link in results:
            href = link.get("href")
            if not href:
                continue
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            real_url = qs.get("uddg", [None])[0] or href
            domain = urllib.parse.urlparse(real_url).netloc.lower()
            # Remove www.
            if domain.startswith("www."):
                domain = domain[4:]
            weight = DOMAIN_WEIGHTS.get(domain, 10)  # default weight low if unknown domain
            scored_links.append((weight, real_url, domain))

        if not scored_links:
            return {"error": "No valid links found in search results."}

        # Pick highest weight
        scored_links.sort(key=lambda x: x[0], reverse=True)
        best_weight, best_url, best_domain = scored_links[0]

        # Wikipedia summary if Wikipedia domain
        if "wikipedia.org" in best_domain:
            return fetch_wikipedia_summary(best_url)
        else:
            return {"result": None, "top_link": best_url}

    except requests.exceptions.Timeout:
        return {"error": "Request to DuckDuckGo timed out."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

EXPORT = {
    "duckduckgo_query": {
        "help": (
            "Perform a DuckDuckGo Instant Answer query and return the top answer or summary."
        ),
        "callable": duckduckgo_query,
        "params": {
            "query": "string, required"
        }
    }
}

if __name__ == "__main__":
    import json
    query = "height of Mount Everest"
    print(json.dumps(duckduckgo_query(query), indent=2))
