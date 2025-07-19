import requests
from bs4 import BeautifulSoup
import html
import time
import re

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_ARTICLE_URL = "https://en.wikipedia.org/wiki/"

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

def clean_text(text):
    return html.unescape(re.sub(r'\s+', ' ', text)).strip()

def format_for_ai(text):
    paragraphs = text.split('\n\n')
    return "\n".join(f"- {p.strip()}" for p in paragraphs if p.strip())

def remove_footnote_markers(text):
    return re.sub(r'\[\s?[a-zA-Z0-9]{1,3}\s?\]', '', text)

def remove_quirks(text):
    text = re.sub(r'/[^/]+?/', '', text)  # Remove phonetic slashes
    text = re.sub(r'[\u2070-\u209F\u00B2\u00B3\u00B9]+', '', text)  # Remove super/subscripts
    text = re.sub(r'\+\d+⁄\d+\s+\w+', '', text)  # Remove +1⁄2 in
    return text

def wiki_scraper(query: str, max_paragraphs: int = 4) -> dict:
    try:
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srwhat": "nearmatch",
            "srprop": "snippet",
            "format": "json"
        }

        # Ensure a minimum of 3 paragraphs
        if max_paragraphs < 3:
            max_paragraphs = 3
        res = requests.get(WIKI_API, params=search_params, headers=headers, timeout=7)
        res.raise_for_status()
        search_data = res.json()
        if not search_data["query"]["search"]:
            return {"error": "No Wikipedia page found for that query."}

        title = search_data["query"]["search"][0]["title"]
        url_title = title.replace(" ", "_")
        article_url = f"{WIKI_ARTICLE_URL}{url_title}"
        time.sleep(0.3) # be respectful
        page_res = requests.get(article_url, headers=headers, timeout=7)
        page_res.raise_for_status()
        soup = BeautifulSoup(page_res.text, "html.parser")

        content_div = soup.find("div", id="mw-content-text")
        if not content_div:
            return {"error": "Wikipedia content section not found."}

        paragraphs = content_div.find_all("p", recursive=True)
        summary_paragraphs = []
        for p in paragraphs:
            text = clean_text(p.get_text())
            text = remove_footnote_markers(text)
            text = remove_quirks(text)
            if text:
                summary_paragraphs.append(text)
            if len(summary_paragraphs) >= max_paragraphs:
                break

        if not summary_paragraphs:
            return {"error": "No readable paragraph content found in Wikipedia article."}

        summary_raw = "\n\n".join(summary_paragraphs)
        return {
            "result": f"Wikipedia summary for '{title}' retrieved.",
            "formatted": format_for_ai(summary_raw),
            "url": article_url
        }

    except Exception as e:
        return {"error": f"Exception during Wikipedia scraping: {str(e)}"}

EXPORT = {
    "wikipedia_search": {
        "help": "Look up a single Wikipedia article **by page title** and return a short summary.",
        "callable": wiki_scraper,
        "params": {
            "query": "string, required — The exact or best‑guess Wikipedia *page title*‑only",
            "max_paragraphs": "int, optional, default 4 (minimum 3)"
        }
    }
}

if __name__ == "__main__":
    import json
    result = wiki_scraper("Hyperpop music history", max_paragraphs=3)
    print(json.dumps(result, indent=2, ensure_ascii=False))
