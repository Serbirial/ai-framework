import requests
from src import classify
from bs4 import BeautifulSoup
import html
import re

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

def simple_webpage_scraper(params: dict, model=None) -> dict:
    url = params.get("url")
    if not url:
        return {"error": "Missing required parameter: url"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if not res.ok:
            return {"error": f"Failed to fetch page: status code {res.status_code}"}

        html_raw = res.text

        summary = classify.summarize_raw_scraped_data(model, html_raw, 2048)
        return {"url": url, "summary": summary}

        return {
            "result": "Raw HTML successfully fetched",
            "raw_html": html_raw
        }

    except Exception as e:
        return {"error": f"Exception during scraping: {str(e)}"}


EXPORT = {
    "simple_url_scraper": {
        "help": "Fetches the full raw HTML of a web page for later summarization/classification.",
        "callable": simple_webpage_scraper,
        "params": {
            "url": "string, required"
        }
    }
}

if __name__ == "__main__":
    import json
    test_model = None  
    test_url = {
        "url": "https://music.youtube.com/playlist?list=PLzpRt2zQpnBO0yzFMjmilb7I7DGckf2jC"
    }
    result = simple_webpage_scraper(test_url, model=test_model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
