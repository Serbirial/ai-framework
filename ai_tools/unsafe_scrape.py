import requests
from bs4 import BeautifulSoup
import html
import re

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

def clean_text(text):
    return html.unescape(re.sub(r'\s+', ' ', text)).strip()

def format_for_ai(text):
    paragraphs = text.split('\n\n')
    return "\n".join(f"- {p.strip()}" for p in paragraphs if p.strip())

def extract_footnotes(soup, max_footnotes=5):
    refs = {}
    count = 0
    for sup in soup.find_all("sup", recursive=True):
        if count >= max_footnotes:
            break
        label = sup.get_text(strip=True)
        if not re.match(r'^\[?[a-z0-9]{1,3}\]?$' , label, re.I):  # Skip weird entries
            continue
        # Try href in <a> tag inside the <sup>
        a = sup.find("a", href=True)
        if a and a['href'].startswith('#cite_note'):
            ref_id = a['href'].lstrip('#')
            target = soup.find(id=ref_id)
            if target:
                hrefs = target.find_all('a', href=True)
                for link in hrefs:
                    if link['href'].startswith("http"):
                        refs[f"[{label}]"] = link['href']
                        count += 1
                        break
    return refs

def remove_footnote_markers(text):
    return re.sub(r'\[\s?[a-zA-Z0-9]{1,3}\s?\]', '', text)

def remove_quirks(text):
    text = re.sub(r'/[^/]+?/', '', text)  # Remove phonetic slashes
    text = re.sub(r'[\u2070-\u209F\u00B2\u00B3\u00B9]+', '', text)  # Remove super/subscripts
    text = re.sub(r'\+\d+⁄\d+\s+\w+', '', text)  # Remove +1⁄2 in
    return text

def simple_webpage_scraper(url: str, max_paragraphs: int = 3) -> dict:
    try:
        res = requests.get(url, headers=headers, timeout=7)
        if not res.ok:
            return {"error": f"Failed to fetch page: status code {res.status_code}"}
        
        soup = BeautifulSoup(res.text, "html.parser")

        # Extract footnotes
        footnotes = extract_footnotes(soup)

        # Try to locate main content
        possible_containers = [
            {"id": "mw-content-text"},  # Wikipedia
            {"id": "content"},
            {"id": "main"},
            {"class": "post-content"},
            {"class": "article-content"},
            {"class": "entry-content"},
            {"class": "content"},
            {"class": "post"},
        ]

        main_text = None
        for selector in possible_containers:
            if "id" in selector:
                main_text = soup.find("div", id=selector["id"])
            elif "class" in selector:
                main_text = soup.find("div", class_=selector["class"])
            if main_text:
                break
        
        if not main_text:
            main_text = soup.body

        paragraphs = main_text.find_all("p", recursive=True)
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
            return {"error": "No readable paragraph content found."}

        summary_raw = "\n\n".join(summary_paragraphs)

        return {
            "result": "Data has been found and formatted",
            "formatted": format_for_ai(summary_raw),
            "footnotes": footnotes
        }

    except Exception as e:
        return {"error": f"Exception during scraping: {str(e)}"}

EXPORT = {
    "simple_webpage_scraper": {
        "help": "Scrapes a raw web URL and returns a formatted version, and top footnotes.",
        "callable": simple_webpage_scraper,
        "params": {
            "url": "string, required",
            "max_paragraphs": "int, optional, default 3"
        }
    }
}

if __name__ == "__main__":
    import json
    test_url = "https://en.wikipedia.org/wiki/Mount_Everest"
    result = simple_webpage_scraper(test_url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
