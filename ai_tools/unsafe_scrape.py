import requests

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

banned_hosts = ["localhost", "127.0.0.1", "192.168"]

def simple_webpage_scraper(params: dict) -> dict:
    url = params.get("url")
    if not url:
        return {"error": "Missing required parameter: url"}
    for host in banned_hosts:
        if host in url:
            return {"error": f"Not allowed to scrape localhost or (W)LAN."}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if not res.ok:
            return {"error": f"Failed to fetch page: status code {res.status_code}"}

        html_raw = res.text

        return {"url": url, "raw_html": html_raw}


    except Exception as e:
        return {"error": f"Exception during scraping: {str(e)}"}


EXPORT = {
    "raw_url_view": {
        "help": "Downloads and summarizes any URL into a simple bare-bones summary.", # this gets intercepted and summarized in the action executor.
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
