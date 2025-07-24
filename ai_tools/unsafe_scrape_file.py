import requests
from urllib.parse import urlparse

headers = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}

banned_hosts = ["localhost", "127.0.0.1", "192.168"]

# Allowed file extensions (non-image)
allowed_extensions = {
    ".pdf": "PDF document",
    ".txt": "Plain text file",
    ".csv": "Comma-separated values file",
    ".json": "JSON data file",
    ".xml": "XML data file",
    ".md": "Markdown document",
    ".log": "Log file",
    ".css": "Cascading Style Sheets",
    ".tsv": "Tab-separated values file",
    ".yaml": "YAML data file",
    ".yml": "YAML data file",
    ".rtf": "Rich Text Format document",
    ".doc": "Microsoft Word document",
    ".docx": "Microsoft Word (OpenXML) document",
    ".xls": "Microsoft Excel spreadsheet",
    ".xlsx": "Microsoft Excel (OpenXML) spreadsheet",

    ".js": "JavaScript code file",
    ".py": "Python code file",
    ".go": "Go source code file",
    ".ts": "TypeScript code file"
}


# extensions to exclude
exclude_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico", ".htm", ".html"}

def simple_file_scraper(params: dict) -> dict:
    url = params.get("url")
    if not url:
        return {"error": "Missing required parameter: url"}

    # Block banned hosts
    for host in banned_hosts:
        if host in url:
            return {"error": f"Not allowed to scrape localhost or (W)LAN."}

    # Parse the URL path to check extension
    path = urlparse(url).path.lower()
    ext = ""
    if "." in path:
        ext = path[path.rfind("."):]
    else:
        # No extension found, reject
        return {"error": "URL does not end in a file extension"}

    # Reject if extension is image or not in allowed
    if ext in exclude_extensions:
        return {"error": "Image/Video and HTML files are not allowed."}

    if ext not in allowed_extensions.keys():
        return {"error": f"File extension '{ext}' is not allowed."}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if not res.ok:
            return {"error": f"Failed to fetch page: status code {res.status_code}"}

        html_raw = res.text

        return {"url": url, "raw_data": html_raw, "type": allowed_extensions[ext]}

    except Exception as e:
        return {"error": f"Exception during scraping: {str(e)}"}


EXPORT = {
    "raw_file_scraper": {
        "help": "Scrapes any URL with a supported file ending and get either the raw data (if under token limit) or a summary of the file contents.",  # this gets intercepted and summarized in the action executor.
        "callable": simple_file_scraper,
        "params": {
            "url": "string, required"
        }
    }
}

if __name__ == "__main__":
    import json
    test_url = {
        "url": "https://example.com/sample.pdf"
    }
    result = simple_file_scraper(test_url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
