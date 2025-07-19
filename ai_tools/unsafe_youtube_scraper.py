import subprocess
import json
import requests
import re

from bs4 import BeautifulSoup

USE_YTDLP = True  # Toggle between yt-dlp (accurate) and HTML (fallback)

HEADERS = {
    "User-Agent": "AI/1.0 (RespectfulAI; +http://127.0.0.1/)",
    "Referer": "http://127.0.0.1/"
}


def try_ytdlp_extract(url):
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-warnings", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        if result.returncode != 0:
            return {"error": f"yt-dlp failed: {result.stderr.decode().strip()}"}
        data = json.loads(result.stdout.decode())
        summary = {
            "title": data.get("title"),
            "uploader": data.get("uploader"),
            "channel": data.get("channel"),
            "upload_date": data.get("upload_date"),
            "duration": data.get("duration"),
            "description": data.get("description"),
            "webpage_url": data.get("webpage_url"),
        }
        return {
            "result": "YouTube data extracted via yt-dlp",
            "video_info": summary
        }
    except Exception as e:
        return {"error": f"yt-dlp exception: {str(e)}"}


def scrape_youtube_html(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=7)
        if not res.ok:
            return {"error": f"Failed to fetch page: status code {res.status_code}"}
        soup = BeautifulSoup(res.text, "html.parser")

        info = {}
        title_tag = soup.find("meta", {"name": "title"})
        if title_tag:
            info["title"] = title_tag.get("content")

        desc_tag = soup.find("meta", {"name": "description"})
        if desc_tag:
            info["description"] = desc_tag.get("content")

        channel_tag = soup.find("link", {"itemprop": "name"})
        if channel_tag:
            info["channel"] = channel_tag.get("content")

        return {
            "result": "YouTube page scraped via HTML",
            "video_info": info
        }
    except Exception as e:
        return {"error": f"HTML scraping error: {str(e)}"}


def youtube_info_scraper(url: str) -> dict:
    url = url.strip()

    # Accept all valid YouTube video URLs
    youtube_video_pattern = re.compile(
        r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/"
        r"(watch\?v=|shorts/|embed/|v/|.+\?v=)?[A-Za-z0-9_-]{11}"
    )

    if not youtube_video_pattern.search(url):
        return {"error": f"Invalid YouTube video URL: {url}"}
    
    if USE_YTDLP:
        return try_ytdlp_extract(url)
    else:
        return scrape_youtube_html(url)


EXPORT = {
    "youtube_url_info": {
        "help": "Extracts metadata from a YouTube video URL. Uses yt-dlp or falls back to HTML scraping.",
        "callable": youtube_info_scraper,
        "params": {
            "url": "string, required — Full YouTube video URL"
        }
    }
}

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=nRROWJa8sdA"
    result = youtube_info_scraper(test_url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
