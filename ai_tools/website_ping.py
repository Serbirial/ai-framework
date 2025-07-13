import requests

def ping_url_checker(url: str, timeout: int = 5) -> dict:
    """
    Verifies if a URL is reachable and responsive (HTTP 200–399).
    Does not download full content, just checks basic reachability.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        status = response.status_code

        if 200 <= status < 400:
            return {
                "result": "URL is active",
                "status_code": status,
                "final_url": response.url
            }
        else:
            return {
                "error": "URL responded with an error status",
                "status_code": status,
                "final_url": response.url
            }
    except requests.exceptions.Timeout:
        return {"error": "Connection timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

EXPORT = {
    "ping_url_checker": {
        "help": "Checks if a web URL is reachable or active. Uses a HEAD request with redirects allowed.",
        "callable": ping_url_checker,
        "params": {
            "url": "string, required — Full URL to ping",
            "timeout": "int, optional — Timeout in seconds (default: 5)"
        }
    }
}

if __name__ == "__main__":
    import json
    test_url = "https://youtube.com"
    result = ping_url_checker(test_url)
    print(json.dumps(result, indent=2))
