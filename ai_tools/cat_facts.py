import requests

def get_cat_fact(max_length: int = 140) -> dict:
    """
    Fetches a cat fact from catfact.ninja with an optional maximum length.

    Args:
        max_length (int): Maximum length of the cat fact (default 140).

    Returns:
        dict: {"fact": str} on success, or {"error": str} on failure.
    """
    url = "https://catfact.ninja/fact"
    params = {"max_length": max_length}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        fact = data.get("fact", "")
        return {"fact": fact}
    except Exception as e:
        return {"error": f"Failed to fetch cat fact: {str(e)}"}

EXPORT = {
    "get_cat_fact": {
        "help": (
            "Fetch a random cat fact with the given max_length from the website \"catfact.ninja\"."
        ),
        "callable": get_cat_fact,
        "params": {
            "max_length": "int, optional, max length of the cat fact (default 140)"
        }
    }
}
