import requests
from datetime import datetime

def get_current_time(timezone: str = "America/New_York") -> dict:
    """
    Get current time and date for a specific timezone using worldtimeapi.org.
    Returns a dict with 'time', 'date', and 'timezone'.
    """
    try:
        res = requests.get(f"https://worldtimeapi.org/api/timezone/{timezone}")
        res.raise_for_status()
        data = res.json()
        dt = datetime.fromisoformat(data["datetime"])
        return {
            "time": dt.strftime("%-I:%M %p"),
            "date": dt.strftime("%A, %B %d, %Y"),
            "timezone": timezone
        }
    except Exception as e:
        return {"error": str(e)}

EXPORT = {
    "get_current_time": {
        "help": (
            "Get current time and date for a given IANA timezone.\n"
            "Returns a dict with 'time', 'date', and 'timezone'. (Uses worldtimeapi.org, this is NOT FOR LOCAL TIMES)"
        ),
        "callable": get_current_time,
        "params": {
            "timezone": "string, optional, default is 'America/New_York'"
        }
    }
}
