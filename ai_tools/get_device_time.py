import datetime
from log import log

def get_current_time(parameters: dict) -> dict:
    """
    Returns the current device time in ISO 8601 format.

    Args:
        parameters: (optional) can be empty or ignored.

    Returns:
        A dict with the current ISO timestamp as "current_time".
    """
    try:
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        iso_time = now.isoformat()
        log("GET-CURRENT-TIME", iso_time)
        return {"current_time": iso_time}
    except Exception as e:
        return {"error": str(e)}

EXPORT = {
    "get_local_time": {
        "help": "Use this function/action to get your current time (in ISO 8601 format).",
        "callable": get_current_time,
        "params": {}
    }
}
