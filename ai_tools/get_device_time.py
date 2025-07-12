import datetime

def get_current_time(parameters: dict = {}) -> dict:
    """
    Returns the current device time in a user-friendly format:
    'MM/DD/YYYY : HH:MM AM/PM'.

    Args:
        parameters: (optional) can be empty or ignored.

    Returns:
        A dict with the current time as "current_time".
    """
    try:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%m/%d/%Y : %I:%M %p")
        return {"current_time": formatted_time}
    except Exception as e:
        return {"error": str(e)}

EXPORT = {
    "get_local_time": {
        "help": "Use this action to get your current time in MM/DD/YYYY : HH:MM AM/PM format (this is the ASSISTANT'S local time, not the user's).",
        "callable": get_current_time,
        "params": {}
    }
}
