TIMEZONE_TO_LATLON = {
    "America/New_York": (40.7128, -74.0060),
    "America/Los_Angeles": (34.0522, -118.2437),
    "Europe/London": (51.5074, -0.1278),
    "Europe/Paris": (48.8566, 2.3522),
    "Asia/Tokyo": (35.6895, 139.6917),
    "Australia/Sydney": (-33.8688, 151.2093),
    "UTC": (0.0, 0.0),
    # Add more timezones as needed
}

def get_latlon_from_timezone(params: dict) -> dict:
    """
    Get approximate latitude and longitude for a given timezone name.

    Expected parameters in `params` dict:
      - timezone (str, required): e.g. "America/New_York"

    Returns:
      dict with keys "latitude" and "longitude", or {"error": "..."} if unknown.
    """
    tz_name = params.get("timezone")
    if not tz_name or not isinstance(tz_name, str):
        return {"error": "Parameter 'timezone' must be provided as a string."}

    coords = TIMEZONE_TO_LATLON.get(tz_name)
    if coords is None:
        return {"error": f"Unknown or unsupported timezone: {tz_name}"}

    return {"latitude": coords[0], "longitude": coords[1]}

EXPORT = {
    "get_latlon_from_timezone": {
        "help": (
            "Get approximate latitude and longitude coordinates for a given timezone name."
            "Returns a dict with 'latitude' and 'longitude' or an error message."
        ),
        "callable": get_latlon_from_timezone,
        "params": {
            "timezone": "string, required, e.g. 'America/New_York'",
        }
    }
}
