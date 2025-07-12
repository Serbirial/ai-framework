import requests

def get_weather(params: dict) -> dict:
    """
    Get weather forecast data from Open-Meteo.

    Expected parameters in `params` dict:
      - latitude (float, required)
      - longitude (float, required)
      - hourly (list of strings, optional): e.g. ["temperature_2m", "precipitation", "windspeed_10m"]
      - daily (list of strings, optional): e.g. ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
      - start_date (str, optional): format 'YYYY-MM-DD'
      - end_date (str, optional): format 'YYYY-MM-DD'
      - timezone (str, optional): e.g. "America/New_York"

    Returns:
      dict with the weather data, or {"error": "..."} on failure.
    """
    base_url = "https://api.open-meteo.com/v1/forecast"

    # Validate required params
    try:
        latitude = float(params.get("latitude"))
        longitude = float(params.get("longitude"))
    except (TypeError, ValueError):
        return {"error": "latitude and longitude must be provided as floats."}

    query = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": params.get("timezone", "UTC")
    }

    # Add optional parameters if provided
    if "hourly" in params:
        if isinstance(params["hourly"], list):
            query["hourly"] = ",".join(params["hourly"])

    if "daily" in params:
        if isinstance(params["daily"], list):
            query["daily"] = ",".join(params["daily"])

    if "start_date" in params:
        query["start_date"] = params["start_date"]

    if "end_date" in params:
        query["end_date"] = params["end_date"]

    try:
        response = requests.get(base_url, params=query, timeout=7)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to get weather data: {str(e)}"}

EXPORT = {
    "get_weather": {
        "help": (
            "Get weather forecast data for given latitude and longitude from the web, Returns weather data JSON. (Uses open-meteo)"
        ),
        "callable": get_weather,
        "params": {
            "latitude": "float, required",
            "longitude": "float, required",
            "hourly": "list of strings, optional",
            "daily": "list of strings, optional",
            "start_date": "string, optional, format YYYY-MM-DD",
            "end_date": "string, optional, format YYYY-MM-DD",
            "timezone": "string, optional, e.g. 'UTC' or 'America/New_York'",
        }
    }
}
