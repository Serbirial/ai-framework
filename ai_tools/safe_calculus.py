import os
import requests

WOLFRAM_APPID = os.getenv("WOLFRAM_APPID")  # Set your Wolfram AppID in env vars
WOLFRAM_URL = "https://api.wolframalpha.com/v2/query"

def run_calculus(params: dict) -> dict:
    """
    Uses Wolfram|Alpha Full Results API to compute calculus queries.
    Params:
      - type: 'derivative' | 'integral' | 'limit'
      - expression: e.g., 'x^2 + 3*x'
      - variable: e.g., 'x'
      - at: for limit (e.g., 0 or {'value': 0, 'dir': '+'}) or bounds for integral
    """
    if not WOLFRAM_APPID:
        return {
            "error": "Wolfram Alpha API key not configured",
            "message": (
                "This tool cannot be used right now because no Wolfram Alpha AppID (API key) is configured.\n"
                "You must inform the user that this function is disabled until an API key is provided.\n"
                "Do NOT attempt to call the Wolfram Alpha API again until after you have replied to the user and there's been a chance for a developer to set the key.\n"
                "Please tell the user clearly that you cannot perform calculus without this API key."
            )
        }


    calc_type = params.get("type")
    expr = params.get("expression")
    at = params.get("at")

    if calc_type not in {"derivative", "integral", "limit"} or not expr:
        return {"error": "Params required: type in (derivative, integral, limit), expression"}

    # Build natural Wolfram query
    if calc_type == "derivative":
        query = f"derivative of {expr}"
    elif calc_type == "integral":
        if isinstance(at, list) and len(at) == 2:
            query = f"integral of {expr} from {at[0]} to {at[1]}"
        else:
            query = f"indefinite integral of {expr}"
    else:  # limit
        if isinstance(at, dict):
            suffix = f"{at['dir']} {at['value']}"
        else:
            suffix = f"{at}"
        query = f"limit of {expr} as {params.get('variable')} -> {suffix}"

    try:
        res = requests.get(WOLFRAM_URL, params={
            "appid": WOLFRAM_APPID,
            "input": query,
            "format": "plaintext",
            "output": "JSON"
        }, timeout=5)

        if res.status_code != 200:
            return {"error": f"Wolfram API error: {res.status_code}"}

        jr = res.json().get("queryresult", {})
        if not jr.get("success"):
            return {"error": "Wolfram didn't understand the query."}

        pods = jr.get("pods", [])
        if not pods:
            return {"error": "No result from Wolfram."}

        sub = next((s for pod in pods if pod.get("primary")
                    for s in pod.get("subpods", [])), None)
        if not sub:
            sub = pods[0].get("subpods", [{}])[0]

        answer = sub.get("plaintext")
        return {
            "result": answer or "No text result in pod.",
            "source": jr.get("datatypes", "")
        }
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

EXPORT = {
    "run_calculus_wolfram_alpha": {
        "help": (
            "Use Wolfram|Alpha to compute derivatives, integrals, or limits."
        ),
        "callable": run_calculus,
        "params": {
            "type": "string, required — 'derivative', 'integral', or 'limit'",
            "expression": "string, required — e.g. 'x^2 + 3*x'",
            "variable": "string, required (for limits)",
            "at": "optional — for limit: value or {value, dir}; for integral: [lower, upper]"
        }
    }
}
