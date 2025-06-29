from log import log
import json

def parse_action_from_response(self, text):
    """Parse <Action> JSON block from AI response."""
    match = re.search(r"<Action>(.*?)</Action>", text, re.DOTALL)
    if not match:
        return None
    try:
        action_json = match.group(1).strip()
        return json.loads(action_json)
    except Exception as e:
        log(f"ERROR parsing action JSON: {e}")
        return None


def perform_action(self, action_dict):
    """Perform the action requested by AI and return results as text or JSON."""
    action = action_dict.get("action")
    params = action_dict.get("parameters", {})

    log(f"Performing action: {action} with params {params}")

    if action == "get_time":
        import datetime
        tz = params.get("timezone", "UTC")
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return {"time": now, "timezone": tz}

    elif action == "run_code":
        # Example: safely run some limited code or command
        code = params.get("code", "")
        # WARNING: Only allow safe code or sandbox!
        # Here just echo back for example
        return {"output": f"Executed code: {code}"}

    # Add more supported actions here...

    return {"error": f"Unknown action '{action}'"}
