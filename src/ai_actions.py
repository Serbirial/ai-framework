from log import log
import json
from . import classify

def parse_action_from_response(self, response: str) -> Optional[str]:
    for line in response.strip().splitlines():
        if line.strip().lower().startswith("action:"):
            action_key = line.split(":", 1)[1].strip().lower()
            return action_key if action_key else None
    return None

def stubbed():
    return "THIS HAS BEEN STUBBED UNTIL THE DEV IMPLEMENTS"

def perform_action(self, action_key: str, data) -> dict:
    actions = {
        "search_web": stubbed,
        "get_memory": classify.interpret_to_remember,
    }
    if action_key in actions:
        return actions[action_key]() or {"result": "OK"}
    else:
        return {"error": f"Unknown action '{action_key}'"}
