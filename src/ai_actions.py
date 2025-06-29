from log import log
import json

def parse_action_from_response(self, response: str) -> Optional[str]:
    for line in response.strip().splitlines():
        if line.strip().lower().startswith("action:"):
            action_key = line.split(":", 1)[1].strip().lower()
            return action_key if action_key else None
    return None


def perform_action(self, action_key: str) -> dict:
    actions = {
        "search_web": self.do_web_search,
        "summarize_memory": self.summarize_memory,
        "query_memory": self.query_memory_facts,
        # Add more actions here
    }
    if action_key in actions:
        return actions[action_key]() or {"result": "OK"}
    else:
        return {"error": f"Unknown action '{action_key}'"}
