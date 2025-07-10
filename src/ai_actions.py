from log import log
import re

import json
from . import classify
from ai_tools import VALID_ACTIONS


def check_for_actions_and_run(text):
    results = []

    matches = re.findall(r"<Action>(.*?)</Action>", text, re.DOTALL)
    if not matches:
        return False  # no actions found

    for raw in matches:
        try:
            action_json = json.loads(raw)
            action_name = action_json.get("action")
            action_params = action_json.get("parameters", {})
            action_label = action_json.get("label", None)  # AI-provided label

            if not action_label:
                # If no label given, fallback to generic
                action_label = f"action_{len(results) + 1}"

            if action_name in VALID_ACTIONS:
                log(f"DEBUG: Executing action: {action_name} with {action_params}")
                result = VALID_ACTIONS[action_name]["callable"](action_params)
                results.append(f"<ActionResult{action_label}>{json.dumps(result)}</ActionResult{action_label}>")
            else:
                error_msg = {"error": f"Unknown action: {action_name}"}
                results.append(f"<ActionResult{action_label}>{json.dumps(error_msg)}</ActionResult{action_label}>")
        except Exception as e:
            error_msg = {"error": f"Failed to execute action: {str(e)}"}
            label = action_label if 'action_label' in locals() else f"action_{len(results) + 1}"
            results.append(f"<ActionResult{label}>{json.dumps(error_msg)}</ActionResult{label}>")

    if len(results)>0:
        if len(results) == 1:
            return results[0]
        else:
            return results
    return False
