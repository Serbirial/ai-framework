from log import log

from . import classify
from ai_tools import VALID_ACTIONS

import json
import re

def log_action_execution(action_name, action_params, action_label, result):
    try:
        with open("executed_actions.txt", "a", encoding="utf-8") as f:
            f.write(f"Action: {action_name}\n")
            f.write(f"Parameters: {json.dumps(action_params)}\n")
            f.write(f"Label: {action_label}\n")
            f.write(f"Result: {result}\n")
            f.write("-" * 40 + "\n")
    except Exception as e:
        log(f"ERROR: Failed to write to executed_actions.txt: {str(e)}")

def check_for_actions_and_run(text):
    results = []

    # Manual scan for <action>...</action> with any capitalization on the tags
    pos = 0
    text_len = len(text)
    while pos < text_len:
        # Find next opening tag '<'
        start_tag_pos = text.find("<", pos)
        if start_tag_pos == -1:
            break

        # Find closing '>' for the opening tag
        end_tag_pos = text.find(">", start_tag_pos)
        if end_tag_pos == -1:
            break

        # Extract the tag name inside <>
        tag_name = text[start_tag_pos + 1:end_tag_pos].strip()
        # Only consider opening tags (no / at start)
        if tag_name.lower() == "action":
            # Find the corresponding closing tag '</action>'
            pattern = re.compile(r"</action>", re.IGNORECASE)
            close_match = pattern.search(text, end_tag_pos + 1)
            if not close_match:
                # No closing tag found, break loop to avoid infinite loop
                break

            close_tag_start = close_match.start()
            close_tag_end = close_match.end()

            # Extract JSON payload between tags
            json_str = text[end_tag_pos + 1 : close_tag_start].strip()

            try:
                action_json = json.loads(json_str)
                action_name = action_json.get("action")
                action_params = action_json.get("parameters", {})
                action_label = action_json.get("label", None)

                if not action_label:
                    action_label = f"action_{len(results) + 1}"

                if action_name in VALID_ACTIONS:
                    log(f"DEBUG: Executing action: {action_name} with {action_params}")
                    result = VALID_ACTIONS[action_name]["callable"](action_params)
                    results.append(f"<ActionResult{action_label}>{json.dumps(result)}</ActionResult{action_label}>")
                    # Log the action execution to file
                    log_action_execution(action_name, action_params, action_label, result)
                else:
                    error_msg = {"error": f"Unknown action: {action_name}"}
                    results.append(f"<ActionResult{action_label}>{json.dumps(error_msg)}</ActionResult{action_label}>")
                    log_action_execution(action_name, action_params, action_label, error_msg)
            except Exception as e:
                error_msg = {"error": f"Failed to execute action: {str(e)}"}
                label = action_label if 'action_label' in locals() else f"action_{len(results) + 1}"
                results.append(f"<ActionResult{label}>{json.dumps(error_msg)}</ActionResult{label}>")
                log_action_execution(action_name if 'action_name' in locals() else "unknown", action_params if 'action_params' in locals() else {}, label, error_msg)

            # Move position past the closing tag for next search
            pos = close_tag_end
        else:
            # Not an <action> tag, skip this tag and continue
            pos = end_tag_pos + 1

    if len(results) > 0:
        if len(results) == 1:
            return results[0]
        else:
            return results
    return False
