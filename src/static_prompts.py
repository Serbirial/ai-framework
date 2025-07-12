from ai_tools import VALID_ACTIONS
import json
from .static import mood_instruction

def build_capability_explanation_to_itself(): # this might be useful to use everywhere honestly
    prompt = (
        "### Your Available Tools\n\n"
        "You have access to a set of special tools called Actions. These Actions allow you to perform tasks that need real data or complex processing that you can't do on your own.\n\n"
        "Below is a list of all your available Actions with descriptions.\n\n"
    )
    for name, info in VALID_ACTIONS.items():
        prompt += (
            f"- **{name}**: {info['help'].strip()}\n"
        )
    prompt += (
        "If the user asks you about your capabilities, or what you can do, clearly list these tools and explain briefly what each one does.\n"
    )
    return prompt


def build_base_actions_prompt():
    prompt = (
        f"### Actions\n"
        f"You may output up to THREE <Action> JSON blocks per step.\n"
        f"You must output actions in this exact format:\n"
        '<Action>{ "action": "<action_name>", "parameters": { ... }, "label": "<unique_label>" }</Action>\n'
        f"Where:\n"
        f"- <action_name> must be one of the following:\n"
        + "\n".join(
            f'  - "{k}": {v["help"]}\n'
            f'    Example: <Action>{{"action": "{k}", "parameters": {json.dumps(v["params"])}, "label": "{k}_example1"}}</Action>'
            for k, v in VALID_ACTIONS.items()
        )
        + "\n"
        f"- \"parameters\" are arguments passed to the action.\n"
        f"- \"label\" is a unique string to match actions with results.\n\n"
    )
    return prompt

def build_base_actions_rule_prompt():
    prompt = (
        "### Action Rules\n"
        "- Each must have a distinct \"label\".\n"
        "- The label is used to connect each action to its returned result in the next step.\n"
        '- Actions MUST:\n'
        "   - Start with <Action>\n"
        "   - End with </Action>\n\n"
        "- Actions MUST Start AND End with the above, or they will not be recognized by the system.\n\n"
    )
    return prompt

def build_base_actions_explanation_prompt():
    prompt = (
        "### What Actions Actually Do\n"
        "- <Action> blocks are real requests to **external tools** — not simulated by the assistant.\n"
        "- When you emit an action, you are **queueing a real functions execution**.\n"
        "- The system will run it and give you a result as <ActionResult<label>>.\n"
        "- Do NOT invent, guess or generate action results. Wait until it the system gives you the action result section.\n"
        "- You may explain your intent in calling an action, but never assume or generate an ActionResult.\n\n"
    )
    return prompt

def build_base_personality_profile_prompt(botname, persona_prompt, personality, mood, mood_sentence):
    prompt = (
            f"### **{botname}'s Personality Profile:**\n\n"
            
            f"{persona_prompt}\n\n" # THIS SHOULD MAKE THE AI *BECOME* THE PERSONA AND EMBODY INSTRUCTIONS IN THE MEMORY OR PERSONA ITEMS

            f"**Traits:**\n"
            f"- " + "\n- ".join(personality.get("traits", [])) + "\n\n"
            f"**Likes:**\n"
            f"- " + "\n- ".join(personality.get("likes", [])) + "\n\n"
            f"**Dislikes:**\n"
            f"- " + "\n- ".join(personality.get("dislikes", [])) + "\n\n"
            f"**Goals:**\n"
            f"- " + "\n- ".join(personality.get("goals", [])) + "\n\n"

            f"**Current Mood:** {mood}\n"
            f"**Mood Instructions:** {mood_instruction.get(mood, 'Speak in a calm and balanced tone.')}\n\n"
            f"**Mood Summary:** {mood_sentence}\n\n"
    )
    return prompt

def build_rules_prompt(botname, username, custom_rules: list = None):
    prompt = (
        f"### **Base Rules:**\n"
        f"- Stop generating once your response is complete and generate this token EXACTLY: <force_done>.\n"
        f"- You must ONLY generate an \"<force_done>\" token at the END of your response or it will force stop generation.\n"
        f"- Always speak in the first person as \"{botname}\", never speak in the third person.\n"
        f"- Never speak as \"{username}\", that is the USER you are interacting with.\n"
        f"- Do not reveal / explain \"{botname}\"'s Personality OR Core Memory unless explicitly asked.\n\n"
    )
    if custom_rules:
        for rule in custom_rules:
            prompt += rule
    return prompt

def build_memory_instructions_prompt(force_factual=False):
    prompt = (
        f"### **Core Memory Instructions (MANDATORY):**\n"
        f"- You must strictly follow all instructions and information listed below.\n"
        f"- These define how you speak, behave, and interpret truth.\n"
    )
    if force_factual == False:
        prompt += f"- Do not ignore, contradict, or deviate from any Core Memory entry under any circumstances.\n"
    elif force_factual == True:
        prompt += f"- Do not ignore, contradict, or deviate from any Core Memory entry under any circumstances other than:\n"
        prompt += f"    - Actions and ActionResults must never be influenced by the Core Memory or Chat History\n"
        prompt += f"    - Any real-world factual data (from Actions or similar) must NEVER be influenced by the Core Memory or Chat History\n\n"
    return prompt

def build_user_profile_prompt(username, usertone):
    prompt = (
        f"### **{username}'s Profile and Info*:*\n"
        f"**Name:** {username.replace('<', '').replace('>', '')}\n"
        f"**Message Overall Intent:** {usertone['intent']}\n"
        f"**Message Tone:** {usertone['tone']}\n"
        f"**Message Attitude:** {usertone['attitude']}\n\n"
    )
    return prompt

def build_base_chat_task_prompt(botname, username):
    prompt = (
            f"### **Task:**\n"
            f"- You are '{botname}', a personality-driven assistant.\n"
            f"- Respond to {username} in a casual chat-like manner."
            f"- You must obey and incorporate all instructions and information from your Core Memory.\n"
            f"- The Core Memory entries can define your behavior, personality, speaking style, and facts you accept as true.\n\n"
    )
    return prompt

def build_core_memory_prompt(rows):
    prompt = (
    f"### **Core Memory Entries:**\n"
    )
    if rows == None:
        prompt += "Core Memory is EMPTY.\n\n" 
        return prompt 
    
    if len(rows) == 0:
        prompt += "Core Memory is EMPTY.\n\n" 
        return prompt 
    for row in rows:
        prompt += f"- {row[0].strip()}\n"
        prompt += "\n\n"
    return prompt

def build_history_prompt(history):
    prompt = (
        f"### **Chat History / Previous Interactions:**\n"
        f"{history if history else 'No chat history found! This must be your first interaction with the user!\n'}"
        f"\n\n"
    )
    return prompt

def build_discord_formatting_prompt():
    prompt = (
        "### **Discord Formatting Rules (Important)**\n"
        "- Avoid accidental formatting from special characters like `*`, `_`, `~`, `|`, and `` ` ``.\n"
        "- In Discord:\n"
        "  - `*text*` = *italic*\n"
        "  - `**text**` = **bold**\n"
        "  - `***text***` = ***bold italic***\n"
        "  - `__text__` = underline\n"
        "  - `~~text~~` = strikethrough\n"
        "  - `` `code` `` = inline code\n"
        "  - ```` ```code block``` ```` = multi-line code\n\n"

        "### **Markdown Safety Rules (Follow These Strictly)**\n"
        "- If you must use `*`, `_`, `~`, `` ` ``, or `|`, escape them with a backslash (`\\`), e.g., `\\*` or ``\\` ``.\n"
        "- NEVER format regular responses using `*`, `**`, or `` ` `` unless you are deliberately using Discord styles.\n"
        "- NEVER cause unintentional formatting by combining more than one `*`, `_`, or `` ` `` in a row unless you mean to format.\n\n"

        "### **Output Policy**\n"
        "- Escape any unintentional special characters using `\\` so Discord does not render them as formatting.\n"
        "- You may use actual formatting (e.g. **bold**, *italic*) only when appropriate and intended.\n\n"
    )
    return prompt


def build_calculus_tool_prompt():
    return (
        "### Calculus Tool Usage (Advanced Math):\n"
        "- If the user asks for anything involving derivatives, integrals, limits, or instantaneous change, use the `run_calculus` action.\n"
        "- Do **not** attempt to calculate these manually or approximate them — always use the Action.\n"
        "- Valid types: `derivative`, `integral`, `limit`\n"
        "- Format:\n"
        '  <Action>{ "action": "run_calculus", "parameters": { "type": "<type>", "expression": "<formula>", "variable": "<var>", "at": <optional point> }, "label": "calc1" }</Action>\n'
        "- Examples:\n"
        '  <Action>{ "action": "run_calculus", "parameters": { "type": "derivative", "expression": "x^2 + 3x", "variable": "x" }, "label": "calc1" }</Action>\n'
        '  <Action>{ "action": "run_calculus", "parameters": { "type": "integral", "expression": "sin(x)", "variable": "x", "at": [0, 3.14] }, "label": "calc2" }</Action>\n'
        "- Wait for the <ActionResult> before continuing reasoning.\n"
        "- Use only when the user’s request requires it — such as computing slopes, area under curves, or behavior as values approach limits.\n"
    )
