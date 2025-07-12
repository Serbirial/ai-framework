from ai_tools import VALID_ACTIONS
import json
from .static import mood_instruction

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
        "- Actions MUST Start AND End with the above, or they will not be recognized by the system."
    )
    return prompt

def build_base_actions_explanation_prompt():
    prompt = (
        "### What Actions Actually Do\n"
        "- <Action> blocks are real requests to **external tools** — not simulated by the assistant.\n"
        "- When you emit an action, you are **queueing a real functions execution**.\n"
        "- The system will run it and give you a result after the current step as <ActionResult<label>>.\n"
        "- Do NOT invent or guess action results. Use only the actual <ActionResult> values returned.\n"
        "- You may explain your intent in calling an action, but never assume or generate its outcome.\n\n"
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
        f"### **Rules:**\n"
        f"- Stop generating once your response is complete and generate this token EXACTLY: <END_GENERATION_NOW>.\n"
        f"- You must ONLY generate an \"<END_GENERATION_NOW>\" token at the END of your response or it will force stop generation."
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