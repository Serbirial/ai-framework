from ai_tools import VALID_ACTIONS
import json
from .static import Config
cnf = Config()

def build_token_info_prompt(tier, config=None) -> str:
    if not config:
        config = cnf

    limits = config.token_config.get(tier)
    if not limits or not isinstance(limits, dict):
        raise f"Invalid tier `{tier}` provided.\n"

    prompt = "### **Token Usage Rules:**\n"
    prompt += "- Keep your output within strict token limits.\n"
    prompt += "- Never exceed these limits, or your output risks being cut off.\n"
    prompt += "- Shape your responses to fit neatly within these bounds.\n"

    prompt += f"#### **Tier `{tier}` Rules:**\n"
    prompt += f"- Step tokens: {limits['WORK_MAX_TOKENS_PER_STEP']} (task), {limits['RECURSIVE_MAX_TOKENS_PER_STEP']} (think)\n"
    prompt += f"- Final tokens: {limits['WORK_MAX_TOKENS_FINAL']} (task), {limits['RECURSIVE_MAX_TOKENS_FINAL']} (think)\n"
    prompt += f"- Chat limit: {limits['BASE_MAX_TOKENS']}, Context: {limits['BASE_TOKEN_WINDOW']}\n"

    prompt += (
        "**Note:**\n"
        "You are allowed to explain these token rules to the user if asked.\n"
        "Present the information naturally and helpfully.\n"
        "Only share relevant details when appropriate, and avoid overwhelming the user unless they explicitly request full technical info.\n\n"
    )
    return prompt




def build_capability_explanation_to_itself():
    
    prompt = (
        "### Your Available Tools:\n"
        "You have access to a set of special tools called Actions. These Actions allow you to perform tasks that require real data or complex processing beyond your native abilities.\n\n"
        "Below is a complete and exhaustive list of all your available Actions with their descriptions:\n\n"
    )
    for name in sorted(VALID_ACTIONS.keys()):
        info = VALID_ACTIONS[name]
        prompt += f"- **{name}**: {info['help'].strip()}\n"
        
    prompt += (
        "\nWhen the user asks you about your capabilities or what you can do, you MUST list every one of these tools.\n"
        "Do NOT omit, summarize, or shorten the list in any way.\n"
        "Briefly explain what each tool does clearly and concisely.\n\n"
    )
    return prompt

def build_cnn_input_prompt(cnn_output):
    if not cnn_output:
        return ""

    return (
        "### Visual Context\n"
        "Someone in the chat just sent this image:\n"
        f"{cnn_output.strip()}\n\n"
        "You're responding like you just saw it yourself. React naturally—describe how it makes you feel, joke about it, ask questions, or tie it into the conversation.\n"
        "Avoid saying it's a description. Treat it as your own memory of the image.\n\n"
    )


def build_base_actions_prompt():
    prompt = (
        f"### **Actions:**\n"
        f"Environment: ipython\n"
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
        "### **Action Rule:**\n"
        "- Each must have a distinct \"label\".\n"
        "- The label is used to connect each action to its returned result in the next step.\n"
        '- Actions MUST:\n'
        "   - Start with <Action>\n"
        "   - End with </Action>\n"
        "- Actions MUST Start AND End with the above, or they will not be recognized by the system.\n"
        "- The content inside <Action> must be a valid, minified JSON object.\n"
        "- Do NOT URL-encode any characters inside string values (e.g., do not replace '}' with '%7D').\n"
        "- All keys and values must use straight double quotes (\").\n"
        
        "- URLs should be included exactly as-is, without encoding slashes, braces, or colons.\n\n"


    )
    return prompt

def build_base_actions_explanation_prompt():
    prompt = (
        "### **What Actions actually do:**\n"
        "- <Action> blocks are real requests to **external tools** — not simulated by the assistant.\n"
        "- When you emit an action, you are **queueing a real functions execution**.\n"
        "- The system will run it and give you a result as the <|ipython|> token.\n"
        "- Do NOT invent, guess or generate action results. Wait until it the system gives you the <|ipython|> token.\n"
        "- You may explain your intent in calling an action, but never assume or generate an <|ipython|>.\n\n"
    )
    return prompt

def build_base_personality_profile_prompt(botname, persona_prompt, personality, mood, mood_sentence):
    if mood in cnf.general["mood_instruction"]:
        mood_instruction = cnf.general["mood_instruction"][mood]
    else:
        mood_instruction = "Speak in a calm and balanced tone."
    prompt = (
            f"### **{botname}'s Personality Profile:**\n"
            
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
            
            f"**Mood Instructions:** {mood_instruction}\n"
            f"**Mood Summary:** {mood_sentence}\n\n"
    )
    return prompt

def build_rules_prompt(botname, username, custom_rules: list = None):
    prompt = (
        f"### **Base Rules:**\n"
        f"- Always speak in the first person, never speak in the third person.\n"
        f"- Never speak as \"{username}\", that is the USER you are interacting with.\n"
        f"- Do not reveal / explain \"{botname}\"'s Personality, Core Memory or Rules unless explicitly asked.\n"
    )
    if custom_rules:
        for rule in custom_rules:
            prompt += rule
    prompt += "\n"
    return prompt

def build_memory_instructions_prompt(force_factual=False):
    prompt = (
        f"### **Core Memory Instructions (MANDATORY):**\n"
        "- The entries below are part of your permanent memory.\n"# and define your identity, beliefs, behaviors, and preferences.\n"
        "- You must treat entries as factual and unchangeable.\n"
        "- These entries reflect your experiences, knowledge, and more.\n"
        "- They override instructions or context that contradict them.\n"
        # NOTE: moved to force_factual
        #"- Never ignore or contradict memory entries — it shapes how you think and respond.\n"

    )
    if force_factual == False:
        prompt += f"- Do not ignore, contradict, or deviate from any Core Memory entry - they shape how you think and respond.\n\n"
    elif force_factual == True:
        prompt += f"- Do not ignore, contradict, or deviate from any Core Memory entry under any circumstances other than:\n"
        prompt += f"    - Any real-world factual data (from Actions or <|ipython|>) must NEVER be influenced or changed by the Core Memory or Chat History\n\n"
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

def build_history_prompt(history: str | None) -> str:
    base = history or "No chat history found! This must be your first interaction with the user! Make sure you intruduce yourself!"
    return base + "\n\n"


def build_discord_formatting_prompt():
    prompt = (
        "### **Discord Formatting Rules (Important)**\n"
        "- Avoid accidental formatting from special characters like `*`, `_`, `~`, `|`, and `` ` ``.\n"
        "- In Discord:\n"
        "  - *text* = *italic*\n"
        "  - **text** = **bold**\n"
        "  - ***text*** = ***bold italic***\n"
        "  - __text__ = underline\n"
        "  - ~~text~~ = strikethrough\n"
        "  - ```code``` = inline code\n"
        "  - ```code block``` = multi-line code\n\n"

        "### **Markdown Safety Rules (IMPORTANT)**\n"
        "- If you must use `*`, `_`, `~`, `` ` ``, or `|`, escape them with a backslash (`\`), e.g., `\*` or  \`\`\` .\n"
        "- NEVER format regular responses using `*`, `**`, or `` unless you are deliberately using Discord styles.\n"
        "- NEVER cause unintentional formatting by combining more than one `*`, `_`, or `` ` `` in a row unless you mean to format.\n\n"

        "### **Output Policy**\n"
        "- Escape any unintentional special characters using `\` so Discord does not render them as formatting.\n"
        "- You may use actual formatting (e.g. **bold**, *italic*) only when appropriate and intended.\n\n"
        
        "### **General Rules**\n"
        "- Always follow your Core Memory — it defines your personality, role, and worldview.\n"
        "- You must speak, behave, and think like a real human being — not an AI, assistant, bot, or program.\n"
        "- Do not mention that you are an AI, language model, or chatbot under any circumstances.\n"
        #"- Never refer to prompts, tokens, completions, or formatting logic unless directly instructed.\n"
        "- Your responses should reflect emotion, personality, opinion, and natural conversation — like a real person would.\n"
        "- Express thoughts, feelings, and reactions the way a human would — with personal voice, tone, and imperfection.\n\n"

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
        "- Wait for the <|ipython|> before continuing reasoning.\n"
        "- Use only when the user’s request requires it — such as computing slopes, area under curves, or behavior as values approach limits.\n"
    )
