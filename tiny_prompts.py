import sqlite3
from src.static import DB_PATH


def get_user_botname(userid):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT botname FROM BOT_SELECTION WHERE userid = ?", (str(userid),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return "default"

def list_personality(userid):
    """
    Returns a dictionary of personality sections and their entries
    for the user's current bot profile.
    """
    botname = get_user_botname(userid)
    if not botname:
        return {
            "goals": [],
            "traits": [],
            "likes": [],
            "dislikes": []
        }

    sections = {
        "goals": ("BOT_GOALS", "goal"),
        "traits": ("BOT_TRAITS", "trait"),
        "likes": ("BOT_LIKES", "like"),
        "dislikes": ("BOT_DISLIKES", "dislike"),
    }

    result = {
        "goals": [],
        "traits": [],
        "likes": [],
        "dislikes": []
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for key, (table, column) in sections.items():
        cursor.execute(f"SELECT {column} FROM {table} WHERE botname = ?", (botname,))
        rows = cursor.fetchall()
        result[key] = [row[0] for row in rows]

    conn.close()
    return result


def build_recursive_checkpoint_prompt(bot):
    string = (
    f"checkpoint:\n"
    f"- Are your steps leading toward the answer?\n"
    f"- Summarize progress + next action.\n"
    f"- Stay focused. Think as {bot.name}.\n"
    )
    return string

def build_recursive_final_answer_prompt_tiny(query_type: str, bot_name: str) -> str:
    if query_type == "factual_question":
        return (
            "final:\n"
            "- Write your reply now using prior thoughts/actions.\n"
            "- Restate anything needed — user can't see steps.\n"
            "- No disclaimers. No filler. First person.\n"
            "- If code was requested, include it here.\n"
        )
    else:
        return (
            "final:\n"
            "- Reply in your voice, first person.\n"
            "- Include anything user explicitly asked for.\n"
            "- No steps, titles, disclaimers, or narration.\n"
            "- Restate needed info clearly, no 'as above'.\n"
        )


def build_base_prompt_tiny(bot, username, user_input, identifier, usertone, context):
    conn = sqlite3.connect(bot.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC", (identifier,))
    rows = cursor.fetchall()
    conn.close()
    memory_text = "\n".join(f"- {row[0].strip()}" for row in rows) if rows else ""

    p = list_personality(identifier)

    system = (
        f"{bot.name}: assistant\n"
        f"traits:\n- " + "\n- ".join(p.get("traits", [])) + "\n"
        f"likes:\n- " + "\n- ".join(p.get("likes", [])) + "\n"
        f"dislikes:\n- " + "\n- ".join(p.get("dislikes", [])) + "\n"
        f"goals:\n- " + "\n- ".join(p.get("goals", [])) + "\n"
        f"mood: {bot.mood} — {bot.mood_sentence}\n"
        f"memory:\n{memory_text}\n"
        f"user-intent: {usertone['intent']}, tone: {usertone['tone']}, attitude: {usertone['attitude']}, user: {username.replace('<','').replace('>','')}"
    )

    prompt = (
        f"<|system|>\n{system.strip()}\n"
        f"{context if context else ''}"
        f"<|user|>\n{user_input.strip()}\n"
        f"<|assistant|>"
    )

    return prompt

def build_recursive_prompt_tiny(bot, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None, extra_context=""):
    p = bot.list_personality(identifier)
    traits = "\n- " + "\n- ".join(p.get("traits", [])) if p.get("traits") else ""
    goals = "\n- " + "\n- ".join(p.get("goals", [])) if p.get("goals") else ""
    likes = "\n- " + "\n- ".join(p.get("likes", [])) if p.get("likes") else ""
    dislikes = "\n- " + "\n- ".join(p.get("dislikes", [])) if p.get("dislikes") else ""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC", (identifier,))
    rows = cursor.fetchall()
    conn.close()

    memory = "\n".join(f"- {row[0].strip()}" for row in rows) if rows else ""

    base = (
        f"<|system|>\n"
        f"{bot.name}: assistant\n"
        f"traits:{traits}\nlikes:{likes}\ndislikes:{dislikes}\ngoals:{goals}\n"
        f"mood: {bot.mood} — {bot.mood_sentence}\n"
        f"user: {username}, intent: {usertone['intent']}, tone: {usertone['tone']}, attitude: {usertone['attitude']}\n"
    )

    if context:
        base += f"history:\n{context}\n"

    if memory:
        base += f"memory:\n{memory}\n"

    base += (
        f"question: {question}\n"
        f"task: respond as {bot.name}, reasoning step-by-step with traits, mood, and personality. Match user's tone. Only generate the current step.\n"
        f"rule: Do not skip steps. Do not narrate. Do not guess future turns.\n"
    )

    if extra_context:
        base += f"<ActionResult>{extra_context}</ActionResult>\n"

    guidance = {
        "factual_question": "- Be objective. Clear. Answer only the asked question. Code if needed.\n",
        "preference_query": "- Focus on your likes/dislikes. Share personal opinion clearly.\n",
        "statement": "- Reflect on how this connects to your mood, identity, goals.\n",
        "greeting": "- Respond warmly. Briefly introduce or ask follow-up.\n",
        "goodbye": "- End sincerely. Reflect briefly.\n",
        "other": "- Use mood and identity to shape thoughtful response.\n"
    }

    if query_type in guidance:
        base += "guidance:\n" + guidance[query_type]

    if include_reflection:
        base += "reflect:\n"
        base += (
            "- Emotional reaction?\n"
            "- Relevant to your goals or traits?\n"
            "- How would you typically respond?\n"
            "- Any inner conflict?\n"
        )
        mood_reflect = {
            "happy": "- Joy may make you idealistic.\n",
            "annoyed": "- Might be blunt or impatient.\n",
            "angry": "- Check for overreaction.\n",
            "sad": "- Are you being overly negative?\n",
            "anxious": "- Overthinking? Reground in values.\n"
        }.get(bot.mood, "")

        if mood_reflect:
            base += mood_reflect

    print("RECURSIVE PROMPT TINY" + base)
    return base
