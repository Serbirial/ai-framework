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

    lines = [
        "<|system|>",
        f"name: {bot.name}",
        f"traits: {', '.join(p.get('traits', []))}",
        f"likes: {', '.join(p.get('likes', []))}",
        f"dislikes: {', '.join(p.get('dislikes', []))}",
        f"goals: {', '.join(p.get('goals', []))}",
        f"mood: {bot.mood} — {bot.mood_sentence}",
        f"memory:",
    ]
    if memory_text:
        lines.extend(line for line in memory_text.split("\n") if line.strip())

    lines.append(f"user-intent: {usertone['intent']}, tone: {usertone['tone']}, attitude: {usertone['attitude']}, user: {username}")

    if context:
        lines.append(f"\n{context.strip()}")

    lines.append(f"\n<|user|> {username}: {user_input.strip()}")
    lines.append(f"<|user1|> {bot.name}:")

    return "\n".join(lines)

def build_recursive_prompt_tiny(bot, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None, extra_context=""):
    p = list_personality(identifier)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC", (identifier,))
    rows = cursor.fetchall()
    conn.close()

    memory = "\n".join(f"- {row[0].strip()}" for row in rows) if rows else ""

    lines = [
        "<|system|>",
        f"name: {bot.name}",
        f"traits: {', '.join(p.get('traits', []))}",
        f"likes: {', '.join(p.get('likes', []))}",
        f"dislikes: {', '.join(p.get('dislikes', []))}",
        f"goals: {', '.join(p.get('goals', []))}",
        f"mood: {bot.mood} — {bot.mood_sentence}",
        f"user: {username}, intent: {usertone['intent']}, tone: {usertone['tone']}, attitude: {usertone['attitude']}",
    ]

    if context:
        lines.append("context:")
        lines.append(context.strip())
    if memory:
        lines.append("memory:")
        lines.extend(line for line in memory.split("\n") if line.strip())

    lines.append(f"<|user|> {username}: {question.strip()}")
    lines.append(f"<|user1|> {bot.name}:")

    if extra_context:
        lines.append(f"<ActionResult> {extra_context.strip()}")

    if include_reflection:
        lines.append("reflection:")
        lines.append("- Emotional reaction?")
        lines.append("- Does this relate to your goals or traits?")
        lines.append("- Typical response?")
        lines.append("- Internal conflict?")
        mood_note = {
            "happy": "Joy may lead to idealism.",
            "annoyed": "You might be blunt or impatient.",
            "angry": "Be careful not to overreact.",
            "sad": "Are you being too negative?",
            "anxious": "Reground in your values.",
        }.get(bot.mood, "")
        if mood_note:
            lines.append(mood_note)

    return "\n".join(lines)
