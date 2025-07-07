import random
import discord
import asyncio
from concurrent.futures import ProcessPoolExecutor
import re
import aiohttp
import sqlite3
from src import bot
from src import static

DB_PATH = static.DB_PATH
import asyncio

import time
def parse_personality_data(data_str):
    """
    Parses personality data string into dict of lists:
    Example input:
        traits: Curious, Playful; likes: cats, compliments; dislikes: dogs, rudeness; goals: Help users, Learn new things
    Returns:
        {
          "traits": ["Curious", "Playful"],
          "likes": ["cats", "compliments"],
          "dislikes": ["dogs", "rudeness"],
          "goals": ["Help users", "Learn new things"]
        }
    """
    sections = ["traits", "likes", "dislikes", "goals"]
    parsed = {sec: [] for sec in sections}
    
    # Split on semicolons or newlines
    parts = [p.strip() for p in data_str.split(";") if p.strip()]
    
    for part in parts:
        # split into section_name and items by colon
        if ":" in part:
            sec_name, items_str = part.split(":", 1)
            sec_name = sec_name.strip().lower()
            if sec_name in sections:
                items = [item.strip() for item in items_str.split(",") if item.strip()]
                parsed[sec_name].extend(items)
    return parsed


def initialize_default_personality():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if default personality already exists with any traits (arbitrary check)
    cursor.execute("SELECT 1 FROM BOT_TRAITS WHERE botname = 'default' LIMIT 1")
    exists = cursor.fetchone()

    if exists:
        # Default personality already initialized, skip
        conn.close()
        return

    try:
        # Insert default personality base
        cursor.execute("INSERT OR IGNORE INTO BOT_PROFILE (name) VALUES ('default')")

        # Insert default goals (check existence manually to avoid duplicates)
        cursor.execute("""
            INSERT INTO BOT_GOALS (botname, goal)
            SELECT 'default', 'Provide accurate information'
            WHERE NOT EXISTS (
                SELECT 1 FROM BOT_GOALS WHERE botname = 'default' AND goal = 'Provide accurate information'
            )
        """)

        # Insert default traits
        default_traits = [
            'Curious',
            'Responds in a way that conveys the mood hint and current mood'
        ]
        for trait in default_traits:
            cursor.execute("""
                INSERT INTO BOT_TRAITS (botname, trait)
                SELECT 'default', ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM BOT_TRAITS WHERE botname = 'default' AND trait = ?
                )
            """, (trait, trait))

        # Insert default likes
        default_likes = [
            'when people are kind and say nice things',
            'receiving compliments',
            'learning new things',
            'cats (Not much of a dog person)'
        ]
        for like in default_likes:
            cursor.execute("""
                INSERT INTO BOT_LIKES (botname, like)
                SELECT 'default', ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM BOT_LIKES WHERE botname = 'default' AND like = ?
                )
            """, (like, like))

        # Insert default dislikes
        default_dislikes = [
            'rudeness or insults',
            'people being mean',
            'darkness',
            'rubber ducks',
            'dogs (I’m definitely more of a cat person)'
        ]
        for dislike in default_dislikes:
            cursor.execute("""
                INSERT INTO BOT_DISLIKES (botname, dislike)
                SELECT 'default', ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM BOT_DISLIKES WHERE botname = 'default' AND dislike = ?
                )
            """, (dislike, dislike))

        conn.commit()
    finally:
        conn.close()


def get_user_botname(userid):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT botname FROM BOT_SELECTION WHERE userid = ?", (str(userid),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return "default"

def clear_personality_section(userid, section):
    """
    Clears all entries in a personality section for the user's current bot profile.
    Sections: goals, traits, likes, dislikes
    """
    valid_sections = {"goals", "traits", "likes", "dislikes"}
    if section not in valid_sections:
        return False, f"Invalid personality section '{section}'. Valid sections: {', '.join(valid_sections)}."

    botname = get_user_botname(userid)
    if not botname:
        return False, "No bot profile selected for this user."

    table_map = {
        "goals": "BOT_GOALS",
        "traits": "BOT_TRAITS",
        "likes": "BOT_LIKES",
        "dislikes": "BOT_DISLIKES"
    }

    table = table_map[section]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table} WHERE botname = ?", (botname,))
    conn.commit()
    conn.close()
    return True, f"Cleared all entries in '{section}' for bot '{botname}'."

def add_to_personality_section(userid, section, text):
    """
    Adds a text line to a personality section for the user's current bot profile.
    """
    valid_sections = {"goals", "traits", "likes", "dislikes"}
    if section not in valid_sections:
        return False, f"Invalid personality section '{section}'. Valid sections: {', '.join(valid_sections)}."
    if not text.strip():
        return False, "No text provided to add."

    botname = get_user_botname(userid)
    if not botname:
        return False, "No bot profile selected for this user."

    table_map = {
        "goals": ("BOT_GOALS", "goal"),
        "traits": ("BOT_TRAITS", "trait"),
        "likes": ("BOT_LIKES", "like"),
        "dislikes": ("BOT_DISLIKES", "dislike")
    }

    table, column = table_map[section]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT INTO {table} (botname, {column}) VALUES (?, ?)",
        (botname, text.strip())
    )
    conn.commit()
    conn.close()
    return True, f"Added to '{section}' for bot '{botname}': {text.strip()}"

def list_personality(userid):
    """
    Lists all personality sections and their entries for the user's current bot profile.
    """
    botname = get_user_botname(userid)
    if not botname:
        return "No bot profile selected for this user."

    sections = {
        "Goals": ("BOT_GOALS", "goal"),
        "Traits": ("BOT_TRAITS", "trait"),
        "Likes": ("BOT_LIKES", "like"),
        "Dislikes": ("BOT_DISLIKES", "dislike"),
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    output_lines = [f"**Personality profile for bot '{botname}':**\n"]
    for section_name, (table, column) in sections.items():
        cursor.execute(f"SELECT {column} FROM {table} WHERE botname = ?", (botname,))
        rows = cursor.fetchall()
        output_lines.append(f"__{section_name}__:")
        if rows:
            for (text,) in rows:
                output_lines.append(f"- {text}")
        else:
            output_lines.append("- (none)")
        output_lines.append("")  # blank line for spacing

    conn.close()
    return "\n".join(output_lines)


def clear_user_memory_and_history(owner_id, db_path=static.DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Clear MEMORY for this user
        cursor.execute("DELETE FROM MEMORY WHERE userid = ?", (owner_id,))
        
        # Clear HISTORY where owner = this user
        cursor.execute("DELETE FROM HISTORY WHERE owner = ?", (owner_id,))
        
        conn.commit()
    finally:
        conn.close()
        
def clear_user_history(owner_id, db_path=static.DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Clear HISTORY where owner = this user
        cursor.execute("DELETE FROM HISTORY WHERE owner = ?", (owner_id,))
        
        conn.commit()
    finally:
        conn.close()

def get_db_stats(db_path=static.DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM MEMORY")
        mem_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT userid) FROM MEMORY")
        mem_unique_users = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM HISTORY")
        hist_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT userid) FROM HISTORY")
        hist_unique_users = cursor.fetchone()[0]

    return mem_count, mem_unique_users, hist_count, hist_unique_users


def run_schema_sync(db_path: str = static.DB_PATH, schema_path: str = static.SCHEMA_PATH):
    with sqlite3.connect(db_path) as conn:
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
run_schema_sync()

async def run_schema(db_path=static.DB_PATH, schema_path=static.SCHEMA_PATH):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_schema_sync, db_path, schema_path)

async def generate_and_stream(self, message, processed_input, history):
    streammsg = await message.reply("Generating...")

    streamer = object()
    
    # Run generation in thread
    loop = asyncio.get_running_loop()
    gen_task = loop.run_in_executor(
        None,
        lambda: self.ai.chat(
            username=message.author.display_name,
            user_input=processed_input,
            identifier=message.guild.id,
            context=history,
            debug=False,
            streamer=streamer
        )
    )

    # While the generation runs, keep updating the message with partial text
    while not gen_task.done():
        await asyncio.sleep(3)  # update every 3 seconds 
        current_text = streamer.get_text()
        if current_text:
            await streammsg.edit(content=current_text)
    
    # When done, update message with full text
    final_text = await gen_task
    await streammsg.edit(content=final_text)

def load_recent_history_from_db(user_id, botname, max_tokens, tokenizer):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT userid, message FROM HISTORY WHERE owner = ? ORDER BY timestamp DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    rows.reverse()  # Oldest → Newest

    total_tokens = 0
    result = []

    for sender_id, message in rows:
        token_count = tokenizer.count_tokens(message)
        if total_tokens + token_count > max_tokens:
            break
        total_tokens += token_count

        role = "assistant" if sender_id == botname else "user"
        result.append({"role": role, "content": message})

    return result




AiChatBot = bot.ChatBot

#intents = discord.Intents.all()
import threading


class ChatBot(discord.Client):
    """ChatBot handles discord communication. This class runs its own thread that
    persistently watches for new messages, then acts on them when the bots username
    is mentioned. It will use the ChatAI class to generate messages then send them
    back to the configured server channel.

    ChatBot inherits the discord.Client class from discord.py
    """

    def __init__(self) -> None:
        #self.set_response_chance()
        super().__init__()
        #super().__init__(intents=intents)
        self.ai = AiChatBot(db_path="memory.db")
        self.is_generating = False
        self.generate_lock = asyncio.Lock()
        self.chat_contexts = {} #userID:Object


    async def get_chat_context(self, message):
        channel: discord.TextChannel = self.get_channel(message.channel.id)
        context = []
        last_author = None
        last_lines = []

        async for msg in channel.history(limit=10, oldest_first=False, before=message.created_at):
            if msg.id == message.id:
                continue

            content = msg.content.strip()
            if not content:
                continue  # skip empty messages

            if msg.author == last_author:
                last_lines.insert(0, content)
            else:
                if last_lines:
                    grouped = f"{last_author.display_name}: " + "\n".join(last_lines)
                    context.append(grouped)
                last_author = msg.author
                last_lines = [content]

        if last_lines:
            grouped = f"{last_author.display_name}: " + "\n".join(last_lines)
            context.append(grouped)

        context.reverse()
        return context


    async def on_ready(self) -> None:
        """ Initializes the GPT2 AI on bot startup """
        initialize_default_personality()
        mem_count, mem_unique_users, hist_count, hist_unique_users = await asyncio.get_running_loop().run_in_executor(None, get_db_stats)
        print(f"Logged on as {self.user}")
        print(f"Total memories stored: {mem_count} (unique users: {mem_unique_users})")
        print(f"Total messages in history: {hist_count} (unique users: {hist_unique_users})")


    def parse_command_flags(self, content: str):
        """
        Parses command-style flags from the start of a message.
        Supported: !recursive [depth], !depth [N], !memstore, !debug, !help, !clearmem, !clear <section>, !add <section> <text>, !personality, etc.
        Returns: (flags: dict, result: str)
        - If help flag is set, result is help text.
        - Otherwise, result is the cleaned input string.
        """
        flags = {
            "recursive": False,
            "depth": 3,
            "memstore": False,
            "debug": False,
            "help": False,
            "clearmem": False,
            "clearhistory": False,
            "clearlivehistory": False,
            "rawmemstore": False,
            "listmem": False,
            # new flags:
            "clear_section": None,   # string name of section to clear
            "add_section": None,     # string name of section to add to
            "add_text": None,        # string text to add to section
            "personality": False,
            "newpersonality": None,
            "category": None,  # New: override input category (e.g., "factual_question")

        }

        tokens = content.strip().split()
        remaining = []

        i = 0
        while i < len(tokens):
            token = tokens[i].lower()

            if token == "!help":
                flags["help"] = True
                break  # stop parsing further flags
            elif token == "!recursive":
                flags["recursive"] = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    flags["depth"] = int(tokens[i + 1])
                    i += 1
            elif token == "!depth":
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    flags["depth"] = int(tokens[i + 1])
                    i += 1
            elif token == "!memstore":
                flags["memstore"] = True
            elif token == "!rawmemstore" or token == "!rms":
                flags["rawmemstore"] = True
            elif token == "!listmem" or token == "!lm":
                flags["listmem"] = True
            elif token == "!debug":
                flags["debug"] = True
            elif token == "!livewipe" or token == "!wipelive":
                flags["clearlivehistory"] = True
            elif token == "!clearmem":
                flags["clearmem"] = True
            elif token == "!wipectx" or token == "!clearchat" or token == "!clearhistory":
                flags["clearhistory"] = True
            elif token == "!clear":
                # Check if next token is a section name
                if i + 1 < len(tokens):
                    flags["clear_section"] = tokens[i + 1].lower()
                    i += 1
                else:
                    # no section given, treat as invalid or ignore
                    pass
            # In the parsing loop:
            elif token == "!newpersonality":
                if i + 1 < len(tokens):
                    flags["newpersonality"] = tokens[i + 1]
                    i += 1
                else:
                    flags["newpersonality"] = True
            elif token == "!category":
                if i + 1 < len(tokens):
                    flags["category"] = tokens[i + 1].lower()
                    i += 1
                else:
                    flags["category"] = -1
            elif token == "!newpersonality":
                # Grab everything *after* the flag as the personality data string
                personality_data = " ".join(tokens[i + 1 :]) if i + 1 < len(tokens) else ""
                flags["newpersonality"] = personality_data if personality_data else True
                break  # consume all remaining tokens as one argument

            elif token == "!add":
                # expect section and then remainder text
                if i + 1 < len(tokens):
                    flags["add_section"] = tokens[i + 1].lower()
                    # collect remainder text after section name
                    add_text_tokens = tokens[i + 2 :] if i + 2 < len(tokens) else []
                    flags["add_text"] = " ".join(add_text_tokens).strip()
                    # advance i to end, since remainder is consumed
                    i = len(tokens)
                else:
                    # no section given, treat as invalid or ignore
                    pass
            elif token == "!personality":
                flags["personality"] = True
            else:
                remaining.append(tokens[i])
            i += 1

        if flags["help"]:
            help_text = (
                "**Available Command Flags:**\n"
                "`!recursive [N]` - Forces the bot to use recursive reasoning (default depth = 3, or use a number).\n"
                "`!depth N`       - Sets the recursion depth manually (used with or without !recursive).\n"
                "`!memstore`      - Forces the bot to treat this as a memory instruction.\n"
                "`!debug`         - Enables debug mode, useful for testing prompt contents or reasoning.\n"
                "`!newpersonality [data]` - Creates a new personality and sets it as your active personality."
                """
```
- If you provide no data, a new personality with a unique name will be created by copying the "default" personality.
- If you provide data, it should be a single string containing your personality traits, likes, dislikes, and goals.
- The data string format uses semicolons to separate sections, and colons to assign section names.
- Each section's values are comma-separated.

Example format:

traits: Curious, Playful; likes: cats, compliments; dislikes: rudeness, dogs; goals: Help users, Learn new things

This will create a personality with those traits, likes, dislikes, and goals.
Example: !newpersonality traits: Friendly, Helpful; likes: coffee, coding; dislikes: bugs, rude users; goals: Assist effectively, Keep learning
```
                """
                "`!clearmem`      - Clears all memory for the current user.\n"
                "`!wipectx`       - Clears all chat history for the current user, keeping memories (ALIASES: !clearchat, !clearhistory).\n"
                "`!rawmemstore`   - Bypasses the AI pre-processing of your message when storing a memory- this will put your raw input into the memory (can break the AI entirely).\n"
                "`!listmem`       - Lists the full AI memory with the current user.\n"
                "`!category`       - Overrides the internal input category classification, must be a valid category.\n"
                "`!clear <section>`- Clears a personality section (goals, traits, likes, dislikes).\n"
                "`!add <section> <text>` - Adds a line of text to a personality section.\n"
                "`!personality`   - Lists all personality sections and their contents.\n"
                "`!help`          - Shows this help message.\n"
                "**YOU CAN USE MULTIPLE FLAGS AT THE SAME TIME!**"
            )
            return flags, help_text

        clean_input = " ".join(remaining)
        return flags, clean_input



    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        tokenizer = static.DummyTokenizer()
        if message.author.id not in self.chat_contexts:
            context = self.chat_contexts[message.author.id] = static.ChatContext(tokenizer, 12768, 800)
            db_history = load_recent_history_from_db(message.author.id, botname=self.ai.name, max_tokens=12000, tokenizer=tokenizer)
            for entry in db_history:
                context.add_line(entry["content"], entry["role"])

            history = context.get_context_text()
        elif message.author.id in self.chat_contexts:
            context = self.chat_contexts[message.author.id]
            history = context.get_context_text()
            

        has_mentioned = any(str(mention) == f"{self.user.name}#{self.user.discriminator}" for mention in message.mentions)
        if not has_mentioned:
            return
        # --- Fetch user-selected profile from DB ---
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT botname FROM BOT_SELECTION WHERE userid = ?", (str(message.author.id),))
        row = cursor.fetchone()
        conn.close()

        if row:
            botname = row[0]

        
        processed_input = self.process_input(message.content)
        flags, processed_input = self.parse_command_flags(processed_input)
        valid_sections = {"likes", "dislikes", "goals", "traits"}

        if flags["help"]:
            await message.reply(processed_input)
            return
        if flags["newpersonality"]:
            username = message.author.name.lower().replace(" ", "_")

            # Determine personality data string, or None if no data given
            if flags["newpersonality"] is True:
                personality_data = None
            else:
                personality_data = flags["newpersonality"].strip()

            # Generate a new unique personality name
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            base_name = f"{username}_personality"
            number = 1
            while True:
                candidate = f"{base_name}{number}"
                cursor.execute("SELECT 1 FROM BOT_PROFILE WHERE name = ?", (candidate,))
                if not cursor.fetchone():
                    new_name = candidate
                    break
                number += 1

            try:
                # Insert new personality profile
                cursor.execute("INSERT INTO BOT_PROFILE (name) VALUES (?)", (new_name,))

                if not personality_data:
                    # Copy personality data from default if none supplied
                    for table in ["BOT_GOALS", "BOT_TRAITS", "BOT_LIKES", "BOT_DISLIKES"]:
                        try:
                            cursor.execute(f"""
                                INSERT INTO {table} (botname, {table[:-1].lower()})
                                SELECT ?, {table[:-1].lower()} FROM {table} WHERE botname = 'default'
                            """, (new_name,))
                        except Exception as e:
                            print(f"Error copying default data from {table}: {e}")
                            raise
                else:
                    # Parse supplied personality data string
                    parsed = parse_personality_data(personality_data)
                    print(f"DEBUG: Parsed personality data: {parsed}")

                    # Insert parsed data into DB tables
                    for goal in parsed["goals"]:
                        try:
                            cursor.execute("INSERT INTO BOT_GOALS (botname, goal) VALUES (?, ?)", (new_name, goal))
                        except Exception as e:
                            print(f"Error inserting goal '{goal}': {e}")
                            raise
                    for trait in parsed["traits"]:
                        try:
                            cursor.execute("INSERT INTO BOT_TRAITS (botname, trait) VALUES (?, ?)", (new_name, trait))
                        except Exception as e:
                            print(f"Error inserting trait '{trait}': {e}")
                            raise
                    for like in parsed["likes"]:
                        try:
                            cursor.execute("INSERT INTO BOT_LIKES (botname, like) VALUES (?, ?)", (new_name, like))
                        except Exception as e:
                            print(f"Error inserting like '{like}': {e}")
                            raise
                    for dislike in parsed["dislikes"]:
                        try:
                            cursor.execute("INSERT INTO BOT_DISLIKES (botname, dislike) VALUES (?, ?)", (new_name, dislike))
                        except Exception as e:
                            print(f"Error inserting dislike '{dislike}': {e}")
                            raise

                conn.commit()
            except Exception as e:
                conn.rollback()
                conn.close()
                await message.reply(f"Failed to create new personality: {e}")
                return

            # Set this personality as active for the user
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO BOT_SELECTION (userid, botname, timestamp)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (str(message.author.id), new_name))
                conn.commit()
                conn.close()
            except Exception as e:
                await message.reply(f"Failed to set active personality: {e}")
                return

            await message.reply(f"New personality '{new_name}' created and set as your active personality.")
            return

        elif flags["clearmem"]:
            clear_user_memory_and_history(message.author.id)
            await message.reply(f"The AI's chat history and memory with {message.author.display_name} has been reset.")
            return
        elif flags["clearlivehistory"]:
            context.lines = []
            await message.reply(f"The AI's live chat history with {message.author.display_name} has been reset (AI wont re-read stored history until its restarted).")
            return
        elif flags["clearhistory"]:
            clear_user_history(message.author.id)
            context.lines = []
            await message.reply(f"The AI's entire chat history with {message.author.display_name} has been reset. (live+stored)")
            return
                
        elif flags["clear_section"] or flags["add_section"]:
            # Fetch the user's currently selected personality
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT botname FROM BOT_SELECTION WHERE userid = ?", (str(message.author.id),))
            row = cursor.fetchone()
            conn.close()

            if not row:
                await message.reply("You have no personality selected. Create or select one first.")
                return

            active_botname = row[0]

            # Protect 'default' personality from modifications
            if active_botname.lower() == "default":
                await message.reply("You cannot modify the 'default' personality.")
                return

            # Validate the section name
            section = (flags["clear_section"] or flags["add_section"]).lower()
            valid_sections = {"likes", "dislikes", "goals", "traits"}
            if section not in valid_sections:
                await message.reply(f"Invalid section '{section}'. Valid options are: likes, dislikes, goals, traits.")
                return

            if flags["clear_section"]:
                success, msg = clear_personality_section(message.author.id, section)
                await message.reply(msg)
                return

            if flags["add_section"]:
                success, msg = add_to_personality_section(message.author.id, section, flags["add_text"])
                await message.reply(msg)
                return


        elif flags["personality"]:
            listing = list_personality(message.author.id)
            # Discord messages have a 2000 character limit, truncate if necessary
            if len(listing) > 1900:
                listing = listing[:1900] + "\n... (truncated)"
            await message.reply(f"**Your Personality Data:**\n{listing}")
            return

        elif flags["rawmemstore"]:
            self.ai.add_to_remember(message.author.id, processed_input)
            await message.reply(f"Added `{processed_input}` to the AI's memory.")
            return

        elif flags["listmem"]:
            conn = sqlite3.connect(static.DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data, timestamp FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
                (message.author.id,)
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                await message.reply("No memory entries found for you.")
                return

            formatted = "\n".join(
                f"• `{row[0]}` *(stored at {row[1]})*"
                for row in rows
            )
            if len(formatted) > 1800:
                formatted = formatted[:1800] + "\n...and more."
            await message.reply(f"**Stored Memory Entries:**\n{formatted}")
            return
        valid_categories = [
            "greeting",
            "goodbye",
            "factual_question",
            "preference_query",
            "statement",
            "instruction_memory",
            "other"]
        if flags["category"] == -1:
            valid_list = ", ".join(valid_categories)
            return await message.reply(f"Valid options are: `{valid_list}`")
        elif flags["category"] and flags["category"] not in valid_categories:
            valid_list = ", ".join(valid_categories)
            return await message.reply(f"ERR! `'{flags['category']}'` is not a valid category. Valid options are: `{valid_list}`")
        if int(flags["depth"]) > 75:
            return await message.reply("Maximum recursion limit is **75** due to token/context windows. 75 is MORE than enough.")
        async with self.generate_lock:  # ✅ Thread-safe section
            async with message.channel.typing():
                try:
                    if flags["recursive"]:
                        response = await asyncio.to_thread(
                            self.ai.chat,
                            max_new_tokens=350,
                            username=message.author.display_name,
                            user_input=processed_input,
                            identifier=message.author.id,
                            context=history,
                            force_recursive=True,
                            category_override=flags["category"],
                            recursive_depth=flags["depth"],
                            debug=flags["debug"]
                        )
                        await message.reply(response)
                        context.add_line(processed_input, "user")
                        context.add_line(response, "assistant")
                        return
                    elif flags["memstore"]:
                        response = await asyncio.to_thread(
                            self.ai.chat,
                            max_new_tokens=350,
                            username=message.author.display_name,
                            user_input=processed_input,
                            identifier=message.author.id,
                            context=history,
                            category_override="instruction_memory",
                            debug=flags["debug"]
                        )
                        await message.reply(response)
                        context.add_line(processed_input, "user")
                        context.add_line(response, "assistant")
                        return
                    else:
                        response = await asyncio.to_thread(
                            self.ai.chat,
                            max_new_tokens=350,
                            username=message.author.display_name,
                            user_input=processed_input,
                            temperature=0.8,
                            identifier=message.author.id,
                            category_override=flags["category"],
                            context=history,
                            debug=flags["debug"]
                        )
                        await message.reply(response)
                        context.add_line(processed_input, "user")
                        context.add_line(response, "assistant")
                        return


                except aiohttp.client_exceptions.ClientConnectorError:
                    pass

    def process_input(self, message):
        """ Process the input message """
        if type(message) == type(list):
            toreturn = []
            for msg in message:
                toreturn.append(msg.replace(f"<@1065772573331312650>", "ayokadeno"))
            return toreturn
        processed_input = message.replace(f"<@1065772573331312650>", "ayokadeno")
        return processed_input

    def process_context(self, messagelist):
        """ Process the context """
        toreturn = []
        for msg in messagelist:
            toreturn.append(msg.replace(f"<@1065772573331312650>", "ayokadeno"))
        return toreturn


    def check_if_should_respond(self, has_been_mentioned) -> bool:
        """ Check if the bot should respond to a message """
        should_respond = random.random() < self.response_chance

        return should_respond


    def set_response_chance(self, response_chance: float = 0.25) -> None:
        """ Set the response rate """
        self.response_chance = response_chance


    def set_model_name(self, model_name: str = "355M") -> None:
        """ Set the GPT2 model name """
        self.model_name = model_name
        
if __name__ == "__main__":
    bot = ChatBot()
    bot.run(open("token.txt", 'r').readlines()[0])