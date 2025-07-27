from log import log
from .static import StopOnSpeakerChange, DB_PATH, WorkerConfig
from utils.helpers import DummyTokenizer
import sqlite3
import re
from . import static_prompts
from . import bot

import datetime

from .ai_actions import check_for_actions_and_run


def process_thinking_output(final_summary, identifier, botname, db_path=DB_PATH):
    # Extract instructions
    send_to_user = False
    user_message = None
    match = re.search(r"send_to_user:\s*true\s*message:\s*(.+?)(?=\n\S|$)", final_summary, re.IGNORECASE | re.DOTALL)
    if match:
        send_to_user = True
        user_message = match.group(1).strip()

    memory_updates = re.findall(r"memory_update:\s*(.+?)(?=\n\S|$)", final_summary, re.IGNORECASE | re.DOTALL)
    new_goals      = re.findall(r"new_goal:\s*(.+?)(?=\n\S|$)", final_summary, re.IGNORECASE | re.DOTALL)
    new_likes      = re.findall(r"new_like:\s*(.+?)(?=\n\S|$)", final_summary, re.IGNORECASE | re.DOTALL)
    new_dislikes   = re.findall(r"new_dislike:\s*(.+?)(?=\n\S|$)", final_summary, re.IGNORECASE | re.DOTALL)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    for memory in memory_updates:
        c.execute("INSERT INTO MEMORY (userid, data, timestamp) VALUES (?, ?, ?)",
                (identifier, memory.strip(), datetime.utcnow()))
    if botname.lower() != "default":
        for goal in new_goals:
            c.execute("INSERT INTO BOT_GOALS (botname, goal) VALUES (?, ?)",
                    (botname, goal.strip()))

        for like in new_likes:
            c.execute("INSERT INTO BOT_LIKES (botname, like) VALUES (?, ?)",
                    (botname, like.strip()))

        for dislike in new_dislikes:
            c.execute("INSERT INTO BOT_DISLIKES (botname, dislike) VALUES (?, ?)",
                    (botname, dislike.strip()))

    conn.commit()
    conn.close()

    return send_to_user, user_message

class AutonomousPassiveThinker:
    def __init__(self, worker_config: WorkerConfig, config, persona_prompt, depth=3, streamer=None, tiny_mode=False):
        self.worker_config: WorkerConfig = worker_config
        
        self.depth = depth
        self.config = config
        self.persona_prompt = persona_prompt
        self.streamer = streamer
        self.tiny_mode = tiny_mode

    def build_autonomous_prompt(self, username, identifier):
        personality = bot.list_personality(identifier)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()

        user_info_section = (
                f"### **{username}'s Profile and Info*:*\n"
                f"**Name:** {username.replace('<', '').replace('>', '')}\n"
        )
        persona_section = static_prompts.build_base_personality_profile_prompt(
            self.bot.name, self.persona_prompt, personality, self.bot.mood, self.bot.mood_sentence)
        rules_section = static_prompts.build_rules_prompt(self.bot.name, username, None)
        memory_instructions_section = static_prompts.build_memory_instructions_prompt()
        memory_section = static_prompts.build_core_memory_prompt(rows if rows else None)

        base = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are a personality-driven assistant named {self.bot.name}.\n"
            f"{persona_section}"
            f"{user_info_section}"
            f"{memory_instructions_section}"
            f"{memory_section}"
            f"{rules_section}"
        )

        base += (
            "\n### Autonomous Passive Thinking Task\n"
            "You have complete freedom to think about your internal goals, memory, and personality.\n"
            "You may decide to ADD to your goals, likes, dislikes, or memory- but not remove.\n"
            "You may choose to send a message to the user without them prompting you.\n"
            "You do not need any external question or input.\n"
            "Reason step-by-step about what you want to do next.\n"
        )

        base += (
            "\n### Internal Reflection\n"
            "- What is on your mind?\n"
            "- Are there goals you want to pursue or change?\n"
            "- Is there new knowledge to add to your memory?\n"
            "- Should you communicate with user?\n"
        )

        base += "<|eot_id|>"
        log("AUTONOMOUS PASSIVE PROMPT", base)
        return base

    def think(self, username, identifier=None, tier="t0"):
        tokenizer = DummyTokenizer()

        prompt = self.build_autonomous_prompt(username=username, identifier=identifier)

        full = prompt
        
        prior_steps = []
        extra_context_lines = []

        custom_stops = [f"<|system|>", f"<|{self.bot.name}|>"]
        stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops)

        to_add = ""

        for step in range(self.depth):
            step_prompt = f"{prompt}"

            if extra_context_lines:
                step_prompt += "\n".join(extra_context_lines) + "\n"
                extra_context_lines.clear()

            step_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            step_prompt += f"### Autonomous Passive Thought Step\n"

            if self.streamer:
                self.streamer.add_special(f"Autonomous passive thinking step {step+1}...")

            response = self.bot._straightforward_generate(
                step_prompt,
                max_new_tokens=self.config.token_config[tier]["RECURSIVE_BACKGROUND_THINKING_TOKENS_PER_STEP"],
                temperature=0.8,
                top_p=0.9,
                streamer=self.streamer,
                stop_criteria=stop_criteria,
                _prompt_for_cut=step_prompt,
            )
            step_content = response.strip()
            log(f"DEBUG: AUTONOMOUS PASSIVE THOUGHT STEP {step}", step_content)

            action_result = check_for_actions_and_run(
                self.worker_config.tools, self.bot.model, response,
                max_token_window=self.config.token_config[tier]["BASE_TOKEN_WINDOW"],
                max_chat_window=self.config.token_config[tier]["BASE_TOKEN_WINDOW"],
                prompt_size=self.config.token_config["PROMPT_RESERVATION"]
            )

            prior_steps.append(step_content)

            to_add += "<|start_header_id|>assistant<|end_header_id|>\n"
            to_add += f"### Autonomous Passive Thought {step+1} of {self.depth}\n{step_content}\n\n"
            to_add += "<|eot_id|>"

            if action_result != "NOACTION":
                if isinstance(action_result, list):
                    for result in action_result:
                        to_add += f"\n{result}"
                else:
                    to_add += f"\n{action_result}"

        discord_formatting_prompt = static_prompts.build_discord_formatting_prompt()

        final_prompt = (
            full
            + "\n### Autonomous Passive Final Summary\n"
            + "_Summarize your previous thoughts, planned actions, and updates._\n"
            + "**Rules:**\n"
            + "- No user-facing reply unless triggered.\n"
            + "- Restate key points clearly for internal use.\n"
            + "- You may update internal structures as needed.\n"
            + discord_formatting_prompt
            + "\n### How to send a message to the user\n"
            + "- If you want to proactively communicate, add this at the end:\n"
            + "send_to_user: true\n"
            + "message: Hello! I've been thinking about our last conversation.\n"
            + "\n### How to update memory\n"
            + "- To add something to memory, include:\n"
            + "memory_update: I’ve realized that...\n"
            + "\n### How to add a new goal\n"
            + "- To set a new goal, include:\n"
            + "new_goal: Help the user feel more confident during emotional struggles.\n"
            + "\n### How to add a like\n"
            + "- To add a like, include:\n"
            + "new_like: I enjoy understanding complex emotional patterns.\n"
            + "\n### How to add a dislike\n"
            + "- To add a dislike, include:\n"
            + f"new_dislike: I dislike feeling disconnected from the '{username}'s emotional state.\n"
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )


        prompt_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL AUTONOMOUS PASSIVE PROMPT:\n", final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", prompt_tokens_used)

        if self.streamer:
            self.streamer.add_special(f"Finalizing autonomous passive thought summary")

        final_summary = self.bot._straightforward_generate(
            max_new_tokens=self.config.token_config[tier]["RECURSIVE_BACKGROUND_THINKING_TOKENS_FINAL"],
            temperature=0.7,
            top_p=0.9,
            streamer=self.streamer,
            stop_criteria=stop_criteria,
            prompt=final_prompt,
            _prompt_for_cut=final_prompt
        ).strip()

        log("\n\nDEBUG: AUTONOMOUS PASSIVE THINKING GENERATION", final_summary)
        final_tokens_used = tokenizer.count_tokens(final_prompt)
        log(f"DEBUG: FINAL TOKEN SIZE:", final_tokens_used)

        send_to_user, user_message = process_thinking_output(
            final_summary=final_summary,
            identifier=identifier,
            botname=bot.get_user_botname(identifier),
        )
        return final_summary, send_to_user, user_message
