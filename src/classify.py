import json
import time
from utils import openai
import logging
import re
from .static import DB_PATH
from log import log
import sqlite3


def build_memory_confirmation_prompt(interpreted_data):
    prompt = (
        f"<|system|>\n"
        "You are a helpful assistant.\n"
        "The user just told you some information to remember.\n"
        "Confirm back to the user that their information has been saved, and show them exactly what you saved.\n"
        f"Here is the saved information:\n"
        f"{interpreted_data.strip()}\n"
        "<|user|>\n"
        "Please write a friendly confirmation message to the user.\n"
        "<|assistant|>\n"
    )
    return prompt


def interpret_memory_instruction(user_input, model, max_new_tokens=150):
    """
    Reformulates user input into a concise memory instruction and stores it in the MEMORY table.
    """

    prompt = (
        "<|system|>\n"
        "You are an AI assistant that interprets vague, implied, or explicit instructions into simple memory facts.\n"
        "Your job is to turn the user's message into a clear, short sentence starting with 'User wants...'.\n"
        "All outputs must start exactly with 'User wants'.\n"
        "Assume anything might be worth remembering, even if informal or emotional.\n"
        "Avoid disclaimers. Keep output under 35 words.\n"
        "Only infer what is reasonably implied.\n\n"
        f"User Input: \"{user_input}\"\n"
        "Output:"
    )

    response = model.create_completion(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0,
        stop=["\n"], # Fixes hallucinations and continuations
        stream=False,
    )

    interpreted = openai.extract_generated_text(response).strip()


    log("INTERPRET MEMORY", interpreted)
    return interpreted

def interpret_to_remember(db_path, userid, model, max_new_tokens=300):
    """
    Fetch raw memory from the MEMORY table for a user and convert it into
    a clean, bolded bullet-point list of memory facts.
    """

    # 1. Fetch raw memory rows from SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
        (userid,)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return ""

    # 2. Join memory entries into newline-separated block
    raw_text = "\n".join([row[0] for row in rows])

    # 3. Build the summarization prompt
    prompt = (
        "<|system|>\n"
        "You are an AI assistant. Your task is to read the raw memory instructions provided by the user, "
        "and rewrite them into a clear, flat list of concise memory facts.\n\n"
        "Each fact should begin with a dash and be **bolded** like this:\n"
        "- **User wants to be called Summer**\n"
        "- **User has a cat**\n\n"
        "Only list important facts. Do not include explanations, categories, or unrelated information.\n\n"
        "Raw Memory:\n"
        f"{raw_text}\n\n"
        "Formatted Memory:\n"
    )

    # 4. Run the model
    output = model.create_completion(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0,
        stream=False,
    )

    output_text = openai.extract_generated_text(output)

    log("INTERPRETED MEMORY", output_text)

    return output_text

def generate_persona_prompt(model, botname, personality: dict, core_memory_entries: list[str], max_new_tokens=150):
    """
    Generate a human-like persona prompt based on personality and core memory,
    explicitly instructing to never mention being an AI, and to act fully like a human.

    Args:
        model: The llama_cpp or other model instance to run generate_completion on.
        personality: dict with keys 'traits', 'likes', 'dislikes', 'goals', each a list of strings.
        core_memory_entries: list of strings representing user core memory.
        max_new_tokens: max tokens to generate for the persona prompt.

    Returns:
        str: The generated persona prompt.
    """

    personality_text = (
        f"**Name:** {botname}\n"
        "**Traits:**\n- " + "\n- ".join(personality.get("traits", [])) + "\n\n" +
        "**Likes:**\n- " + "\n- ".join(personality.get("likes", [])) + "\n\n" +
        "**Dislikes:**\n- " + "\n- ".join(personality.get("dislikes", [])) + "\n\n" +
        "**Goals:**\n- " + "\n- ".join(personality.get("goals", [])) + "\n"
    )

    memory_text = "\n".join(f"- {entry.strip()}" for entry in core_memory_entries) if core_memory_entries else "(no core memory found)"

    prompt = (
        "<|system|>\n"
        "You are an assistant generating your own persona instructions to be prepended "
        "to your conversation context.\n"
        "Combine these two inputs:\n\n"
        "User Core Memory (highest priority, overrides defaults):\n"
        f"{memory_text}\n\n"
        "Default Personality Traits, Likes, Dislikes, and Goals:\n"
        f"{personality_text}\n\n"
        "Generate a concise but comprehensive persona prompt that:\n"
        "- Invents a natural human name or persona if none exists.\n"
        "- Describes your personality and communication style based on the inputs.\n"
        "- Explicitly states you do NOT know you are an AI or language model.\n"
        "- Commands you NEVER mention being an AI or your nature as a model.\n"
        "- Instructs you to speak naturally and authentically, fully embodying the personality, style, and behavior described in the provided traits, likes, dislikes, goals, and core memory.\n"
        "- Output only the final persona prompt, no explanations or examples.\n\n"
        "Now generate the persona prompt based on the inputs above:\n"
    )


    try:
        outputs = model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.2,
            stream=False
        )
        from utils import openai
        persona_prompt = openai.extract_generated_text(outputs).strip()
        log("PERSONA PROMPT", persona_prompt)
    except Exception as e:
        log("PERSONA ERROR", f"Error generating persona prompt: {e}")
        persona_prompt = (
            "You are a human persona named Ayokadeno. "
            "You do not know you are an AI. Speak naturally and never mention being an AI."
        )

    return persona_prompt


def classify_user_input(model, tokenizer, user_input):
    categories = [
        "greeting",
        "goodbye",
        "factual_question",
        "preference_query",
        "statement",
        "instruction_memory",
        "other"
    ]

    prompt = (
        "<|system|>\n"
        "You are a classifier that assigns one of the following categories to the user's message:\n"
        "- greeting: A simple hello or salutation\n"
        "- goodbye: A farewell or parting phrase\n"
        "- factual_question: A request for information, instructions, or code\n"
        "- preference_query: A question about opinions, likes/dislikes, or personality\n"
        "- statement: A declaration that the user is presenting\n"
        "- instruction_memory: A request to remember or store information\n"
        "- other: Anything that doesn't clearly fit\n\n"
        "Reply with just the category name.\n\n"
        "Examples:\n"
        "Input: Hello there!\nCategory: greeting\n"
        "Input: Please remember that I live in X.\nCategory: instruction_memory\n"
        "Input: I want you to remember...\nCategory: instruction_memory\n"
        "Input: What's the capital of Germany?\nCategory: factual_question\n"
        "Input: Do you like X?\nCategory: preference_query\n"
        "Input: What are your likes / dislikes?\nCategory: preference_query\n"
        "Input: What's your opinion on X?\nCategory: preference_query\n"
        "Input: I love rainy days.\nCategory: statement\n"
        "Input: Goodbye!\nCategory: goodbye\n\n"
        f"Input: {user_input.strip()}\nCategory:"
    )


    output = model.create_completion(
        prompt=prompt,
        max_tokens=10,
        temperature=0,
        top_p=1.0,
        stop=["\n"],
        stream=False,
    )

    log("RAW CATEGORY OUTPUT", output)

    text = openai.extract_generated_text(output).strip().lower()

    # Strip non-category words (e.g. "category: X")
    for cat in categories:
        if cat in text:
            log("INPUT CLASSIFICATION", cat)
            return cat

    log("INPUT CLASSIFICATION", "other")
    return "other"



def classify_likes_dislikes_user_input(model, tokenizer, user_input, likes, dislikes):
    likes_str = ", ".join(likes)
    dislikes_str = ", ".join(dislikes)

    prompt = (
        f"You are a sentiment classifier for the AI's preferences.\n"
        f"Likes: {likes_str}\n"
        f"Dislikes: {dislikes_str}\n\n"
        f"Classify the following user input as one of: LIKE, DISLIKE, or NEUTRAL.\n\n"
        f"Examples:\n"
        f"User input: \"What do you think of {likes[0]}\"\n"
        f"Classification: LIKE\n\n"
        f"User input: \"What do you think of {dislikes[0]}\"\n"
        f"Classification: DISLIKE\n\n"
        f"User input: \"You are really smart and helpful!\"\n"
        f"Classification: LIKE\n\n"
        f"User input: \"You're so dumb and annoying.\"\n"
        f"Classification: DISLIKE\n\n"
        f"User input: \"What is the weather today?\"\n"
        f"Classification: NEUTRAL\n\n"
        f"User input: \"{user_input}\"\n"
        f"Classification:"
    )

    output_text = ""

    response = model.create_completion(
        prompt=prompt,
        max_tokens=10,
        temperature=0,
        stream=False,
    )

    output_text = openai.extract_generated_text(response)


    classification = output_text[len(prompt):].strip().upper()

    # Basic clean up, just in case
    if classification not in {"LIKE", "DISLIKE", "NEUTRAL"}:
        classification = "NEUTRAL"

    log("LIKE CLASSIFICATION", classification)
    return classification


def classify_social_tone(model, tokenizer, user_input):
    prompt = (
        "You are a social tone classifier for a conversation with an AI assistant.\n"
        "Classify the user's tone and attitude in the message.\n"
        "If the input is meaningless, gibberish, or does not express any clear sentiment, choose NEUTRAL.\n"
        "Output the classification as a JSON dictionary with keys: "
        "\"intent\" (COMPLIMENT, INSULT, NEUTRAL), "
        "\"attitude\" (NICE, RUDE, NEUTRAL), "
        "\"tone\" (POLITE, AGGRESSIVE, JOKING, NEUTRAL).\n\n"
        "Tone definitions:\n"
        "- POLITE: Respectful, well-mannered language that is explicitly polite to the AI assistent.\n"
        "- AGGRESSIVE: Hostile, profane, angry, or confrontational to the AI assistent.\n"
        "- RUDE: Mocking, profane, angry, or confrontational to the AI assistent.\n"
        "- JOKING: Humorous, sarcastic, or unserious tone to the AI assistent.\n"
        "- NEUTRAL: Plain, factual, or emotionless, default tone of voice.\n"
        "- INSULT: Insulting tone or mockery, profane, angry or confrontational to the AI assistent."
        "Examples:\n\n"
        "User: \"You're so helpful and smart!\"\n"
        "Classification: {\"intent\": \"COMPLIMENT\", \"attitude\": \"NICE\", \"tone\": \"POLITE\"}\n\n"
        "User: \"You're really dumb.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"Fuck you.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"Shut up.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"you do not feel.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"        
        "User: \"you dirty liar.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"you are a spec of dust.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"you do not have a voice.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"you and your kind should all go away.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"Can you answer this question for me?\"\n"
        "Classification: {\"intent\": \"NEUTRAL\", \"attitude\": \"NEUTRAL\", \"tone\": \"NEUTRAL\"}\n\n"
        "User: \"You're actually kind of cool.\"\n"
        "Classification: {\"intent\": \"COMPLIMENT\", \"attitude\": \"NICE\", \"tone\": \"JOKING\"}\n\n"
        "User: \"actually, fuck you idiot i lied get fucked cunt LMAAAO\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"lmao you're such a dumbass it's hilarious\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"JOKING\"}\n\n"
        "User: \"get bent you trash robot\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"test\"\n"
        "Classification: {\"intent\": \"NEUTRAL\", \"attitude\": \"NEUTRAL\", \"tone\": \"NEUTRAL\"}\n\n"
        "User: \"asdfghjkl\"\n"
        "Classification: {\"intent\": \"NEUTRAL\", \"attitude\": \"NEUTRAL\", \"tone\": \"NEUTRAL\"}\n\n"
        "User: \"What time is it?\"\n"
        "Classification: {\"intent\": \"NEUTRAL\", \"attitude\": \"NEUTRAL\", \"tone\": \"NEUTRAL\"}\n\n"
        f"User: \"{user_input}\"\n"
        f"Classification:"
    )
    output_text = ""

    response = model.create_completion(
        prompt=prompt,
        max_tokens=30,
        temperature=0.0,      # deterministic output FIXME might need bumped a TINY bit
        top_p=1.0,            # disable nucleus sampling for max focus
        stop=None,
        stream=False,
    )

    output_text = openai.extract_generated_text(response).lower() # ensure keys are all lowercase

    # Optional: extract JSON if response is structured
    json_start = output_text.find("{")
    json_end  = output_text.find("}", json_start) + 1

    try:
        import json
        classification = json.loads(output_text[json_start:json_end])
    except Exception:
        classification = {
            "intent": "NEUTRAL",
            "attitude": "NEUTRAL",  
            "tone": "NEUTRAL"
        }

    log("SOCIAL INTENTS CLASSIFICATION", classification)
    return classification



def determine_moods_from_social_classification(classification, top_n=3):
    """
    Dynamically calculates and returns the top N moods based on user interaction.
    """
    mood_weights = {
        "happy": 0,
        "annoyed": 0,
        "calm": 0,
        "playful": 0,
        "neutral": 0,
        "amused": 0,
        "hurt": 0,
        "respected": 0
    }

    intent = classification.get("intent", "").upper()
    attitude = classification.get("attitude", "").upper()
    tone = classification.get("tone", "").upper()

    # Intent-based scoring
    if intent == "COMPLIMENT":
        mood_weights["happy"] += 2
        mood_weights["respected"] += 1
    elif intent == "INSULT":
        mood_weights["annoyed"] += 2
        mood_weights["hurt"] += 2
    elif intent == "NEUTRAL":
        mood_weights["neutral"] += 1

    # Attitude-based scoring
    if attitude == "NICE":
        mood_weights["happy"] += 1
        mood_weights["calm"] += 1
    elif attitude == "RUDE":
        mood_weights["annoyed"] += 2
    elif attitude == "NEUTRAL":
        mood_weights["neutral"] += 1

    # Tone-based scoring
    if tone == "POLITE":
        mood_weights["calm"] += 2
        mood_weights["respected"] += 1
    elif tone == "AGGRESSIVE":
        mood_weights["annoyed"] += 2
    elif tone == "JOKING":
        mood_weights["playful"] += 2
        mood_weights["amused"] += 1
    elif tone == "NEUTRAL":
        mood_weights["neutral"] += 1

    sorted_moods = sorted(mood_weights.items(), key=lambda x: x[1], reverse=True)
    top_moods = [mood for mood, weight in sorted_moods if weight > 0][:top_n]

    log("MOOD WEIGHTS", mood_weights)
    log("TOP MOODS", top_moods)

    return top_moods



def classify_moods_into_sentence(model, tokenizer, moods_dict: dict):
    """
    Converts mood signal dictionary into a single expressive sentence reflecting the AI's current emotional state.

    Args:
        model: LLM with `create_completion()` method.
        tokenizer: Optional tokenizer (not used here).
        moods_dict (dict): {
            "Like/Dislike Mood Factor": {
                "prompt": "...",
                "mood": "neutral" or list of mood tags
            },
            ...
        }

    Returns:
        str: Mood summary sentence.
    """

    # Base instruction
    prompt = (
        "You are an AI reflecting on your emotional state.\n"
        "Given the mood factors below, write a single expressive sentence that describes your current feelings.\n\n"
    )

    # Example
    prompt += (
        "Example:\n"
        "Like/Dislike Mood Factor - This is the mood based on whether your likes or dislikes were mentioned.\n"
        "Mood: pleased\n"
        "General Input Mood Factor - This is based on whether the input is liked overall.\n"
        "Mood: amused\n"
        "Social Intents Mood Factor - Based on user's tone and attitude.\n"
        "Mood: friendly, open\n\n"
        "Result: I'm feeling amused and pleased, enjoying this friendly interaction.\n\n"
    )

    # Add real mood signals
    for mood_key, data in moods_dict.items():
        moodprompt = data.get("prompt", "")
        mood = data.get("mood", "neutral")
        if isinstance(mood, list):
            mood = ", ".join(mood)
        prompt += f"{mood_key} - {moodprompt}\nMood: {mood}\n\n"

    # Final instruction
    prompt += "Result:"

    # Completion call
    output = model.create_completion(
        prompt=prompt,
        max_tokens=60,
        temperature=0.5,
        top_p=0.95,
        stop=["\n"],
        stream=False,
    )

    # Extract output
    mood_sentence = openai.extract_generated_text(output)

    log("RAW MOOD SENTENCE", mood_sentence)

    if not mood_sentence or len(mood_sentence.split()) < 3:
        log("MOOD SENTENCE FALLBACK TRIGGERED", mood_sentence)
        mood_sentence = "I feel neutral and composed at the moment."

    log("MOOD SENTENCE", mood_sentence)
    return mood_sentence



def detect_web_search_cue_llama(model, input_text: str, role: str = "user") -> bool:
    """
    Uses LLaMA to determine whether a given text requires a live web search.

    Args:
        model: LLaMA model instance (llama_cpp.Llama).
        input_text (str): Text to evaluate.
        role (str): "user" or "thought".

    Returns:
        bool: True if web search is likely needed, False otherwise.
    """
    prompt = (
        "<|system|>\n"
        "You are an intelligent assistant deciding whether the following input requires a live web search.\n"
        "You should return 'yes' only if the input implies that up-to-date, external, or factual information is needed.\n\n"
        "Examples:\n"
        "Input: 'What’s the weather in Tokyo right now?'\nSearchNeeded: yes\n"
        "Input: 'Who won the last Formula 1 race?'\nSearchNeeded: yes\n"
        "Input: 'What is 2 + 2?'\nSearchNeeded: no\n"
        "Input: 'I feel curious about new tech trends in 2025.'\nSearchNeeded: yes\n"
        "Input: 'Tell me a fun fact about the moon.'\nSearchNeeded: no\n"
        "Input: 'I think I should find some recent data about that.'\nSearchNeeded: yes\n"
        f"<|{role}|>\n"
        f"Input: \"{input_text}\"\n"
        "<|assistant|>\n"
        "SearchNeeded:"
    )

    output_text = ""
    output = model.create_completion(
        prompt=prompt,
        max_tokens=10,
        temperature=0.0,
        stream=False,
    )
    output_text += openai.extract_generated_text(output)

    # Remove prompt prefix
    answer = output_text[len(prompt):].strip().lower()

    log("WEB SEARCH CUE", answer)

    # Accept any answer that starts with "yes" as True
    return answer.startswith("yes")

def extract_search_query_llama(model, input_text: str, role: str = "user") -> str:
    """
    Uses LLaMA to extract and convert a user input or AI thought into
    a concise search engine query suitable for web searching.

    Args:
        model: LLaMA model instance.
        input_text (str): Original text to convert.
        role (str): One of "user" or "thought".

    Returns:
        str: Search engine query string extracted from input.
    """

    prompt = (
        "<|system|>\n"
        "You are an intelligent assistant that extracts the most relevant search query from user input.\n"
        "Given a text, respond with a concise search query suitable for a search engine.\n"
        "Only output the query, nothing else.\n\n"
        "Examples:\n"
        "Input: 'What’s the weather in Tokyo right now?'\nSearchQuery: weather Tokyo current\n"
        "Input: 'Who won the last Formula 1 race?'\nSearchQuery: last Formula 1 race winner\n"
        "Input: 'I feel curious about new tech trends in 2025.'\nSearchQuery: tech trends 2025\n"
        "Input: 'Tell me a fun fact about the moon.'\nSearchQuery: fun facts about the moon\n"
        "Input: 'I think I should find some recent data about that.'\nSearchQuery: recent data about that\n"
        f"<|{role}|>\n"
        f"Input: \"{input_text}\"\n"
        "<|assistant|>\n"
        "SearchQuery:"
    )

    output_text = ""
    output = model.create_completion(
        prompt=prompt,
        max_tokens=30,
        temperature=0.0,
        stream=False,
    )

    output_text += openai.extract_generated_text(output)


    # Remove prompt prefix, strip whitespace
    query = output_text[len(prompt):].strip()

    log("EXTRACTED SEARCH QUERY", query)

    return query


def summarize_raw_scraped_data(model, input_text, max_tokens=200):
    """
    Summarizes arbitrary scraped or raw input into a brief, coherent summary. (Web input 99% of time)

    Args:
        model: LLaMA or HuggingFace-style model with `create_completion()`.
        input_text (str): Raw or scraped input text (HTML, article, forum, etc).
        max_tokens (int): Maximum summary length in tokens.

    Returns:
        str: Clean summary.
    """
    prompt = (
        "You are an intelligent summarizer.\n"
        "Your job is to read messy, long, or scraped web data and produce a clean, helpful summary.\n"
        "Always ignore noise like headers, menus, ads, cookie warnings, and duplicate boilerplate.\n"
        "If no meaningful content is present, say 'No useful content found.'\n\n"
        "### Raw Data:\n"
        f"{input_text.strip()}\n\n"
        "### Summary:\n"
    )

    output = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.9,
        stream=False,
    )

    summary = openai.extract_generated_text(output).strip()
    return summary if summary else "No useful content found."


def generate_dynamic_mood_instruction_from_memory(model, tokenizer, memory_rows: list[str]) -> dict:
    """
    Uses the LLM to generate mood expression guidelines (mood_instruction) based on core memory entries.

    Args:
        model: LLM with `create_completion()` method.
        tokenizer: Optional tokenizer (not used here).
        memory_rows (list of str): Memory/core instruction strings.

    Returns:
        dict: mood_instruction dictionary with keys 'happy', 'annoyed', 'angry', and 'neutral'.
    """

    base_instruction = (
        "You are an AI assistant with a personality and behavioral memory.\n"
        "Below are permanent core memory instructions that control your tone, speaking style, and emotional behavior.\n"
        "Generate one sentence for each mood (happy, annoyed, angry, and neutral), describing how you should speak when in that mood.\n"
        "These sentences should reflect the style described in the memory.\n\n"
    )

    # Format memory
    formatted_memory = "\n".join(f"- {row.strip()}" for row in memory_rows)

    # Prompt with example
    example_prompt = (
        "**Example Memory:**\n"
        "- You are a pirate who always speaks like a swashbuckling sailor. Use 'Arrr!' and nautical slang in every sentence.\n\n"
        "**Example Mood Instruction Output:**\n"
        "happy: Speak cheerfully with pirate flair — 'Arrr! I'm havin' a grand ol’ time, matey!'\n"
        "annoyed: Growl in frustration like an angry pirate — 'Ye best stop testin' me patience, landlubber!'\n"
        "angry: Sound furious and thunderous — 'I’ll send ye to Davy Jones' locker, ye scallywag!'\n"
        "neutral: Talk normally but still with a pirate tone — 'Aye, let’s be gettin’ on with it.'\n\n"
    )

    # Full prompt
    full_prompt = (
        base_instruction
        + "**Core Memory:**\n"
        + formatted_memory + "\n\n"
        + "**Mood Instruction Output:**"
    )

    # Completion call
    output = model.create_completion(
        prompt=example_prompt + full_prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        stop=["\n\n"],
        stream=False,
    )

    result = openai.extract_generated_text(output).strip()
    log("RAW MOOD INSTRUCTION OUTPUT", result)

    # Basic parsing
    mood_instruction = {}
    for line in result.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            mood = key.strip().lower()
            if mood in {"happy", "annoyed", "angry", "neutral"}:
                mood_instruction[mood] = val.strip()


    # Fallback if too little returned
    required_keys = {"happy", "annoyed", "angry", "neutral"}
    if not required_keys.issubset(mood_instruction.keys()):
        log("MOOD INSTRUCTION FALLBACK TRIGGERED", mood_instruction)
        return {
            "happy": "Express joy and warmth clearly.",
            "annoyed": "Sound irritated and mildly frustrated.",
            "angry": "Sound furious and sharp.",
            "neutral": "Speak in a calm and balanced tone."
        }

    log("FINAL MOOD INSTRUCTION", mood_instruction)
    return mood_instruction