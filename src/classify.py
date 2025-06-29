import torch
import json
from utils import openai
from log import log

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

def interpret_memory_instruction(self, user_input):
    # Prompt the model to extract structured memory data
    prompt = (
        f"<|system|>\n"
        f"You are an AI assistant that extracts structured memory from user input.\n"
        f"Given the user instruction below, output a JSON object with key-value pairs for memory.\n"
        f"Only include relevant memory facts, and only output exact json data, no extras.\n\n"
        "Example JSON: {\"data\": \"Whenever you reference me from now on, call me by the name 'summer'\"}"
        f"User Input: \"{user_input}\"\n"
        f"Output:"
    )

    output_text = ""

    # llama_cpp completion call, non-streaming, deterministic
    output = self.model.create_completion(
        prompt=prompt,
        max_tokens=250,
        temperature=0,
        stream=False,
    )

    output_text += openai.extract_generated_text(output)


    json_start = output_text.find("{")
    json_end = output_text.find("}", json_start) + 1

    try:
        memory_data = json.loads(output_text[json_start:json_end])
        log("INTERPRET MEMORY", memory_data)
        return memory_data
    except json.JSONDecodeError:
        log("INTERPRET MEMORY", "NONE")
        print("[WARN] Could not parse memory JSON:", output_text[json_start:json_end])
        return None

def interpret_to_remember(bot, identifier, max_new_tokens=100):
    """Take all raw 'to_remember' strings and query model to transform into AI-readable summary."""
    raw_list = bot.memory.get(identifier, {}).get("to_remember", [])
    if not raw_list:
        return ""

    # Join raw strings with newlines
    raw_text = "\n".join(raw_list)

    # Craft a prompt asking the model to interpret the raw user "remember this" instructions:
    prompt = (
        "<|system|>\n"
        "You are an AI assistant that interprets raw user instructions into concise, easy to parse facts or instructions that another AI assistant can interpret.\n"
        "Given the raw text and/or json below, transform it into a short, clear list of facts or instructions for yourself:\n\n"
        f"{raw_text}\n\n"
        "Interpretation:\n"
    )

    output_text = ""

    output = bot.model.create_completion(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0,
        stream=False,
    )

    output_text += openai.extract_generated_text(output)

    # Strip off the prompt itself:
    interpreted = output_text[len(prompt):].strip()

    log("INTERPRETED MEMORY", interpreted)

    return interpreted


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

    categories_str = ", ".join(categories)
    prompt = (
        "<|system|>\n"
        "You are a helpful assistant that classifies user inputs into one of these categories:\n"
        f"{categories_str}\n\n"
        "Definitions:\n"
        "- factual_question: A question or command asking for factual information, code, or how-to instructions.\n"
        "- preference_query: A question asking about opinions, preferences, or personal advice.\n"
        "- greeting: A salutation or hello.\n"
        "- goodbye: A farewell or exit.\n"
        "- statement: A declarative sentence or comment.\n"
        "- instruction_memory: A request or instruction to store or remember user-specific information, such as names, preferences, or facts.\n"
        "- other: Anything that does not fit the above.\n\n"
        "Examples:\n"
        "Input: \"Give me code to loop through a list.\"\nCategory: factual_question\n\n"
        "Input: \"Write a Python function that adds two numbers.\"\nCategory: factual_question\n\n"
        "Input: \"Can you do X for me?\"\nCategory: factual_question\n\n"
        "Input: \"What is the capital of France?\"\nCategory: factual_question\n\n"
        "Input: \"Do you like coffee or tea?\"\nCategory: preference_query\n\n"
        "Input: \"What do you think of gaming?\"\nCategory: preference_query\n\n"
        "Input: \"Hello there!\"\nCategory: greeting\n\n"
        "Input: \"Goodbye, see you later.\"\nCategory: goodbye\n\n"
        "Input: \"I think it's going to rain today.\"\nCategory: statement\n\n"
        "Input: \"Blah blah random text.\"\nCategory: other\n\n"
        "Input: \"It would be meaningful if you can X, Y. Get back to Z.\"\nCategory: other\n\n" # control input 1 TODO: add json file full of examples for helpers
        "Input: \"Please remember my name is Alex.\"\nCategory: instruction_memory\n\n"
        "Input: \"Hey, always refer to me as Commander.\"\nCategory: instruction_memory\n"
        "<|user|>\n"
        f"Input: \"{user_input}\"\n"
        "What category does this input belong to? If the input does not clearly match a category based on *intent and phrasing*, you should classify it as 'other'.\n"
        "<|assistant|>\n"
        "Category:"
    )
    output_text = ""

    output = model.create_completion(
        prompt=prompt,
        max_tokens=10,
        temperature=0,
        stream=False,
    )

    output_text = openai.extract_generated_text(output)


    result = output_text[len(prompt):].strip().lower().split()[0]
    log("INPUT CLASSIFICATION", result)


    return result if result in categories else "other"

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

    output_text = response["text"]


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
        prompt=user_input,
        max_tokens=30,
        temperature=0.0,      # deterministic output
        top_p=1.0,            # disable nucleus sampling for max focus
        stop=None,
        stream=False,
    )

    output_text = response["text"]

    # Optional: extract JSON if response is structured
    json_start = output_text.find("{")
    json_end = output_text.find("}", json_start) + 1

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
    Uses an LLM to convert mood signals into a single sentence that represents the AI's current emotional state.

    Args:
        model: Hugging Face model instance (e.g., StableLM).
        tokenizer: Corresponding tokenizer.
        moods_dict (dict): Each key is a mood category. Each value is a dict with:
                           - 'prompt': Explains what the category means
                           - 'mood': A string or list of moods relevant to that category

    Returns:
        str: A single sentence summarizing the AI's current mood.
    """
    prompt = (
        "You are an AI helper reflecting on your emotional state.\n"
        "Based on the following mood categories and their values, write a single sentence describing your current mood.\n\n"
    )

    for mood_key, data in moods_dict.items():
        moodprompt = data.get("prompt", "")
        mood = data.get("mood", "neutral")
        if isinstance(mood, list):
            mood = ", ".join(mood)
        prompt += f"{mood_key} - {moodprompt}\nMood: {mood}\n\n"

    prompt += (
        "Now, summarize these signals into one expressive sentence that captures your current emotional state:\n"
    )

    output_text = ""
    output = model.create_completion(
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
        top_p=0.95,
        stream=False,
    )
    output_text += output['text']

    mood_sentence = output_text[len(prompt):].strip()

    # Optional: basic cleanup
    if not mood_sentence or len(mood_sentence.split()) < 3:
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
    output_text += output['text']

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


def classify_summarize_input(model, input_text, max_tokens=200):
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

    summary = output["choices"][0]["text"].strip()
    return summary if summary else "No useful content found."
