from .static import tokenizer
import torch
import json
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
        f"Only include relevant memory facts.\n\n"
        f"User Input: \"{user_input}\"\n"
        f"Output:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

    with torch.no_grad():
        output = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.5,
            top_p=0.9,
            do_sample=False,
        )

    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    json_str = raw_output[len(prompt):].strip()

    try:
        memory_data = json.loads(json_str)
        log("INTERPRET MEMORY", memory_data)
        return memory_data
    except json.JSONDecodeError:
        log("INTERPRET MEMORY", "NONE")

        print("[WARN] Could not parse memory JSON:", json_str)
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
        "You are an AI assistant that interprets raw user memory instructions into concise, easy to parse facts.\n"
        "Given the raw text below, transform it into a short, clear list of facts or instructions for yourself:\n\n"
        f"{raw_text}\n\n"
        "Interpretation:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(bot.model.device)
    with torch.no_grad():
        output = bot.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip off the prompt itself:
    interpreted = result[len(prompt):].strip()
    log("INTERPRETED MEMORY",interpreted)

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
        "- factual_question: A question asking for objective, factual information.\n"
        "- preference_query: A question asking for opinions, preferences, or personal advice.\n"
        "- greeting: A salutation or hello.\n"
        "- goodbye: A farewell or exit.\n"
        "- statement: A declarative sentence or comment.\n"
        "- instruction_memory: A request or instruction to store or remember user-specific information, such as names, preferences, or facts.\n"
        "- other: Anything that does not fit the above.\n\n"
        "Examples:\n"
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    result = result[len(prompt):].strip().lower().split()[0]
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    classification = tokenizer.decode(outputs[0], skip_special_tokens=True)
    classification = classification[len(prompt):].strip().upper()

    # Basic clean up, just in case
    if classification not in {"LIKE", "DISLIKE", "NEUTRAL"}:
        classification = "NEUTRAL"
    log("LIKE CLASSIFICATION", classification)
    return classification

def classify_social_tone(model, tokenizer, user_input):
    prompt = (
        "You are a sentiment and social tone classifier for a conversation with an AI assistant.\n"
        "Classify the user's tone and attitude in the following message.\n"
        "Output the classification as a JSON dictionary with keys: "
        "\"intent\" (COMPLIMENT, INSULT, NEUTRAL), "
        "\"attitude\" (NICE, RUDE, NEUTRAL), "
        "\"tone\" (POLITE, AGGRESSIVE, JOKING, NEUTRAL).\n\n"
        "Examples:\n\n"
        "User: \"You're so helpful and smart!\"\n"
        "Classification: {\"intent\": \"COMPLIMENT\", \"attitude\": \"NICE\", \"tone\": \"POLITE\"}\n\n"
        "User: \"You're really dumb.\"\n"
        "Classification: {\"intent\": \"INSULT\", \"attitude\": \"RUDE\", \"tone\": \"AGGRESSIVE\"}\n\n"
        "User: \"Can you answer this question for me?\"\n"
        "Classification: {\"intent\": \"NEUTRAL\", \"attitude\": \"NEUTRAL\", \"tone\": \"NEUTRAL\"}\n\n"
        "User: \"Just kidding, you're actually kind of cool.\"\n"
        "Classification: {\"intent\": \"COMPLIMENT\", \"attitude\": \"NICE\", \"tone\": \"JOKING\"}\n\n"
        f"User: \"{user_input}\"\n"
        f"Classification:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_start = result_text.find("{")
    json_end = result_text.find("}", json_start) + 1

    try:
        import json
        classification = json.loads(result_text[json_start:json_end])
    except Exception:
        classification = {
            "intent": "NEUTRAL",
            "attitude": "NEUTRAL",
            "tone": "NEUTRAL"
        }

    return classification
