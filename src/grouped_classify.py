import json
from utils import openai
from typing import Dict, Tuple, List

def basic_preprocessing(
    model,
    user_input: str,
    likes: List[str],
    dislikes: List[str],
    history: str | None = None,
    max_new_tokens: int = 60,
) -> Tuple[Dict[str, str], str, str]:
    """
    Performs social tone classification, category classification,
    and likes/dislikes sentiment detection using ONE generation.
    
    Returns:
        - social: {"intent", "attitude", "tone"}
        - category: e.g. "statement", "task", etc.
        - likes: "LIKE", "DISLIKE", or "NEUTRAL"
    """
    categories = [
        "greeting", "goodbye", "preference_query", "statement",
        "instruction_memory", "task", "other"
    ]
    
    likes_str = ", ".join(likes) or "None"
    dislikes_str = ", ".join(dislikes) or "None"
    convo = history or "**No Conversation History**"

    prompt = (
        "<|system|>\n"
        "You are a helpful assistant that performs three tasks on each message:\n"
        "1. Classify the user's social tone (intent, attitude, tone).\n"
        "2. Classify the message category.\n"
        "3. Detect whether the message mentions a LIKE, DISLIKE, or NEUTRAL topic, based on the assistant's preferences.\n\n"

        "Tone classification is a JSON object with:\n"
        "- intent: COMPLIMENT, INSULT, or NEUTRAL\n"
        "- attitude: NICE, RUDE, or NEUTRAL\n"
        "- tone: POLITE, AGGRESSIVE, JOKING, or NEUTRAL\n\n"

        "Category must be one of:\n"
        "- greeting: A simple hello or salutation\n"
        "- goodbye: A farewell or parting phrase\n"
        "- preference_query: A question about opinions, likes/dislikes, or personality\n"
        "- statement: A declaration that the user is presenting\n"
        "- instruction_memory: A request to remember or store information\n"
        "- task: A request to do something specific, or an instruction to do a task, like searching the web, doing math, running code, anything that might need access to internal tools.\n"
        "- other: Anything that doesn't clearly fit the listed categories.\n\n"

        "Assistant likes: " + likes_str + "\n"
        "Assistant dislikes: " + dislikes_str + "\n\n"
        
        "Return a JSON object with these fields:\n"
        "{\n"
        "  \"social\": {\"intent\": ..., \"attitude\": ..., \"tone\": ...},\n"
        "  \"category\": ..., \n"
        "  \"like\": ...\n"
        "}\n"
        "<|eot|>\n"

        f"{convo}" + "\n"
        f"<|user|>\n{user_input.strip()}\n<|eot|>\n"
        "<|assistant|>\n"
    )

    response = model.create_completion(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stream=False,
    )

    raw = openai.extract_generated_text(response)

    # Extract JSON
    try:
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        parsed = json.loads(raw[json_start:json_end])
    except Exception:
        parsed = {
            "social":   {"intent": "NEUTRAL", "attitude": "NEUTRAL", "tone": "NEUTRAL"},
            "category": "other",
            "likes":    "NEUTRAL",
        }

    # Clean + validate
    social = parsed.get("social", {})
    social = {
        "intent":   social.get("intent", "NEUTRAL").upper(),
        "attitude": social.get("attitude", "NEUTRAL").upper(),
        "tone":     social.get("tone", "NEUTRAL").upper(),
    }

    category = parsed.get("category", "other").lower()
    if category not in categories:
        category = "other"

    like = parsed.get("like", "NEUTRAL").upper()
    if like not in {"LIKE", "DISLIKE", "NEUTRAL"}:
        like = "NEUTRAL"

    return social, category, like
