import json
from utils import openai
from typing import Dict, Tuple, List

def basic_preprocessing(
    model,
    user_input: str,
    likes: List[str],
    dislikes: List[str],
    history: str | None = None,
    max_new_tokens: int = 256,
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
        "greeting", "goodbye", "statement",
        
        "self_explain", # only used internally

        "preference_query", "instruction_memory", "task", "other" # most implemented
    ]
    
    likes_str = ", ".join(likes) or "None"
    dislikes_str = ", ".join(dislikes) or "None"
    convo = history or "**No Conversation History**"

    prompt = (
        "<|system|>\n"
        "You are a helpful assistant that performs four tasks on each message:\n"
        "1. Classify the user's social tone with fields: intent (COMPLIMENT, INSULT, NEUTRAL), attitude (NICE, RUDE, NEUTRAL), and tone (POLITE, AGGRESSIVE, JOKING, NEUTRAL).\n"
        "2. Classify the message category.\n"
        "3. Detect whether the message mentions a LIKE, DISLIKE, or NEUTRAL topic, based on the assistant's preferences.\n"
        "4. Generate a natural-sounding sentence summarizing the assistant’s current emotional state, based off all of the above.\n\n"

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
        "  \"like\": ..., \n"
        "  \"mood_sentence\": \"...\"\n"
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
            "social": {"intent": "NEUTRAL", "attitude": "NEUTRAL", "tone": "NEUTRAL"},
            "category": "other",
            "like": "NEUTRAL",
            "mood_sentence": "I feel neutral and composed at the moment."
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

    mood_sentence = parsed.get("mood_sentence", "").strip()
    if not mood_sentence or len(mood_sentence.split()) < 3:
        mood_sentence = "I feel neutral and composed at the moment."

    print(f"Preprocessing: {social}\n{category}\n{like}\n{mood_sentence}\n")
    return social, category, like, mood_sentence
