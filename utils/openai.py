def extract_generated_text(response):
    """
    Extract generated text from llama-cpp-python completion response
    that may be either OpenAI-like with 'choices' or simple with 'text'.

    Args:
        response (dict): The output dict from model.create_completion()

    Returns:
        str: Generated text, or empty string if not found.
    """
    if not isinstance(response, dict):
        return ""

    # OpenAI style response with choices list
    if "choices" in response and isinstance(response["choices"], list):
        if len(response["choices"]) > 0 and "text" in response["choices"][0]:
            return response["choices"][0]["text"]

    # Simple style response with top-level 'text'
    if "text" in response:
        return response["text"]

    return ""

def translate_llama_prompt_to_chatml(prompt: str) -> str:
    """
    Converts a <|system|>, <|user|>, <|assistant|> formatted prompt into ChatML-style prompt.

    Returns the translated prompt.
    """
    # Define replacements in order
    replacements = [
        ("<|system|>\n", "<|im_start|>system\n"),
        ("<|user|>\n", "<|im_start|>user\n"),
        ("<|assistant|>\n", "<|im_start|>assistant\n"),
    ]

    # Apply each replacement
    for old, new in replacements:
        prompt = prompt.replace(old, new)

    # Add end tags
    # Insert <|im_end|> after each <|im_start|>block until assistant block
    parts = prompt.split("<|im_start|>")
    final = []
    for part in parts:
        if not part.strip():
            continue
        if part.strip().startswith("assistant"):
            final.append(f"<|im_start|>{part.strip()}")  # assistant gets no <|im_end|>
        else:
            final.append(f"<|im_start|>{part.strip()}\n<|im_end|>")

    return "\n".join(final)
