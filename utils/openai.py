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
