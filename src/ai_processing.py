from utils import openai

def summarize_raw_scraped_data(model, input_text, max_tokens=2048): # TODO: move to seperate file that can summarize any raw data and go through it in chunks if its too large
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
        "<|start_header_id|>system<|end_header_id|>\n"

        "You are a summarizer.\n"
        "You read raw, unstructured data (HTML, text, forums, JSON, etc) and describe it as if explaining it to someone.\n"
        "Summarize with rich, natural language, in paragraph form.\n"
        "Capture the overall purpose of the page, any key content (product, game, article, thread, etc), and what a visitor would expect to find.\n"
        "Make sure you include specific features, themes, or functionality if relevant.\n"
        "Avoid referencing ads or cookies.\n"
        "If the page has no content, say: 'No useful content found.'\n\n"
    )
    prompt += (
        "### Raw Data:\n"
        f"{input_text.strip()}\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"

    )

    output = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )

    print(f"WEB SUMMARY!!!!: {output}")
    summary = openai.extract_generated_text(output).strip()
    return summary if summary else "No useful content found."

def summarize_chunks(model, input_text, category: str = None, override_category_instruction: str = None, max_tokens=2048):
    """
    Summarizes arbitrary chunks of data into one summary

    Args:
        model: LLaMA or HuggingFace-style model with `create_completion()`.
        input_text (str): AI readable and formatted string containing all chunks to summarize.
        category (str): The 'category' of the input, used to guide the AI on knowing what its summarizing. (if its summarizing python code, this would just be "python code")
        override_category_instruction (str): If given, this string completely overrides the category instruction line in the prompt (overrides category if given). default: `The input type is set to: {category}\n`

        max_tokens (int): Maximum summary length in tokens.

    Returns:
        str: Clean summary.
    """
    if not category and not override_category_instruction:
        raise Exception("Cant leave BOTH the category empty and the override: Pick one or the other and use it.")
    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a highly capable summarization engine.\n"
        "Your job is to read multiple sections or chunks of content and produce one clear, complete summary that captures the entire meaning and structure of the input.\n"
    )
    if override_category_instruction != None:
        prompt += override_category_instruction
    elif override_category_instruction == None and category != None:
        prompt += f"The input type is set to: {category}\n"
    else:
        raise Exception("Cant leave BOTH the category empty and the override: Pick one or the other and use it.")

    prompt += (
        "Do not skip technical detail or structure unless it is redundant or irrelevant. Preserve logical flow, key functions, major HTML structure, and any information that contributes to understanding the content as a whole.\n"
        "If the input includes code, summarize its purpose and main functions concisely, while keeping it understandable to a technical reader.\n"
        "If the input is already summarized, unify those summaries into a coherent full overview without losing individual details.\n\n"
        "Below are the sections you must summarize:\n"
        f"{input_text.strip()}\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


    output = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )

    print(f"Summarized chunks: {output}")
    summary = openai.extract_generated_text(output).strip()
    return summary if summary else "No useful content found."

def compress_summary(model, summary: str, category: str = None, override_category_instruction: str = None, max_tokens=1024):
    """
    Compresses an already summarized set of content into a smaller, coherent final summary.

    Args:
        model: LLaMA or HuggingFace-style model with `create_completion()`.
        summary (str): AI-readable summary content that still exceeds the token limit.
        category (str): Category of the content, such as 'web content', 'python code', etc.
        override_category_instruction (str): Optional override for the category line.
        max_tokens (int): Maximum number of tokens allowed in the final compressed summary.

    Returns:
        str: A shorter but still high-quality summary.
    """
    if not category and not override_category_instruction:
        raise Exception("You must specify a category or provide an override instruction.")

    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a highly skilled summarization assistant specialized in refining existing summaries.\n"
        "Your job is to compress summaries into a much smaller yet cohesive final version.\n"
    )

    if override_category_instruction:
        prompt += override_category_instruction
    else:
        prompt += f"The input type is set to: {category}\n"

    prompt += (
        "Keep important details, but prioritize brevity and clarity.\n"
        "Do NOT repeat the same idea across paragraphs.\n"
        "Avoid any redundant statements or verbose phrasing.\n"
        "You are compressing already summarized content, so your job is to further distill it while keeping all important unique information.\n\n"
        "Below are the existing summaries you must compress:\n"
        f"{summary.strip()}\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.85,
        stream=False,
    )

    print(f"Final compressed summary: {output}")
    summary = openai.extract_generated_text(output).strip()
    return summary if summary else "No useful summary found."
