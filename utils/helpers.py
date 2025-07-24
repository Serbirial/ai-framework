import sqlite3
from src.static import DummyTokenizer, DB_PATH


def get_mem_tokens_n(identifier, limit):
    tokenizer = DummyTokenizer()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT data FROM MEMORY
        WHERE userid = ?
        ORDER BY timestamp DESC
    """, (identifier,))

    rows = cursor.fetchall()
    total_tokens = 0

    for row in rows:
        entry = row[0]
        tokens = tokenizer.count_tokens(entry)
        if total_tokens + len(tokens) > limit:
            break
        total_tokens += tokens

    conn.close()
    return total_tokens



def pick_ctx_size(available_mb: int, model_size_mb: int) -> int:
    """
    Estimate the maximum safe ctx_size based on available RAM.
    """
    kv_per_token_mb = 0.0028  # ~2.8KB per token
    buffer = 150              # Overhead (OS + llama.cpp buffer)
    usable = available_mb - model_size_mb - buffer
    if usable <= 0:
        return 512

    tokens = int(usable / kv_per_token_mb)
    tokens = min(tokens, 2048)
    tokens = max(tokens, 512)
    return tokens - (tokens % 64)  # round down to nearest multiple of 64


class DummyTokenizer: # TODO: remove and use static.dummytokenizer
    eos_token_id = 0
    eos_token = ""

    def encode(self, text):
        # Just split on whitespace, approx tokens
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, list):
            return " ".join(tokens)
        return str(tokens)

    def __call__(self, text, return_tensors=None, padding=None):
        tokens = self.encode(text)
        # Return dummy tensor-like object to avoid breaking code
        class DummyTensor:
            def to(self, device):
                return self
            def __len__(self):
                return len(tokens)
        return {"input_ids": DummyTensor(), "attention_mask": DummyTensor()}

    def count_tokens(self, text):
        return len(self.encode(text))


def trim_context_to_fit(base_prompt, context_lines, max_ctx, reserved_for_output=400):
    """
    Trims a list of context lines to fit within the maximum token context window,
    leaving space for both the base prompt and the expected model output.

    This function ensures that only the most recent context lines that can fit within
    the remaining allowed tokens are included, preserving their original order.

    Args:
        base_prompt (str): The full system/user prompt that precedes the context.
        context_lines (List[str]): List of context/history lines (e.g., conversation log).
        max_ctx (int): Maximum token context size the model supports.
        reserved_for_output (int, optional): Tokens to reserve for model's response generation.
            Defaults to 400.

    Returns:
        str: A newline-joined string of the most recent context lines that fit within the token budget.
    """

    tokenizer = DummyTokenizer()
    base_tokens = tokenizer.count_tokens(base_prompt)
    remaining_tokens = max_ctx - reserved_for_output - base_tokens

    trimmed_context = []
    total_context_tokens = 0

    # Add newer lines first (keep most recent)
    for line in reversed(context_lines):
        line_tokens = tokenizer.count_tokens(line)
        if total_context_tokens + line_tokens > remaining_tokens:
            break
        trimmed_context.insert(0, line)  # Maintain original order
        total_context_tokens += line_tokens

    return "\n".join(trimmed_context)
