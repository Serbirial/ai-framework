
def get_token_availability(
    current_prompt_tokens: int,
    other_tokens: int,
    total_token_window: int,
    reserved_buffer: int = 512,
):
    """
    Calculates available space in the token window and overflow amount.

    Args:
        current_prompt_tokens: Tokens used by the current prompt (the ENTIRE prompt from start to finish).
        other_tokens: Any non-prompt tokens (e.g: tool responses or internal messages / data that isnt reliably held in the prompt).
        total_token_window: Model's full token window.
        reserved_buffer: Headroom buffer to leave unused, allowing for less cases where a tool returns lots of tokens and overflows the entire token window (default 512- raise if expectingh heavy tool usage).

    Returns:
        available_tokens: How many tokens are still available.
        overflow: How many tokens over budget (0 if not over).
    """
    allowed_tokens = total_token_window - reserved_buffer
    if other_tokens > 0:
        total_needed = current_prompt_tokens + other_tokens

    overflow = max(0, total_needed - allowed_tokens)
    available_tokens = max(0, allowed_tokens - current_prompt_tokens)

    return available_tokens, overflow

def find_compression_ratio(available_tokens, raw_data_tokens, min_ratio=0.5, max_ratio=0.80):
    """
    Dynamically calculates a compression ratio to fit data within available token space.

    Args:
        available_tokens (int): Remaining space in the token window.
        raw_data_tokens (int): Actual token count of the data you're compressing.
        min_ratio (float): Lower bound to prevent excessive compression.
        max_ratio (float): Upper bound to prevent undercompression.

    Returns:
        float: Clamped compression ratio.
    """
    if raw_data_tokens == 0:
        return max_ratio

    ratio = available_tokens / raw_data_tokens
    return max(min_ratio, min(max_ratio, ratio))
