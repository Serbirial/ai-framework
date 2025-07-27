from utils.tokens import get_token_availability
from .agent import AgentInstance
from utils import openai

def check_if_overflowing(total_tokens: int, token_window: int, buffer: int):
    """Checks if the total tokens is overflowing outside of the token window

    Args:
        total_tokens (int): Total prompt tokens
        token_window (int): The maximum allowed tokens

    """
    available_tokens, overflow = get_token_availability(
        current_prompt_tokens=total_tokens,
        other_tokens=0,
        total_token_window=token_window,
        reserved_buffer=buffer,
    )
    if overflow > 0:
        return True
    else:
        return False

def check_if_would_overflow(total_tokens: int, step_generation_tokens: int, token_window: int, buffer: int):
    """Checks if the next step would cause an overflow of the token window

    Args:
        total_tokens (int): Total prompt tokens
        step_generation_tokens (int): How many tokens the model would be told to generate 
        token_window (int): The maximum allowed tokens

    """
    available_tokens, overflow = get_token_availability(
        current_prompt_tokens=total_tokens,
        other_tokens=step_generation_tokens,
        total_token_window=token_window,
        reserved_buffer=buffer,
    )
    if overflow > 0:
        return True
    else:
        return False


class StateManager:
    """Can be used to manage state in recursive workers, mostly to be used when the token limit is hit before fully done. 
    """
    def __init__(self, model, bot: AgentInstance, category: str):
        """
        bot: the high-level ChatBot interface.
        category: a valid category. (not checked if valid)
        """
        self.model = model
        self.bot = bot
        self.category = category

    def save_state(self, progress_list: list[str], max_new_tokens: int) -> str:
        """
        Uses the model to generate a summary message that encapsulates current progress.
        The model will phrase this in a way that helps it resume seamlessly later.
        """

        if not progress_list:
            return "There's no progress to save yet — the task hasn't started."

        # Build input to send to model
        joined_progress = "\n".join(f"{i+1}. {s.strip()}" for i, s in enumerate(progress_list) if s.strip())

        prompt = f"""
You are {self.bot.name}, and you have been working through something internally. Below is your progress so far. You are not finished yet.

Your goal is to generate a message to the user that:
1. Explains the task is still in progress.
2. Recaps what has been accomplished so far.
3. Makes it easy for you (the assistant) to pick up where you left off by re-reading this message later.

Format the message as a friendly summary from the assistant, **not** a technical dump.
Avoid repeating every step verbatim — instead, summarize the overall direction, approach, and key insights so far.

--- **PROGRESS START** ---
{joined_progress}
--- **PROGRESS END** ---

Now, generate a message for the user that encapsulates this progress and invites them to continue when ready.
"""

        result = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,      
            top_p=0.9,           
            stream=False,
        )
        result = openai.extract_generated_text(result)

        return result.strip()

    def resume(self, username, state_message, identifier, tier, max_new_tokens=None, temperature=0.7, top_p=0.9, context=None, debug=False, streamer=None, force_recursive=False, recursive_depth=3, tiny_mode=False, cnn_file_path=None):
        """
        Resumes the conversation using the stored category so it re-enters the correct thinker route.
        """
        return self.bot.chat(
            username=username,
            user_input=state_message,
            identifier=identifier,
            tier=tier,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            context=context,
            debug=debug,
            streamer=streamer,
            force_recursive=force_recursive,
            recursive_depth=recursive_depth,
            category_override=self.category,  # this is key to re-routing to same thinker
            tiny_mode=tiny_mode,
            cnn_file_path=cnn_file_path
        )
