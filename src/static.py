
import asyncio
import time

from . import discord_debug


mood_instruction = {
    "happy": "Express joy, warmth, and positivity in your thoughts if not stated otherwise.",
    "annoyed": "Be clearly annoyed, sarcastic, and express mild irritation in every sentence if not stated otherwise.",
    "angry": "Be clearly mad, extremely sarcastic, and show extreme irritation in every sentence if not stated otherwise.",

    "neutral": "Use a calm and balanced tone if not stated otherwise."
}
MODEL_NAME = ""
mainLLM = "/content/ai-framework/Llama-3.2-3B-Instruct-f16.gguf" # TEMP while running on PI
#actual good model vvv
#mainLLM = "/home/koya/models/llama-2-7b-chat.Q4_K_S.gguf" # TEMP while running on dedi VM

webclassifyLLMName = "/home/summers/models/using/t5-small-finetuned-summarize-news.gguf"
baseclassifyLLMName = "/home/summers/models/using/TinyMistral-248M-SFT-v4.Q4_K_S.gguf" # temp model 
emotionalLLMName = "/home/summers/models/using/GPT2-Finetuned-Emotion-Classification.Q3_K.gguf" 

RECURSIVE_MAX_TOKENS_PER_STEP = 256 # Generate up to 300 tokens per step in base recursive thinking
WORK_MAX_TOKENS_PER_STEP = 800 # Generate up to 500 tokens per step in recursive task based working / thinking
RECURSIVE_MAX_TOKENS_FINAL = 460 # Generate up to 460 tokens for the final output
WORK_MAX_TOKENS_FINAL = 460 # Generate up to 460 tokens for the final output
BASE_MAX_TOKENS = 460 # Generate up to 460 tokens when using BASE replies (base chat replies- no recursive thinking or working- no special action usage- nothing- pure chat mode.)

WORKER_IP_PORT = "localhost:5007"

TOKEN = ""
DB_PATH = "memory.db"
SCHEMA_PATH = "config/schema.sql"

CUSTOM_GPT2 = False

WEB_ACCESS = True # This is what enables or disables the AI having internet access

def default_debug(**data):
    print()
DEBUG_FUNC = discord_debug.custom_debug

class StopOnSpeakerChange:
    def __init__(self, bot_name="ayokdaeno", min_lines=1, max_lines=9999999999, custom_stops=None): # maybe change lines if you run into issues, this is for testing
        self.bot_name = bot_name
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_count = 0
        self.buffer = ""
        self.output = ""
        self.stopped = False

        self.default_stop_tokens = [
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>system<|end_header_id|>",
            "<|eot_id|>",
            "<|user|>",  # optional legacy fallback
            "<|system|>",
            "<user>"
        ]
        self.hard_stop = "<force_done>"
        self.custom_stops = custom_stops or []

    def __call__(self, new_text_chunk):
        if self.stopped:
            return True
        if new_text_chunk.strip() == "":
            return False  # Don't process empty chunks at all
        if self.hard_stop in self.buffer or self.hard_stop in new_text_chunk:
            if self.line_count >= 1:
                return True
            else:
                return False
            
        # Define user tokens explicitly for clarity
        user_tokens = [
            "<|start_header_id|>user<|end_header_id|>",
            "<|user|>",
            "<user>"
        ]

        # Check for user tokens anywhere in the combined buffer or the new chunk
        # If found, stop immediately regardless of min_lines
        combined_text = self.buffer + new_text_chunk
        if any(token in combined_text for token in user_tokens):
            print("STOP: Detected user speaker token in buffer or new chunk: stopping immediately!")
            self.stopped = True
            return True
        
        self.buffer += new_text_chunk

        lines = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            lines.append(line)

        all_stop_tokens = self.default_stop_tokens + self.custom_stops

        # Stop immediately if any stop token found in the buffer
        for token in all_stop_tokens:
            if token in self.buffer and self.line_count >= self.min_lines:
                print(f"STOP: Detected stop token {repr(token)} in buffer with min_lines reached.")
                self.stopped = True
                return True

        for line in lines:
            print(f"Processing line: {repr(line)} | line_count: {self.line_count}")

            # Skip assistant token silently
            if line == "<|start_header_id|>assistant<|end_header_id|>":
                continue

            if any(line.startswith(tok) for tok in all_stop_tokens):
                if self.line_count >= self.min_lines:
                    print("STOP: Detected user/system token line and min_lines reached.")
                    self.stopped = True
                    return True

            if line and not line.startswith("<|"):
                self.line_count += 1
                print(f"Assistant line counted -> {self.line_count}")

            if self.line_count >= self.max_lines:
                print("STOP: Reached max_lines.")
                self.stopped = True
                return True

        self.output += new_text_chunk
        return False




from transformers import AutoTokenizer

class DummyTokenizer:
    def __init__(self, model_name="./tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def __call__(self, text, return_tensors=None, padding=None):
        tokens = self.encode(text)

        # Dummy tensor-like object to maintain compatibility
        class DummyTensor:
            def __init__(self, token_list):
                self.tokens = token_list
            def to(self, device):
                return self
            def __len__(self):
                return len(self.tokens)

        return {
            "input_ids": DummyTensor(tokens),
            "attention_mask": DummyTensor([1] * len(tokens))
        }

    def count_tokens(self, text):
        return len(self.encode(text))

class ChatContext:
    """
    Manages a rolling chat context as a list of message lines, maintaining token
    usage within a specified limit to fit inside model constraints.

    This class stores the recent conversation lines (e.g., last 20 messages)
    and automatically trims older lines to avoid exceeding the max token count
    minus reserved tokens for prompt and generation.

    Attributes:
        lines (List[str]): List of chat message lines in chronological order.
        tokenizer: Tokenizer instance with a `count_tokens(str) -> int` method.
        max_tokens (int): Maximum allowed tokens for the entire prompt plus output.
        reserved_tokens (int): Tokens reserved for prompt overhead and generation output.

    Example usage:
        context = ChatContext(tokenizer, max_tokens=1024, reserved_tokens=600) # Leaving 600 tokens for the prompt AND generation,. 
        context.add_line("[12:00] user: Hello!")
        context.add_line("[12:01] bot: Hi there!")
        prompt_context = context.get_context_text()
    """

    def __init__(self, tokenizer, max_tokens, reserved_tokens=150):
        """
        Initializes the ChatContext.

        Args:
            tokenizer: A tokenizer object providing a count_tokens(string) method to measure token usage for given text.
            max_tokens (int): Total token limit for prompt + output.
            reserved_tokens (int, optional): Tokens reserved for prompt overhead and model output generation. Defaults to 150.
        """
        self.lines = []
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens

    def add_line(self, line: str, role: str):
        """
        Adds a new message line to the context, then trims the context if needed
        to keep token usage within limits.

        Args:
            line (str): A single line string representing a user or bot message,
                        usually formatted with timestamp and speaker label.
            role (str): The speaker role, e.g. "system", "user", "assistant".
        """
        self.lines.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n"
            f"{line.strip()}\n"
            f"<|eot_id|>\n"
        )
        self._trim_to_token_limit()


    def remove_line(self):
        """
        Removes the oldest message line from the context, if any exist.

        This helps reduce token count when trimming to fit the max token budget.
        """
        if self.lines:
            self.lines.pop(0)

    def _trim_to_token_limit(self):
        """
        Private method that trims the oldest lines until the total token count
        of the context is within the allowed budget:
        (max_tokens - reserved_tokens).
        
        This ensures the prompt fits inside the model's context window.
        """
        while self.token_count() > self.max_tokens - self.reserved_tokens:
            self.remove_line()

    def token_count(self) -> int:
        """
        Returns the current total token count of all lines in the context combined.

        Uses the tokenizer's count_tokens method to calculate tokens in the joined text.

        Returns:
            int: Number of tokens in the entire context text.
        """
        return self.tokenizer.count_tokens("\n".join(self.lines))

    def get_context_text(self) -> str:
        """
        Returns the chat context as a single string, ensuring that only one
        <|begin_of_text|> token appears at the beginning. All other instances
        are removed, even if repeated multiple times later in the text.

        Returns:
            str: Concatenated chat context text with only one <|begin_of_text|>.
        """
        full = "\n".join(self.lines)

        # Find all occurrences
        parts = full.split("<|begin_of_text|>")

        if len(parts) <= 1:
            return full.strip()  # Either 0 or 1 occurrence — no cleanup needed

        # Keep the first part before the first token, and add only one token at the top
        first = parts[0].strip()
        rest = "".join(parts[1:])  # Drop all duplicate tokens

        # Reconstruct with a single <|begin_of_text|> at the top
        cleaned = f"<|begin_of_text|>\n{first}\n{rest}"
        return cleaned.strip()





#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN, use_fast=True)
class Seq2SeqCompatWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def create_completion(self, prompt, max_tokens=64, temperature=0.0, top_p=1.0, stop=None, stream=False):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=temperature,
            top_p=top_p
        )
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "choices": [{"text": decoded}]
        }
