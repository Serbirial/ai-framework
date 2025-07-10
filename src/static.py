
import asyncio
import time

mood_instruction = {
    "happy": "Express joy, warmth, and positivity in your thoughts if not stated otherwise.",
    "annoyed": "Be clearly annoyed, sarcastic, and express mild irritation in every sentence if not stated otherwise.",
    "angry": "Be clearly mad, extremely sarcastic, and show extreme irritation in every sentence if not stated otherwise.",

    "neutral": "Use a calm and balanced tone if not stated otherwise."
}
MODEL_NAME = ""
mainLLM = "/home/summers/models/using/maybe/SmolLM2-360M-Instruct-Q4_K_M.gguf" # TEMP while running on PI
#actual good model vvv
#mainLLM = "/home/koya/models/llama-2-7b-chat.Q4_K_S.gguf" # TEMP while running on dedi VM

webclassifyLLMName = "/home/summers/models/using/t5-small-finetuned-summarize-news.gguf"
baseclassifyLLMName = "/home/summers/models/using/TinyMistral-248M-SFT-v4.Q4_K_S.gguf" # temp model 
emotionalLLMName = "/home/summers/models/using/GPT2-Finetuned-Emotion-Classification.Q3_K.gguf" 

#llama = Llama(model_path=MODEL_PATH, n_batch=8, n_threads=4, n_gpu_layers=0, low_vram=True)
# If this uncomments itself ONE MORE TIME im gonna nuke somneone

WORKER_IP_PORT = "localhost:5007"

TOKEN = ""
DB_PATH = "memory.db"
SCHEMA_PATH = "config/schema.sql"

CUSTOM_GPT2 = True

WEB_ACCESS = False # This is what enables or disables the AI having internet access

def default_debug(**data):
    print()

DEBUG_FUNC = default_debug

class StopOnSpeakerChange:
    def __init__(self, bot_name="ayokdaeno", min_lines=1, max_lines=20, custom_stops=None):
        self.bot_name = bot_name
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_count = 0
        self.buffer = ""
        self.stopped = False

        self.default_stop_tokens = ["<|user|>", "<|system|>", "<|end|>", "<|eos|>", "<user>"]
        self.custom_stops = custom_stops or []

    def __call__(self, new_text_chunk):
        if self.stopped:
            return True

        self.buffer += new_text_chunk

        lines = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            lines.append(line)

        all_stop_tokens = self.default_stop_tokens + self.custom_stops

        for line in lines:
            print(f"Processing line: {repr(line)} | line_count: {self.line_count}")

            if line == "<|assistant|>":
                continue

            if any(line.startswith(tok) for tok in all_stop_tokens):
                if self.line_count >= self.min_lines:
                    print("STOP: Detected token in line and min_lines reached.")
                    self.stopped = True
                    return True

            if line and not line.startswith("<|"):
                self.line_count += 1
                print(f"Assistant line counted → {self.line_count}")

            if self.line_count >= self.max_lines:
                print("STOP: Reached max_lines.")
                self.stopped = True
                return True

        return False


    
class DiscordTextStreamer:
    def __init__(self, discord_message, update_interval=5.0):
        """
        discord_message: a discord.Message or discord.InteractionResponse object with an edit() coroutine method.
        update_interval: seconds between edits.
        """
        self.message = discord_message
        self.update_interval = update_interval
        self.buffer = ""
        self.last_update_time = 0
        self._lock = asyncio.Lock()
        self._task = None

    async def _periodic_update(self):
        while True:
            await asyncio.sleep(self.update_interval)
            async with self._lock:
                if self.buffer:
                    try:
                        await self.message.edit(content=self.buffer)
                    except Exception as e:
                        print(f"Error editing Discord message: {e}")
                    self.buffer = ""
                    self.last_update_time = time.time()

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._periodic_update())

    async def update(self, text_chunk):
        async with self._lock:
            self.buffer += text_chunk

        # Start the periodic task if not started
        self.start()

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

class AssistantOnlyFilter: # this filters speaker changes- and special tokens
    def __init__(self, assistant_token="<|assistant|>", other_tokens=None):
        self.assistant_token = assistant_token
        self.other_tokens = other_tokens or ["<|user|>", "<|system|>", "<|end|>", "<user>"]

        self.buffer = ""
        self.filtered_output = ""
        self.saw_any_speaker_change = False  # Flag for first detected speaker change
        self.in_assistant_mode = True        # Start accumulating regardless until speaker change

    def __call__(self, new_text_chunk):
        self.buffer += new_text_chunk

        lines = self.buffer.split("\n")
        self.buffer = lines.pop()  # Save the last (possibly incomplete) line for next chunk

        for line in lines:
            stripped = line.strip()

            # Detect non-assistant speaker tokens
            if stripped in self.other_tokens:
                self.saw_any_speaker_change = True
                self.in_assistant_mode = False
                continue  # Skip speaker tokens entirely

            # Detect assistant token
            if stripped == self.assistant_token:
                self.saw_any_speaker_change = True
                self.in_assistant_mode = True
                continue  # Skip speaker tokens entirely

            # Accumulate only assistant-mode lines, but skip speaker tokens
            if (not self.saw_any_speaker_change) or (self.in_assistant_mode and stripped != ""):
                self.filtered_output += line + "\n"

        return False  # Never stop generation

    def get_filtered_output(self):
        leftover = self.buffer.strip()

        # Don't add assistant or other speaker tokens accidentally
        if leftover in self.other_tokens + [self.assistant_token]:
            return self.filtered_output

        # Only add leftover if it's valid assistant output
        if (not self.saw_any_speaker_change) or (self.in_assistant_mode and leftover):
            return self.filtered_output + leftover + "\n"

        return self.filtered_output





from transformers import AutoTokenizer

class DummyTokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        """
        self.lines.append(f"<{role}> {line}")
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
        Returns the entire chat context as a single string, with each line
        separated by a newline character.

        This is ready to be inserted into the prompt.

        Returns:
            str: Concatenated chat context text.
        """
        return "\n".join(self.lines)



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
