from llama_cpp import Llama


mood_instruction = {
    "happy": "Express joy, warmth, and positivity in your thoughts.",
    "annoyed": "Be clearly annoyed, sarcastic, and express mild irritation in every sentence.",
    "angry": "Be clearly mad, extremely sarcastic, and show extreme irritation in every sentence.",

    "neutral": "Speak in a calm and balanced tone."
}
MODEL_NAME = ""
MODEL_PATH = "/home/summers/models/mistral/mistral-nemo-3b-unhealed-q3_k_m.gguf" # new
llama = Llama(model_path=MODEL_PATH, n_batch=8, n_threads=4, n_gpu_layers=0, low_vram=True)

TOKEN = ""
MEMORY_FILE = "memory.json"

class StopOnSpeakerChange:
    def __init__(self, bot_name="ayokdaeno", min_lines=1, max_lines=20):
        self.bot_name = bot_name
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_count = 0
        self.buffer = ""

    def __call__(self, new_text_chunk):
        self.buffer += new_text_chunk
        lines = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line != "":
                lines.append(line)

        assistant_lines = []
        for line in lines:
            if line == "<|assistant|>":
                continue
            elif line.startswith("<|user|>") or line.startswith("<|system|>"):
                if self.line_count >= self.min_lines:
                    return True
            if line and not line.startswith("<|"):
                assistant_lines.append(line)

        self.line_count += len(assistant_lines)

        if self.line_count >= self.max_lines:
            return True

        return False


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

    def add_line(self, line: str):
        """
        Adds a new message line to the context, then trims the context if needed
        to keep token usage within limits.

        Args:
            line (str): A single line string representing a user or bot message,
                        usually formatted with timestamp and speaker label.
        """
        self.lines.append(line)
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
