from transformers import StoppingCriteria
from transformers import AutoTokenizer

mood_instruction = {
    "happy": "Express joy, warmth, and positivity in your thoughts.",
    "annoyed": "Be clearly annoyed, sarcastic, and express mild irritation in every sentence.",
    "angry": "Be clearly mad, extremely sarcastic, and show extreme irritation in every sentence.",

    "neutral": "Speak in a calm and balanced tone."
}
MODEL_NAME = "stabilityai/stablelm-2-1_6b-chat"
TOKEN = ""
MEMORY_FILE = "memory.json"

class StopOnSpeakerChange(StoppingCriteria):
    def __init__(self, tokenizer, bot_name="ayokdaeno", min_lines=1, max_lines=5):
        self.tokenizer = tokenizer
        self.bot_name = bot_name
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        # Decode entire sequence so far
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        lines = decoded.splitlines()

        assistant_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("<|user|>") or line.startswith("<|assistant|>"):
                assistant_lines = []  # reset count per new assistant block
            elif line:
                assistant_lines.append(line)

        self.line_count = len(assistant_lines)

        # Force stop at max_lines, regardless of speaker
        if self.line_count >= self.max_lines:
            return True

        # Detect speaker change
        last_line = lines[-1].strip() if lines else ""
        speaker_change = (
            (last_line.endswith(":") and not last_line.startswith(self.bot_name)) or
            last_line.startswith("<|user|>") or
            last_line.startswith("<|assistant|>")
        )

        # Allow stop if we're past min_lines AND speaker changed
        if self.line_count >= self.min_lines and speaker_change:
            return True

        return False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN, use_fast=True)
