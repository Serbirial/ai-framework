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
    def __init__(self, tokenizer, bot_name="ayokdaeno", min_lines=1, max_lines=20):
        self.tokenizer = tokenizer
        self.bot_name = bot_name
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_count = 0
        self.prev_len = 0
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        current_len = input_ids.shape[1]
        new_tokens = input_ids[0, self.prev_len:current_len]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        self.prev_len = current_len

        self.buffer += decoded

        # Split only if newline suggests a line end
        lines = []
        while True:
            if "\n" in self.buffer:
                part, self.buffer = self.buffer.split("\n", 1)
                data = part.strip()
                if data != "":
                    lines.append(data)
                # else: skip blank lines silently
            else:
                break  # wait for more text before splitting


        assistant_lines = []

        for line in lines:
            if line == "<|assistant|>":
                continue
            elif line.startswith("<|user|>") or line.startswith("<|system|>"):
                return self.line_count >= self.min_lines
            if line and not line.startswith("<|"):
                assistant_lines.append(line)

        self.line_count += len(assistant_lines)

        if self.line_count >= self.max_lines:
            return True

        return False





tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN, use_fast=True)
