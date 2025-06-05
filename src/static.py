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

from transformers import StoppingCriteria

class StopOnSpeakerChange(StoppingCriteria):
    def __init__(self, tokenizer, bot_name="ayokdaeno", min_lines=1, max_lines=20):
        self.tokenizer = tokenizer
        self.bot_name = bot_name
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        lines = decoded.splitlines()

        assistant_lines = []
        recording = False

        for line in lines:
            line = line.strip()

            if line.startswith("<|assistant|>"):
                recording = True
                assistant_lines = []
                continue
            elif line.startswith("<|user|>"):
                recording = False
                continue

            if recording and line and not line.startswith("<|") and not line.startswith("```"):
                assistant_lines.append(line)

        self.line_count = len(assistant_lines)

        if self.line_count >= self.max_lines:
            return True

        last_line = lines[-1].strip() if lines else ""
        speaker_change = (
            (last_line.endswith(":") and not last_line.lower().startswith(self.bot_name.lower()))
            or last_line.startswith("<|user|>")
        )

        # 🛠 Prevent stopping if assistant has not said *anything meaningful* yet
        if self.line_count == 0:
            return False

        if self.line_count >= self.min_lines and speaker_change:
            return True

        if self.line_count >= self.min_lines:
            return True

        return False



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN, use_fast=True)
