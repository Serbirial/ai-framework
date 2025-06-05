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

            # Start recording after <|assistant|> marker
            if line.startswith("<|assistant|>"):
                recording = True
                assistant_lines = []
                continue
            elif line.startswith("<|user|>"):
                recording = False
                continue

            # Only track assistant lines, ignore speaker markers or empty lines
            if recording and line and not line.startswith("<|") and not line.startswith("```"):
                assistant_lines.append(line)

        self.line_count = len(assistant_lines)

        # Check for double newline (used as a soft logical stopping point)
        double_newline = "\n\n" in decoded

        # Enforce hard maximum
        if self.line_count >= self.max_lines:
            return True

        # Look at last non-empty line to check for speaker switch
        last_line = lines[-1].strip() if lines else ""
        speaker_change = (
            (last_line.endswith(":") and not last_line.lower().startswith(self.bot_name.lower()))
            or last_line.startswith("<|user|>")
        )

        # Stop if speaker change AND enough lines were generated
        if self.line_count >= self.min_lines and speaker_change:
            return True

        # Optional: stop softly if double newline and enough content
        if self.line_count >= self.min_lines and double_newline:
            return True

        return False


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN, use_fast=True)
