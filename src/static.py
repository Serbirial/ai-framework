mood_instruction = {
    "happy": "Express joy, warmth, and positivity in your thoughts.",
    "annoyed": "Be clearly annoyed, sarcastic, and express mild irritation in every sentence.",
    "angry": "Be clearly mad, extremely sarcastic, and show extreme irritation in every sentence.",

    "neutral": "Speak in a calm and balanced tone."
}

class StopOnSpeakerChange(StoppingCriteria):
    def __init__(self, tokenizer, bot_name="ayokdaeno"):
        self.tokenizer = tokenizer
        self.bot_name = bot_name

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=False)
        lines = decoded.splitlines()
        if not lines:
            return False
        last_line = lines[-1].strip()
        # Add check for special speaker tokens
        if last_line.endswith(":") and not last_line.startswith(self.bot_name):
            return True
        if last_line.startswith("<|user|>") or last_line.startswith("<|assistant|>"):
            return True
        return False

