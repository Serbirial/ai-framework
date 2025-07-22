import time
from llama_cpp import Llama

class Concurrent_Llama_Gen:
    """_summary_
    """
    def __init__(self, model_path: str, max_concurrent: int = 3, n_ctx: int = 16384): # 4096 * 3  
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=6,
            n_batch=64,
            use_mlock=False, # Low prio
            logits_all=False,
            verbose=False,
        )
        self.max_concurrent = max_concurrent

        self.generations = {
            i: {
                "prompt": None,        # Full prompt
                "identifier": None,    # UID
                "input": None,         # input
                "output": None,        # output
                "is_ready": False      # True when done
            }
            for i in range(max_concurrent)
        }

    def has_free_slot(self) -> bool:
        return any(slot["identifier"] is None for slot in self.generations.values())

    def assign(self, identifier: str, full_prompt: str, user_input: str) -> int | None:
        """Assigns a user to a slot with full prompt + message input."""
        for i, slot in self.generations.items():
            if slot["identifier"] is None:
                self.generations[i] = {
                    "prompt": full_prompt,
                    "identifier": identifier,
                    "input": user_input,
                    "output": "",
                    "is_ready": False
                }
                return i
        return None

    def remove(self, slot: int):
        self.generations[slot] = {
            "prompt": None,
            "identifier": None,
            "input": None,
            "output": "",
            "is_ready": False
        }

    def build_prompt(self) -> str:
        """Combine prompts for all active users, clearly delimited."""
        blocks = []
        for i, gen in self.generations.items():
            if gen["identifier"] is None:
                continue
            block = f"[User {i}]\n{gen['prompt']}\n<|user|>\n{gen['input']}\n<|assistant|>\n{gen['output']}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def parse_output(self, raw_text: str):
        """Parse model output into correct slot based on [User X] header."""
        splits = raw_text.split("[User ")
        for section in splits[1:]:
            try:
                idx, rest = section.split("]", 1)
                slot = int(idx.strip())
                if slot in self.generations:
                    self.generations[slot]["output"] += rest.strip()
                    self.generations[slot]["is_ready"] = True
            except:
                continue

    def count_active(self):
        return sum(1 for g in self.generations.values() if g["identifier"] is not None)

    def get_output(self, slot: int) -> str:
        return self.generations[slot]["output"]

    def get_all_outputs(self) -> dict[int, str]:
        return {
            i: v["output"]
            for i, v in self.generations.items()
            if v["identifier"] is not None
        }

    async def tick(self, tokens: int = 12):
        """Generate a few tokens for all users in a single call."""
        if self.count_active() == 0:
            return

        prompt = self.build_prompt()
        if not prompt:
            return

        result = self.model(
            prompt,
            max_tokens=tokens,
            stop=["[User"],
            temperature=0.6,
            top_p=0.9,
        )["choices"][0]["text"]

        self.parse_output(result)