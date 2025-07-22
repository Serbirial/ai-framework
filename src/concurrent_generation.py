import time, re
from llama_cpp import Llama
from src.static import Config

config = Config()

class Concurrent_Llama_Gen:
	def __init__(self, model_path: str, max_concurrent: int = 3, n_ctx: int = None):
		if not n_ctx:
			n_ctx = config.token_config["t0"]["BASE_MAX_TOKENS"] * max_concurrent
		
		self.max_concurrent = max_concurrent
		self.generations = {
			i: {
				"prompt": None,
				"identifier": None,
				"output": "",
				"is_ready": False,
				"max_token_generation": 0,
				"model": Llama(
					model_path=model_path,
					n_ctx=n_ctx,
					n_threads=2,
					n_batch=64,
					n_gpu_layers=0,
					use_mlock=False,
					logits_all=False,
					verbose=False,
				),
				"token_count": 0  
			}
			for i in range(max_concurrent)
		}

	def has_free_slot(self) -> bool:
		return any(slot["identifier"] is None for slot in self.generations.values())

	def assign(self, identifier: str, full_prompt: str, user_input: str,
			   token_limit: int = config.token_config["t0"]["BASE_MAX_TOKENS"]) -> int | None:
		"""Assign a user slot, set token_limit per‑user."""
		for i, slot in self.generations.items():
			if slot["identifier"] is None:
				self.generations[i] = {
					"prompt": full_prompt,
					"identifier": identifier,
					"input": user_input,
					"output": "",
					"is_ready": False,
					"token_limit": token_limit,
					"token_count": 0,
				}
				return i
		return None

	def remove(self, slot: int):
		self.generations[slot] = {
			"prompt": None,
			"identifier": None,
			"input": None,
			"output": "",
			"is_ready": False,
			"token_limit": 0,
			"token_count": 0,
		}

	def build_prompt(self) -> str:
		blocks = []
		for i, gen in self.generations.items():
			if gen["identifier"] is None:
				continue
			block = (
				f"[User {i}]\n{gen['prompt']}\n"
				f"<|user|>\n{gen['input']}\n"
				f"<|assistant|>\n{gen['output']}"
			)
			blocks.append(block)
		return "\n\n".join(blocks)

	def parse_output(self, raw_text: str):
		"""Append tokens & mark ready when token_limit reached."""
		splits = raw_text.split("[User ")
		for section in splits[1:]:
			try:
				idx, rest = section.split("]", 1)
				slot = int(idx.strip())
				if slot not in self.generations:
					continue
				gen = self.generations[slot]

				new_text = rest.strip()
				if not new_text:
					continue

				gen["output"] += new_text
				# rough token count via whitespace‑split
				gen["token_count"] += len(re.findall(r"\S+", new_text))

				if gen["token_count"] >= gen["token_limit"]:
					gen["is_ready"] = True
			except ValueError:
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
