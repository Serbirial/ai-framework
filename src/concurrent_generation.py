import asyncio, re
from llama_cpp import Llama
from src.static import Config

from src.bot import ChatBot

config = Config()


class Concurrent_Llama_Gen:
	"""
	Spins up N independent llama.cpp models (one per slot) and runs an isolated instance of ChatBot for concurrency.
	"""

	def __init__(self, model_path: str, botname: object, max_concurrent: int = 2, n_ctx: int | None = None):
		if not n_ctx:
			self.n_ctx = config.token_config["t0"]["BASE_MAX_TOKENS"] * max_concurrent

		self.max_concurrent = max_concurrent
		self.generations: dict[int, dict] = {
			i: {
				"in_use": None,
				"model": ChatBot(name=botname, model=Llama()),
			}
			for i in range(max_concurrent)
		}

	def has_free_slot(self) -> bool:
		return any(v["identifier"] is None for v in self.generations.values())

	def assign(self, identifier):
		"""Attach a user to the first free slot for a generation."""
		if token_limit is None:
			token_limit = config.token_config["t0"]["BASE_MAX_TOKENS"]

		for slot_id, slot in self.generations.items():
			if slot["in_use"] == False:
				slot["identifier"] = identifier
				return slot_id
		return None

	def remove(self, slot: int):
		"""Free a slot (but keep its pre‑instantiated model object)."""
		model = self.generations[slot]["model"]
		self.generations[slot].update(
			{
				"identifier": None,
				"model": model,
			}
		)

	def count_active(self) -> int:
		return sum(1 for g in self.generations.values() if g["identifier"] is not None)

	def chat(self, slot: int, username, user_input, identifier, tier, max_new_tokens=None, temperature=0.7, top_p=0.9, context = None, debug=False, streamer = None, force_recursive=False, recursive_depth=3, category_override=None, tiny_mode=False, cnn_file_path=None):
		return self.generations[slot]["model"].chat(  
                            max_new_tokens=max_new_tokens,
							username=username,
							top_p=top_p,
							user_input=user_input,
							temperature=temperature,
							streamer=streamer,
							recursive_depth=recursive_depth,
							identifier=identifier,
							category_override=category_override,
							tiny_mode=tiny_mode,
							context=context,
							debug=debug,
							tier=tier,
							cnn_file_path=cnn_file_path)