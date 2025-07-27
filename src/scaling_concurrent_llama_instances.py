import asyncio, re
from llama_cpp import Llama
from src.static import Config
from multiprocessing import Process, Pipe
import psutil
import os


from src.bot import AgentInstance

config = Config()
def _model_worker(conn, model_path, botname, core_ids, n_threads):
	import psutil, os
	from multiprocessing.connection import Connection
	psutil.Process(os.getpid()).cpu_affinity(core_ids)

	model = AgentInstance(name=botname, model=Llama(
		model_path=model_path,
		n_ctx=8096,
		n_threads=n_threads,
		use_mlock=False,
		logits_all=False,
		verbose=False,
		use_mmap=False,
		n_gpu_layers=0,
		low_vram=True,
		n_batch=64
	))

	try:
		msg = conn.recv()

		def streamer(text):
			conn.send({"token": text})

		result = model.chat(**msg, streamer=streamer)
		conn.send({"done": True, "final": result})

	except Exception as e:
		conn.send({"error": str(e)})
	finally:
		conn.close()



class Concurrent_Llama_Gen:
	"""
	Spins up N independent llama.cpp models (one per slot) and runs an isolated instance of ChatBot for concurrency.
	"""

	def __init__(self, model_path: str, botname: str, max_concurrent: int = 2, n_ctx: int | None = None):

		self.max_concurrent = max_concurrent
		self.generations: dict[int, dict] = {}

		for i in range(self.max_concurrent):
			parent_conn, child_conn = Pipe()
			proc = Process(
				target=_model_worker,
				args=(child_conn, model_path, botname, [i*2, i*2+1], 3)
			)
			proc.start()
			self.generations[i] = {
				"in_use": False,
				"identifier": None,
				"pipe": parent_conn,
				"proc": proc,
				"output": "",
				"done": False
			}


	def has_free_slot(self) -> bool:
		return any(v["identifier"] is None for v in self.generations.values())

	def assign(self, identifier):
		"""Attach a user to the first free slot for a generation."""
		for slot_id, slot in self.generations.items():
			if slot["in_use"] == False:
				slot["identifier"] = identifier
				return slot_id
		return None

	def remove(self, slot: int):
		"""Free a slot (but keep its pre‑instantiated model object)."""
		self.generations[slot].update(
			{
				"in_use": False,
				"identifier": None,
				"output": "",
				"done": False
			}
		)

	def check_for_identifier(self, identifier):
		for slot in self.generations.keys():
			if self.generations[slot]["identifier"] == identifier:
				return True
		return None

	def count_active(self) -> int:
		return sum(1 for g in self.generations.values() if g["in_use"] is not False)

	def chat(self, slot: int, username, user_input, identifier, tier, max_new_tokens=None, temperature=0.7, top_p=0.9,
			context=None, debug=False, streamer=None, force_recursive=False, recursive_depth=3,
			category_override=None, tiny_mode=False, cnn_file_path=None):
		conn = self.generations[slot]["pipe"]
		data = {
			"max_new_tokens": max_new_tokens,
			"username": username,
			"top_p": top_p,
			"user_input": user_input,
			"temperature": temperature,
			"recursive_depth": recursive_depth,
			"identifier": identifier,
			"category_override": category_override,
			"tiny_mode": tiny_mode,
			"context": context,
			"debug": debug,
			"tier": tier,
			"cnn_file_path": cnn_file_path,
			"force_recursive": force_recursive
		}
		conn.send(data)

		self.generations[slot]["output"] = ""
		self.generations[slot]["done"] = False

		while True:
			if conn.poll(0.05):
				msg = conn.recv()
				if "token" in msg:
					self.generations[slot]["output"] += msg["token"]
					if streamer != None:
						streamer(msg["token"])
				if "done" in msg:
					self.generations[slot]["done"] = True
					break
				if "error" in msg:
					self.generations[slot]["done"] = True
					self.generations[slot]["output"] += f"[ERROR: {msg['error']}]"
					break
		return self.generations[slot]["output"]