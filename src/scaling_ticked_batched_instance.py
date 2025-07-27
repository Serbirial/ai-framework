from .bot import AgentInstance, list_personality
import asyncio
import sqlite3
import prompt_builder
from .static import Config, StopOnSpeakerChange, DummyTokenizer
from llama_cpp import Llama

config = Config()
tokenizer = DummyTokenizer()

temp_persona_prompt = """You are Ayok! A friendly personality!"""
temp_mood = "neutral"
temp_mood_sentence = "Speak in a calm and collected manner"
temp_usertone = {"intent": "neutral", "tone": "neutral", "attitude": "neutral"}

class BatchedSimpleChat:
    def __init__(self, model_path: str, botname: str, max_concurrent: int = 2, n_ctx: int | None = None):
        self.n_ctx = n_ctx 
        self.max_concurrent = max_concurrent
        self.outputs = {}
        self.model = Llama(
            model_path=model_path,
            n_ctx=8096,
            n_threads=3,
            use_mlock=False,
            logits_all=False,
            verbose=False,
            use_mmap=True,
            n_gpu_layers=32,
            low_vram=False,
            n_batch=64
        )
        self.bot = AgentInstance(name=botname, model_path=model_path, model=self.model)


        self.db_path = self.bot.db_path

        for i in range(self.max_concurrent):
            self.outputs[i] = {
                "in_use": False,
                "done": False,
                "identifier": None,
                "username": None,
                "prompt": None,
                "input": None,
                "output": None,
                "max_new_tokens": None
            }
            
    def tick(self, new_tokens: int = 16):
        active_slots = [slot_id for slot_id, data in self.outputs.items() if data["in_use"]]
        if not active_slots:
            return  # no active slots

        prompts = [self.outputs[slot]["prompt"] for slot in active_slots]
        max_new_tokens_list = [self.outputs[slot]["max_new_tokens"] for slot in active_slots]

        for idx, slot in enumerate(active_slots):
            if self.outputs[slot]["output"] == None:
                self.outputs[slot]["output"] = ""
            max_new_tokens = max_new_tokens_list[idx]
            if tokenizer.encode(self.outputs[slot]["output"]) >= max_new_tokens:
                self.outputs[slot]["done"] == True
            else:
                if not self.outputs[slot]["done"]:
                    try:
                        prompt = prompts[idx]

                        output = self.bot._straightforward_generate(
                            prompt=prompt,
                            max_new_tokens=new_tokens,
                            temperature=0.7,
                            top_p=0.9,
                            streamer=None,
                            stop_criteria=None,
                            _prompt_for_cut=prompt
                        )
                        self.outputs[slot]["output"] += output
                        if tokenizer.encode(self.outputs[slot]["output"]) >= max_new_tokens:
                            self.outputs[slot]["done"] == True
                    except Exception as e:
                        self.outputs[slot]["output"] = {"error": str(e)}
                        self.outputs[slot]["in_use"] = False

    def assign(self, identifier, username, userinput, tier, context=None):
        free_slot = None
        for slot_id, slot_data in self.outputs.items():
            if not slot_data["in_use"]:
                free_slot = slot_id
                break
        
        if free_slot is None:
            return None  

        max_new_tokens = config.token_config.get(tier, {}).get("BASE_MAX_TOKENS", 256)
        
        personality = list_personality(identifier)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()

        persona_section = prompt_builder.build_base_personality_profile_prompt(self.bot.name, temp_persona_prompt, personality, temp_mood, temp_mood_sentence)
        rules_section = prompt_builder.build_rules_prompt(self.bot.name, username, None)
        memory_section =  prompt_builder.build_core_memory_prompt(rows if rows else None)
        memory_instructions_section = prompt_builder.build_memory_instructions_prompt()
        user_section = prompt_builder.build_user_profile_prompt(username, temp_usertone)
        task_section = prompt_builder.build_base_chat_task_prompt(self.bot.name, username)
        history_section = prompt_builder.build_history_prompt(context or "")

        system_prompt = (
            f"You are a personality-driven assistant named \"{self.bot.name}\", talking to a user named \"{username}\".\n\n"
            f"{persona_section}"
            f"{user_section}"
            f"{memory_instructions_section}"
            f"{memory_section}"
            f"{task_section}"
            f"{rules_section}"
        ).strip()
        
        prompt = (
            f"{history_section}"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{userinput.strip()}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        self.outputs[free_slot].update({
            "in_use": True,
            "identifier": identifier,
            "username": username,
            "prompt": prompt,
            "input": userinput,
            "output": None,
            "max_new_tokens": max_new_tokens
        })

        return free_slot

    def get_output(self, slot):
        """
        Retrieve output for a slot and clear the slot after reading.
        """
        slot_data = self.outputs.get(slot)
        if not slot_data or slot_data["in_use"]:
            return None
        
        output = slot_data["output"]

        self.outputs[slot] = {
            "in_use": False,
            "done": False, 
            "identifier": None,
            "username": None,
            "prompt": None,
            "input": None,
            "output": None,
            "max_new_tokens": None
        }
        return output

    async def wait_for_output(self, slot):
        while self.outputs[slot]["done"] != True:
            asyncio.sleep(5)
            if self.outputs[slot]["done"] == True:
                break
            
        return self.get_output(slot)