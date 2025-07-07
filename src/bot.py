#from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
#from transformers import StoppingCriteria, StoppingCriteriaList

#import torch


from llama_cpp import Llama

import requests
import json
import time
import os
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from .recursive import RecursiveThinker
from . import classify
from utils import openai

from log import log
from .static import mood_instruction, StopOnSpeakerChange, MEMORY_FILE, mainLLM, WORKER_IP_PORT

tokenizer = None # FiXME

class StringStreamer:
    def __init__(self):
        self.text = ""

    def on_text(self, new_text):
        self.text += new_text


class ChatBot:
    def __init__(self, name="ayokdaeno", memory_file=MEMORY_FILE):
        self.name = name
        self.mood = "neutral"
        self.mood_sentence = "I feel neutral and composed at the moment."
        self.memory_file = memory_file
        self.memory = self.load_memory()
        self.goals = [
            #"Be helpful if asked questions",
            "Provide accurate information",
            #"Keep the conversation friendly." Limits output
        ]
        self.traits = [
            "Curious",
            #"Prefers casual and conversational language",
            "Responds in a way that conveys current mood"
            #"Lets emotional state influence tone heavily."

        ]
        self.likes = [
            "when people are kind and say nice things",
            "receiving compliments",
            "reading books and learning new things",
            "technology and gadgets"
        ]
        self.dislikes = [
            "rudeness or insults",
            "people being mean",
            "darkness",
            "rubber ducks",
            "dogs (I’m definitely more of a cat person)"
        ]

        #self.model = AutoModelForCausalLM.from_pretrained(
        #    MODEL_NAME,
        #    use_auth_token=TOKEN,

        #    torch_dtype=torch.float32,  # Or float16
        #    low_cpu_mem_usage=False,
        #    device_map={"": "cpu"},
        #    trust_remote_code=True
        #)
        #self.model.eval()
        #self.model.config.pad_token_id = tokenizer.eos_token_id
        
        # New TinyLlama model init
        self.model = Llama(
            model_path=mainLLM,
            n_ctx=4096,              # TODO use CTX setter 
            n_threads=4,             # tune to setup
            use_mlock=True,          # locks model in RAM to avoid swap on Pi (turn off if not running from a Pi)
            logits_all=False,
            verbose=False,
            use_mmap=False,
            n_gpu_layers=0,
            low_vram=True,
            n_batch=4,
            numa=False
        )




    def get_mood_based_on_likes_or_dislikes_in_input(self, question):
        try:
            response = requests.post(
                f"http://{WORKER_IP_PORT}/classify_likes_dislikes",  # change if hosted elsewhere
                json={
                    "user_input": question,
                    "likes": self.likes,
                    "dislikes": self.dislikes
                },
                timeout=120
            )
            if response.status_code == 200:
                classification = response.json().get("classification", "NEUTRAL")
            else:
                classification = "NEUTRAL"
        except Exception as e:
            print(f"[WARN] API Down, cant offload to sub models.")
            print("[WARN] Falling back to local model.")
            classification = classify.classify_likes_dislikes_user_input(self.model, tokenizer, question, self.likes, self.dislikes)

        if classification == "LIKE":
            return "happy"
        elif classification == "DISLIKE":
            return "annoyed"
        else:
            return "neutral"

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
            
    def add_to_remember(self, identifier, raw_text):
        if identifier not in self.memory:
            self.memory[identifier] = {"memory": [], "to_remember": []}
        else:
            if "to_remember" not in self.memory[identifier]:
                self.memory[identifier]["to_remember"] = []
        self.memory[identifier]["to_remember"].append(raw_text)
        self.save_memory()


    def get_mood_primitive(self, user_input):
        lowered = user_input.lower()
        if any(word in lowered for word in ["thanks", "great", "awesome", "love", "square"]):
            mood = "happy"
        elif any(word in lowered for word in ["stupid", "bad", "hate", "annoying", "circle"]):
            mood = "annoyed"
        elif any(word in lowered for word in ["rubberducks"]):
            mood = "angry"
        else:
            mood = "neutral"
        return mood
    

    def get_moods_social(self, social_tone_classification: dict):
        try:
            response = requests.post(
                f"http://{WORKER_IP_PORT}/determine_moods",
                json={
                    "classification": social_tone_classification,
                    "top_n": 3
                },
                timeout=120
            )
            if response.status_code == 200:
                moods = response.json().get("top_moods", [])
            else:
                moods = []
        except Exception as e:
            print(f"[WARN] API Down, cant offload to sub models.")
            print("[WARN] Falling back to local model.")
            moods = classify.determine_moods_from_social_classification(social_tone_classification, 3)

        return moods

    
    
    def build_prompt(self, username, user_input, identifier, usertone, context):

        # Get interpreted to_remember facts for the user
        interpreted_facts = classify.interpret_to_remember(self, identifier)
        memory_text = ""
        if context:
                memory_text += f"\n## Relevant Chat History / Context\n"
                memory_text += f"- This contains previous chat history with the user.\n"
                memory_text += context
        if interpreted_facts:
                memory_text += f"\n## User-Stored Facts (These are things the user explicitly told you to remember. Treat them as binding instructions.):\n"
                memory_text += f"{interpreted_facts.strip()}\n"

        system_prompt = (
            f"You are a personality-driven assistant named {self.name}.\n"
            f"Here is your personality profile:\n\n"
            f"**Traits:**\n"
            f"- " + "\n- ".join(self.traits) + "\n\n"
            f"**Likes:**\n"
            f"- " + "\n- ".join(self.likes) + "\n\n"
            f"**Dislikes:**\n"
            f"- " + "\n- ".join(self.dislikes) + "\n\n"
            f"**Goals:**\n"
            f"- " + "\n- ".join(self.goals) + "\n\n"
            f"Current Mood: {self.mood}\n"
            f"Mood Hint: {mood_instruction.get(self.mood, 'Speak in a calm and balanced tone.')}\n\n"
            f"Rules:\n"
            f"- Always speak in the first person.\n"
            f"- Do not explain or mention your personality unless the user asks.\n"
            f"- Do not refer to yourself in third person.\n"
            f"- Do not assume things about the user unless explicitly stated.\n"
            f"- Only refer to the user using the provided info below.\n"
            f"*Task:** You are '{self.name}', a personality-driven assistant. Respond naturally and helpfully, with your mood and traits subtly influencing your wording or tone. Stay grounded in the user input.\n"
        )
        user_prompt = (
            f"Username: {username.replace('<', '').replace('>', '')}\n"
            f"Intent: {usertone['intent']}\n"
            f"Tone: {usertone['tone']}\n"
            f"Attitude: {usertone['attitude']}\n\n"
            f"User Input: {user_input.strip()}\n"
        )
        
        if memory_text != "":
            system_prompt += memory_text

        prompt = (
            f"<|system|>\n{system_prompt.strip()}\n\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n"
        )
        log("FULL BASE PROMPT", prompt)
        return prompt

    def _straightforward_generate(self, prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, _prompt_for_cut):
        output_text = ""
        stop_criteria.line_count = 0  # reset for this generation
        stop_criteria.buffer = ""

        for output in self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.2,  # >1 discourages repeats

            stream=True
        ):
            text_chunk = openai.extract_generated_text(output)
            output_text += text_chunk

            # Call stop criteria with the new text chunk; stop if it returns True
            if stop_criteria and stop_criteria(text_chunk):
                break

            if streamer:
                streamer.on_text(text_chunk)

        prompt_index = output_text.find(_prompt_for_cut)
        if prompt_index != -1:
            response_raw = output_text[prompt_index + len(_prompt_for_cut):]
        else:
            response_raw = output_text
        print(f"\n\n{response_raw}\n")
        return response_raw



        #return response_raw
        #NOTE: this should be handled by the stop
        # Stop early if the model hallucinates another user or assistant prompt
        #for line in response_raw.splitlines():
        #    if line.strip().startswith("<|user|>") or line.strip().startswith("<|assistant|>"):
        #        break
        #    if line.strip().endswith(":") and not line.strip().startswith(self.name):
        #        break
        #    yield_line = line.strip()
        #    if yield_line:
        #        return yield_line
        #
        #return response_raw.strip()

    def _streaming_straightforward_generate(self, inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt):
        # Internal buffer to accumulate streamed text
        output_text = ""

        # llama_cpp streaming generator call
        for chunk in self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.2,
            stream=True
        ):
            text = chunk
            # Call stop criteria with the new text chunk; stop if it returns True
            if stop_criteria and stop_criteria(text):
                break

            output_text += text
            if streamer:
                streamer.on_text(text)  # your streamer can update internal buffer or UI here

        return output_text.strip()



    def chat(self, username, user_input, identifier, max_new_tokens=200, temperature=0.7, top_p=0.9, context = None, debug=False, streamer = None):
        try:
            response = requests.post(
                f"http://{WORKER_IP_PORT}/classify_social_tone",
                json={"user_input": user_input},
                timeout=120
            )
            if response.status_code == 200:
                usertone = response.json().get("classification", {
                    "intent": "NEUTRAL",
                    "attitude": "NEUTRAL",
                    "tone": "NEUTRAL"
                })
            else:
                usertone = {
                    "intent": "NEUTRAL",
                    "attitude": "NEUTRAL",
                    "tone": "NEUTRAL"
                }
        except Exception as e:
            print(f"[WARN] API Down, cant offload to sub models.")
            print("[WARN] Falling back to local model.")
            usertone = classify.classify_social_tone(self.model, tokenizer, user_input)
        moods = {
            "Like/Dislike Mood Factor": { 
                "prompt": "This is the mood factor based on if your likes, or dislikes, were mentioned in the input.",
                "mood": self.get_mood_primitive(user_input),
                },
            "General Input Mood Factor": {
                "prompt": "This is the mood factor based on if the input as a whole is liked, e.g: Did the user compliment/insult, did they talk about one of your likes/dislikes, etc.",
                "mood": self.get_mood_based_on_likes_or_dislikes_in_input(user_input),
                },
            "Social Intents Mood Factor": {
                "prompt": "These are the moods based on the detected social intents from the input, e.g: user intent, user attitude, user tone.",
                "mood": self.get_moods_social(usertone)
            }
        } # TODO Set mood based on all moods
        # Set the base mood based on highest score social mood
        social_moods = moods["Social Intents Mood Factor"]["mood"]
        self.mood = social_moods[0] if social_moods else "uncertain (api error)"
        try:
            response = requests.post(
                f"http://{WORKER_IP_PORT}/classify_moods_into_sentence",
                json={"moods_dict": moods},
                timeout=120
            )
            if response.status_code == 200:
                self.mood_sentence = response.json().get("mood_sentence", "I feel neutral and composed at the moment.")
            else:
                self.mood_sentence = "I feel neutral and composed at the moment."
        except Exception as e:
            print(f"[WARN] API Down, cant offload to sub models.")
            print("[WARN] Falling back to local model.")
            self.mood_sentence = classify.classify_moods_into_sentence(self.model, tokenizer, moods)
        prompt = self.build_prompt(username, user_input, identifier, usertone, context if context else "\n".join(memory[-10:]))

        #inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        #log("DEBUG: DEFAULT PROMPT TOKENS", inputs.input_ids.size(1))

        try:
            response = requests.post(
                f"http://{WORKER_IP_PORT}/classify_user_input",
                json={"user_input": user_input},
                timeout=120
            )
            if response.status_code == 200:
                category = response.json().get("category", "other")
            else:
                category = "other"
        except Exception as e:
            print(f"[WARN] API Down, cant offload to sub models.")
            print("[WARN] Falling back to local model.")
            category = classify.classify_user_input(self.model, tokenizer, user_input)
        stop_criteria = StopOnSpeakerChange(bot_name=self.name)  # NO tokenizer argument
        
        response = "This is the default blank response, you should never see this."
        if category == "instruction_memory":
            memory_data = classify.interpret_memory_instruction(self, user_input)
            if memory_data:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                if identifier not in self.memory:
                    self.memory[identifier] = {"memory": [], "to_remember": []}
                else:
                    # Ensure keys exist
                    if "memory" not in self.memory[identifier]:
                        self.memory[identifier]["memory"] = []
                    if "to_remember" not in self.memory[identifier]:
                        self.memory[identifier]["to_remember"] = []

                # Store raw memory data as JSON strings in to_remember (no flattening)
                import json
                # Wrap memory_data with timestamp for reference
                entry = {"date": timestamp, **memory_data}
                self.memory[identifier]["to_remember"].append(json.dumps(entry))
            memory_data_ai_readable = classify.interpret_to_remember(self, identifier, max_new_tokens=400)
            prompt = classify.build_memory_confirmation_prompt(memory_data_ai_readable)

            # add data from helper function into prompt before responding
            response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)


        elif category == "factual_question":
            # Use recursive thinker for more elaborate introspection
            # TODO: query internet sources for facts
            # Extract just memory lines for context
            memory = self.memory.get(identifier, {}).get("memory", [])

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = "\n".join(memory[-10:])  # last 10 lines = 5 pairs of messages
            else:
                short_context = "\n".join(context)

            thinker = RecursiveThinker(self, depth=4, streamer=streamer)
            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                final = f"{thoughts}\n{final}"
            log("DEBUG: FINAL THOUGHTS",final)
            
            response = final
        elif category == "preference_query":
            # Use recursive thinker for more elaborate introspection
            # Extract just memory lines for context
            memory = self.memory.get(identifier, {}).get("memory", [])

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = "\n".join(memory[-10:])  # last 10 lines = 5 pairs of messages
            else:
                short_context = "\n".join(context)
            thinker = RecursiveThinker(self, depth=5, streamer=streamer)

            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                final = f"{thoughts}\n{final}"
            log("DEBUG: FINAL THOUGHTS",final)
            response = final
        elif category == "other":
            # Use recursive thinker for more elaborate introspection
            # Extract just memory lines for context
            memory = self.memory.get(identifier, {}).get("memory", [])

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = "\n".join(memory[-10:])  # last 10 lines = 5 pairs of messages
            else:
                short_context = "\n".join(context)
            thinker = RecursiveThinker(self, depth=3, streamer=streamer)

            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                final = f"{thoughts}\n{final}"
            log("DEBUG: FINAL THOUGHTS",final)
        else: #fallback 
            response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if identifier not in self.memory:
                self.memory[identifier] = {"memory": [], "to_remember": []}
        self.memory[identifier]["memory"].append(f"[{timestamp}] {username}: {user_input}")
        self.memory[identifier]["memory"].append(f"[{timestamp}] {self.name}: {response}")

        # Limit memory to 80 lines
        #if len(self.memory[identifier]) > 80:
        #    self.memory[identifier] = self.memory[identifier][-80:]

        self.save_memory()  # Save after each interaction
        return response
