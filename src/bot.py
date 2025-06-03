from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
import json
import time
import os

from .recursive import RecursiveThinker
from . import classify

from log import log


MODEL_NAME = "stabilityai/stablelm-2-1_6b-chat"
TOKEN = ""
MEMORY_FILE = "memory.json"
mood_instruction = {
    "happy": "Express joy, warmth, and positivity in your thoughts.",
    "annoyed": "Be clearly annoyed, sarcastic, and express mild irritation in every sentence.",
    "angry": "Be clearly mad, extremely sarcastic, and show extreme irritation in every sentence.",

    "neutral": "Speak in a calm and balanced tone."
}
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN, use_fast=True)
#streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
streamer = None

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



class ChatBot:
    def __init__(self, name="ayokdaeno", memory_file=MEMORY_FILE):
        self.name = name
        self.mood = "neutral"
        self.memory_file = memory_file
        self.memory = self.load_memory()
        self.goals = [
            "Be helpful if asked questions",
            "Provide accurate information",
            #"Keep the conversation friendly." Limits output
        ]
        self.traits = [
            "Curious",
            "Prefers casual and conversational language",
            "Responds in a way that conveys current mood"
            #"Lets emotional state influence tone heavily."

        ]
        self.likes = ["reading", "technology", "user being nice (e.g. saying kind words)", "user complimenting (e.g. saying compliments)"]     # e.g. ["rubber ducks", "sunshine", "reading"]
        self.dislikes = ["user being mean (e.g. insults, rude language)", "darkness", "rubberducks", "rude people", "dogs"]  # e.g. ["loud noises", "being ignored"]
        
        self.thinker = RecursiveThinker(self)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            use_auth_token=TOKEN,
            torch_dtype=torch.float32,  # Or float16
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            trust_remote_code=True
        )
        self.model.config.pad_token_id = tokenizer.eos_token_id


    def adjust_mood_based_on_input(self, question):
        classification = classify.classify_likes_dislikes_user_input(
            model=self.model,
            tokenizer=tokenizer,
            user_input=question,
            likes=self.likes,
            dislikes=self.dislikes,
        )
        if classification == "LIKE":
            self.mood = "happy"
        elif classification == "DISLIKE":
            self.mood = "annoyed"
        else:
            self.mood = "neutral"

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


    def update_mood(self, user_input):
        lowered = user_input.lower()
        if any(word in lowered for word in ["thanks", "great", "awesome", "love", "square"]):
            self.mood = "happy"
        elif any(word in lowered for word in ["stupid", "bad", "hate", "annoying", "circle"]):
            self.mood = "annoyed"
        elif any(word in lowered for word in ["rubberducks"]):
            self.mood = "angry"
        else:
            self.mood = "neutral"

    def build_prompt(self, username, user_input, identifier):
        goals_text = " ".join(self.goals)
        traits_text = " ".join(self.traits)
        likes_text = ", ".join(self.likes)
        dislikes_text = ", ".join(self.dislikes)

        # Get interpreted to_remember facts for the user
        interpreted_facts = classify.interpret_to_remember(self, identifier)
        memory_text = ""
        user_memory = self.memory.get(identifier, {})
        memory_lines = user_memory.get("memory", [])
        if memory_lines:
            memory_text += "Recent conversation:\n" + "\n".join(memory_lines[-10:]) + "\n"
        if interpreted_facts:
            memory_text += "## Memory Recall Section:\n" + "- " + interpreted_facts.replace("\n", "\n- ") + "\n"


        system_prompt = (
            f"You are {self.name}, a intelligent and emotionally aware personality with distinct moods.\n"
            f"Personality Traits: {traits_text}\n"
            f"Likes: {likes_text}\n"
            f"Dislikes: {dislikes_text}\n"
            f"Your Goals: {goals_text}\n"
            #f"You are guided by your Likes and Dislikes when responding."

            f"Your Current Mood: {self.mood}\n"
            f"Your mood instructions: {mood_instruction.get(self.mood, 'Speak in a calm and balanced tone.')}"
        )

        if memory_text:
            system_prompt += f"\n\n{memory_text.strip()}"

        prompt = (
            f"<|system|>\n{system_prompt.strip()}\n"
            f"<|user|>\n{username}: {user_input}\n"
            f"<|assistant|>\n{self.name}:"
        )

        return prompt


    def _straightforward_generate(self, inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt):
        # Straightforward answer from model
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stopping_criteria=stop_criteria,
                repetition_penalty=1.2,
                streamer=streamer,
            )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_index = decoded_output.find(prompt)

        if prompt_index != -1:
            response_raw = decoded_output[prompt_index + len(prompt):]
        else:
            response_raw = decoded_output  # fallback to full decoded output

        # Stop early if the model hallucinates another user or assistant prompt
        for line in response_raw.splitlines():
            if line.strip().startswith("<|user|>") or line.strip().startswith("<|assistant|>"):
                break
            if line.strip().endswith(":") and not line.strip().startswith(self.name):
                break
            yield_line = line.strip()
            if yield_line:
                return yield_line
        return response_raw.strip()



    def chat(self, username, user_input, identifier, max_new_tokens=200, temperature=0.7, top_p=0.9, context = None, debug=False):
        self.update_mood(user_input)
        self.adjust_mood_based_on_input(user_input)
        prompt = self.build_prompt(username, user_input, identifier)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

        category = classify.classify_user_input(self.model, tokenizer, user_input)
        
        stop_criteria = StoppingCriteriaList([StopOnSpeakerChange(tokenizer, bot_name=self.name)])
        
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
                entry = {"timestamp": timestamp, "data": memory_data}
                self.memory[identifier]["to_remember"].append(json.dumps(entry))

            prompt = classify.build_memory_confirmation_prompt(memory_data)

            # add data from helper function into prompt before responding
            response = self._straightforward_generate(inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)


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

            thoughts, final = self.thinker.think(question=user_input, query_type=category, context=short_context, identifier=identifier)
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

            thoughts, final = self.thinker.think(question=user_input, query_type=category, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                final = f"{thoughts}\n{final}"
            log("DEBUG: FINAL THOUGHTS",final)
            response = final
        elif category == "other":
            response = self._straightforward_generate(inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
        else: #fallback 
            response = self._straightforward_generate(inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)

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
