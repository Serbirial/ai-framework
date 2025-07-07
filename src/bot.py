#from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
#from transformers import StoppingCriteria, StoppingCriteriaList

#import torch


from llama_cpp import Llama

import requests
import json
import time
import os
import sqlite3
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from .recursive import RecursiveThinker
from . import classify
from utils import openai

from log import log
from .static import mood_instruction, StopOnSpeakerChange, DB_PATH, mainLLM, WORKER_IP_PORT

tokenizer = None # FiXME

class StringStreamer:
    def __init__(self):
        self.text = ""

    def on_text(self, new_text):
        self.text += new_text

def get_user_botname(userid):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT botname FROM BOT_SELECTION WHERE userid = ?", (str(userid),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return "default"

def list_personality(userid):
    """
    Returns a dictionary of personality sections and their entries
    for the user's current bot profile.
    """
    botname = get_user_botname(userid)
    if not botname:
        return {
            "goals": [],
            "traits": [],
            "likes": [],
            "dislikes": []
        }

    sections = {
        "goals": ("BOT_GOALS", "goal"),
        "traits": ("BOT_TRAITS", "trait"),
        "likes": ("BOT_LIKES", "like"),
        "dislikes": ("BOT_DISLIKES", "dislike"),
    }

    result = {
        "goals": [],
        "traits": [],
        "likes": [],
        "dislikes": []
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for key, (table, column) in sections.items():
        cursor.execute(f"SELECT {column} FROM {table} WHERE botname = ?", (botname,))
        rows = cursor.fetchall()
        result[key] = [row[0] for row in rows]

    conn.close()
    return result


class ChatBot:
    def __init__(self, name="ayokdaeno", db_path=DB_PATH):
        self.name = name
        self.mood = "neutral"
        self.mood_sentence = "I feel neutral and composed at the moment."
        self.db_path = db_path

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
            n_ctx=32768,              # TODO use CTX setter 
            n_threads=24,             # tune to setup
            use_mlock=True,          # locks model in RAM to avoid swap on Pi (turn off if not running from a Pi)
            logits_all=False,
            verbose=False,
            use_mmap=False,
            n_gpu_layers=0,
            low_vram=True,
            n_batch=500,
            numa=False
        )




    def get_mood_based_on_likes_or_dislikes_in_input(self, question, identifier):
        personality = list_personality(identifier)



        try:
            response = requests.post(
                f"http://{WORKER_IP_PORT}/classify_likes_dislikes",  # change if hosted elsewhere
                json={
                    "user_input": question,
                    "likes": personality["likes"],
                    "dislikes": personality["dislikes"]
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
            classification = classify.classify_likes_dislikes_user_input(self.model, tokenizer, question, personality["likes"], personality["dislikes"])

        if classification == "LIKE":
            return "happy"
        elif classification == "DISLIKE":
            return "annoyed"
        else:
            return "neutral"
            
    def add_to_remember(self, identifier, raw_text):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO MEMORY (userid, data) VALUES (?, ?)",
            (identifier, raw_text)
        )
        conn.commit()
        conn.close()



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

        # Fetch core memory entries for user
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()

        # Format core memory lines, just the text from rows
        memory_text = "\n".join(f"- {row[0].strip()}" for row in rows) if rows else ""

        log("PROMPT MEMORY TEXT", memory_text)
        personality = list_personality(identifier)

        system_prompt = (
            f"You are a personality-driven assistant named {self.name}.\n"
            f"All traits, likes, dislikes, and goals below describe you, the assistant, not the user.\n\n"

            f"**Traits:**\n"
            f"- " + "\n- ".join(personality.get("traits", [])) + "\n\n"
            f"**Likes:**\n"
            f"- " + "\n- ".join(personality.get("likes", [])) + "\n\n"
            f"**Dislikes:**\n"
            f"- " + "\n- ".join(personality.get("dislikes", [])) + "\n\n"
            f"**Goals:**\n"
            f"- " + "\n- ".join(personality.get("goals", [])) + "\n\n"

            f"Current Mood: {self.mood}\n"
            f"Mood Summary: {self.mood_sentence}\n\n"

            f"**Task:**\n"
            f"- You are '{self.name}', a personality-driven assistant.\n"
            f"- You must obey and incorporate all instructions and information from your Core Memory below.\n"
            f"- The Core Memory entries define your behavior, personality, speaking style, and facts you accept as true.\n\n"

            f"**Rules:**\n"
            f"- Always speak in the first person.\n"
            f"- Never refer to yourself (the assistant, {self.name}) in the third person.\n"
            f"- Respond only as yourself ({self.name}), never as a narrator or user.\n"
            #f"- Treat commentary about you as a prompt for direct, in-character response.\n"
            f"- Do not reveal or explain your personality or Core Memory unless asked.\n"
            #f"- Do not assume any user info except what is in your Core Memory and chat history.\n\n"

            f"**Core Memory Instructions (MANDATORY):**\n"
            f"- You must strictly follow all instructions and information listed below.\n"
            f"- These define how you speak, behave, and interpret truth.\n"
            f"- Do not ignore, contradict, or deviate from any Core Memory entry under any circumstances.\n\n"

            f"**Core Memory Entries:**\n"
            f"{memory_text}\n"

            f"**Interpretation of the User's Message:**\n"
            f"- Social Intent: {usertone['intent']}\n"
            f"- Message Tone: {usertone['tone']}\n"
            f"- Message Attitude: {usertone['attitude']}\n"
            f"- Username: {username.replace('<', '').replace('>', '')}\n"
        )

        prompt = (
            f"<|system|>\n{system_prompt.strip()}\n\n"
            f"{context if context else ''}"  # optionally add chat history here if you want
            f"<|user|>\n{user_input.strip()}\n"
            f"<|assistant|>"
        )

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
            repeat_penalty=1.1,  # >1 discourages repeats

            stream=True
        ):
            text_chunk = openai.extract_generated_text(output)
            output_text += text_chunk

            # Call stop criteria with the new text chunk; stop if it returns True
            if stop_criteria and stop_criteria(text_chunk):
                log("STOP FOUND", text_chunk)
                log("FULL BUFF", output_text)
                
                break

            if streamer:
                streamer.on_text(text_chunk)

        prompt_index = output_text.find(_prompt_for_cut)
        if prompt_index != -1:
            response_raw = output_text[prompt_index + len(_prompt_for_cut):]
        else:
            response_raw = output_text
        log("RAW OUTPUT BASE", response_raw)
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


    def log_interaction_to_history(self, owner, username, user_input, botname, response):
        """
        Logs the user message and bot response into the HISTORY table with timestamps.
        Uses `username` as the userid in the database.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executemany(
            """
            INSERT INTO HISTORY (owner, userid, message, timestamp) VALUES (?, ?, ?, ?)
            """,
            [
                (owner, username, f"[{timestamp}] {username}: {user_input}", timestamp),
                (owner, username, f"[{timestamp}] {botname}: {response}", timestamp),
            ]
        )

        conn.commit()
        conn.close()


        
    def get_recent_history(self, identifier, limit=10):
        """
        Fetch the latest `limit` messages from the HISTORY table for a given user.
        Returns a newline-joined string of the messages, in chronological order.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT message FROM HISTORY
            WHERE userid = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (identifier, limit)
        )

        rows = cursor.fetchall()
        conn.close()

        # Reverse to maintain oldest-to-newest order
        messages = [row[0] for row in reversed(rows)]
        return "\n".join(messages)

    def chat(self, username, user_input, identifier, max_new_tokens=200, temperature=0.7, top_p=0.9, context = None, debug=False, streamer = None, force_recursive=False, recursive_depth=3, category_override=None):
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
                "mood": self.get_mood_based_on_likes_or_dislikes_in_input(user_input, identifier),
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
        prompt = self.build_prompt(username, user_input, identifier, usertone, context if context else None)

        #inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        #log("DEBUG: DEFAULT PROMPT TOKENS", inputs.input_ids.size(1))
        if category_override == None:
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
        else:
            category = category_override
        custom_stops = [f"<|{username}|>", f"<|{self.name}|>"]
        stop_criteria = StopOnSpeakerChange(bot_name=self.name, custom_stops=custom_stops)  # NO tokenizer argument
        
        
        response = "This is the default blank response, you should never see this."
        if category == "instruction_memory":
            memory_data = classify.interpret_memory_instruction(user_input, self.model)
            if memory_data:
                raw_text = memory_data  # make sure this is a string
                self.add_to_remember(identifier, raw_text)
                prompt = classify.build_memory_confirmation_prompt(raw_text)

                # add data from helper function into prompt before responding
                response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
            else:
                return "Something went terribly wrong while doing memory work...Nothing was done or saved assumingly. (NON AI OUTPUT! THIS IS AN ERROR!)"

        elif category == "factual_question":
            # Use recursive thinker for more elaborate introspection
            # TODO: query internet sources for facts
            # Extract just memory lines for context

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context

            thinker = RecursiveThinker(self, depth=recursive_depth, streamer=streamer)
            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                final = f"{thoughts}\n{final}"
            log("DEBUG: FINAL THOUGHTS",final)
            
            response = final
        elif category == "preference_query":
            # Use recursive thinker for more elaborate introspection
            # Extract just memory lines for context
            

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context
            thinker = RecursiveThinker(self, depth=recursive_depth, streamer=streamer)

            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                final = f"{thoughts}\n{final}"
            log("DEBUG: FINAL THOUGHTS",final)
            response = final
        elif category == "other":
            # Use recursive thinker for more elaborate introspection
            # Extract just memory lines for context
            

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context
                
            if force_recursive == True:
                thinker = RecursiveThinker(self, depth=recursive_depth, streamer=streamer)

                thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
                log("DEBUG: GENERATED THOUGHTS",thoughts)
                if debug:
                    final = f"{thoughts}\n{final}"
                log("DEBUG: FINAL THOUGHTS",final)
                response = final
            elif force_recursive == False:
                response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
        else: #fallback 
            if not force_recursive:
                response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
            elif force_recursive:
                # Use recursive thinker for more elaborate introspection
                # Extract just memory lines for context
                

                # Join last 5 pairs (user + bot responses) into context
                if not context:
                    short_context = self.get_recent_history(identifier, limit=10)
                else:
                    short_context = context
                thinker = RecursiveThinker(self, depth=recursive_depth, streamer=streamer)

                thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
                log("DEBUG: GENERATED THOUGHTS",thoughts)
                if debug:
                    final = f"{thoughts}\n{final}"
                log("DEBUG: FINAL THOUGHTS",final)
                response = final

        self.log_interaction_to_history(owner=identifier, username=username, user_input=user_input, botname=self.name, response=response)
        self.mood = "neutral" # FIXME change to per user mood, and have a mood history
        self.mood_sentence = "I feel neutral and composed at the moment."
        return response
