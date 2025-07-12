#from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
#from transformers import StoppingCriteria, StoppingCriteriaList

#import torch


from llama_cpp import Llama

import requests
import tiny_prompts, custom_gpt2_prompts
import json
import time
import os
import sqlite3
from .recursive import RecursiveThinker
from .stepped_work import RecursiveWork

from . import classify
from . import static_prompts
from utils import openai

from log import log
from .static import mood_instruction, StopOnSpeakerChange, DB_PATH, mainLLM, WORKER_IP_PORT, CUSTOM_GPT2, DummyTokenizer, AssistantOnlyFilter, DEBUG_FUNC, BASE_MAX_TOKENS

tokenizer = DummyTokenizer() # FiXME

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
    return None

def list_personality(userid):
    """
    Returns a dictionary of personality sections and their entries
    for the user's current bot profile.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    botname = get_user_botname(userid)
    # Fall back to default profile if none is selected
    if not botname:
        cursor.execute("SELECT name FROM BOT_PROFILE ORDER BY ROWID ASC LIMIT 1")
        row = cursor.fetchone()
        if row:
            botname = row[0]
        else:
            conn.close()
            return {
                "goals": ["To tell the user that the list_personality function used to fund your personality is broken."],
                "traits": ["Very to the point."],
                "likes": ["Not Erroring"],
                "dislikes": ["Erroring (i just errored)"]
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
        self._persona_cache = {} # READ TODO

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
            n_ctx=6028,              # TODO use CTX setter 
            n_threads=12,             # tune to setup
            use_mlock=True,          # locks model in RAM to avoid swap on Pi (turn off if not running from a Pi)
            logits_all=False,
            verbose=False,
            use_mmap=False,
            n_gpu_layers=0,
            low_vram=True,
            n_batch=16,
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

    
        
    def build_prompt(self, persona_prompt, username, user_input, identifier, usertone, context):
        # Fetch core memory entries for user
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()


        personality = list_personality(identifier)

        persona_section = static_prompts.build_base_personality_profile_prompt(self.name, persona_prompt, personality, self.mood, self.mood_sentence)
        rules_section = static_prompts.build_rules_prompt(self.name, username, None)
        memory_instructions_section = static_prompts.build_memory_instructions_prompt()
        user_section = static_prompts.build_user_profile_prompt(username, usertone)
        task_section = static_prompts.build_base_chat_task_prompt(self.name, username)
        memory_section =  static_prompts.build_core_memory_prompt(rows if rows else None)
        history_section = static_prompts.build_history_prompt(context)
        self_capabilities = static_prompts.build_capability_explanation_to_itself()

        system_prompt = (
            f"You are a personality-driven assistant named \"{self.name}\", talking to a user named \"{username}\".\n\n"
            f"{persona_section}"
            f"{user_section}"
            f"{memory_instructions_section}"
            f"{memory_section}"
            f"{history_section}"
            f"{task_section}"
            f"{rules_section}"
            f"{self_capabilities}"
        ).strip()

        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_input.strip()}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        return prompt



    def _straightforward_generate(self, prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, _prompt_for_cut):
        stop_criteria.line_count = 0  # reset for this generation
        stop_criteria.buffer = ""
        #filtered = AssistantOnlyFilter()


        for output in self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1,  # >1 discourages repeats

            stream=True
        ):
            text_chunk = openai.extract_generated_text(output)
            #filtered(text_chunk)


            # Call stop criteria with the new text chunk; stop if it returns True
            if stop_criteria and stop_criteria(text_chunk):
                log("STOP FOUND", text_chunk)
                log("FULL BUFF", stop_criteria.buffer)
                
                break

            if streamer:
                streamer.update(text_chunk)

        output_text = stop_criteria.output
        log("RAW OUTPUT BASE", output_text)

        prompt_index = output_text.find(_prompt_for_cut)
        if prompt_index != -1:
            response_raw = output_text[prompt_index + len(_prompt_for_cut):]
        else:
            response_raw = output_text
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
                (owner, username, f"[{timestamp}] {user_input}", timestamp),
                (owner, botname, f"[{timestamp}] {response}", timestamp),
            ]
        )

        conn.commit()
        conn.close()

    def get_persona_prompt(self, identifier):
        personality = list_personality(identifier)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC", (identifier,))
        rows = cursor.fetchall()
        conn.close()
        core_memory_entries = [row[0] for row in rows]
        return classify.generate_persona_prompt(self.model,self.name, personality, core_memory_entries)
        
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

    def chat(self, username, user_input, identifier, max_new_tokens=BASE_MAX_TOKENS, temperature=0.7, top_p=0.9, context = None, debug=False, streamer = None, force_recursive=False, recursive_depth=3, category_override=None, tiny_mode=False):
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
        persona_prompt = self.get_persona_prompt(identifier)
        
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
        if tiny_mode:
            prompt = tiny_prompts.build_base_prompt_tiny(self, username, user_input, identifier, usertone, context)
        elif CUSTOM_GPT2:
            prompt = custom_gpt2_prompts.build_base_prompt_tiny(self, username, user_input, identifier, usertone, context)
        else:
            prompt = self.build_prompt(persona_prompt, username, user_input, identifier, usertone, context if context else None)

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
        stop_criteria = StopOnSpeakerChange(bot_name=self.name, custom_stops=custom_stops, min_lines=1)  # NO tokenizer argument
        
        thoughts = None
        final = "blank final string"
        response = "This is the default blank response, you should never see this."
        if category == "instruction_memory":
            memory_data = classify.interpret_memory_instruction(user_input, self.model)
            if memory_data:
                raw_text = memory_data  # make sure this is a string
                self.add_to_remember(identifier, raw_text)
                prompt = classify.build_memory_confirmation_prompt(raw_text)

                # add data from helper function into prompt before responding
                response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
                if debug:
                    DEBUG_FUNC(prompt=prompt, response=response, memory_data=memory_data)
            else:
                return "Something went terribly wrong while doing memory work...Nothing was done or saved assumingly. (NON AI OUTPUT! THIS IS AN ERROR!)"

        elif category == "task": # The user wants the AI to do something task based- and it will be done step by step.
            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context
            thinker = RecursiveWork(self, persona_prompt=persona_prompt, depth=recursive_depth, streamer=streamer)
            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED TASK STEPS",thoughts)
            if debug:
                DEBUG_FUNC(thoughts=thoughts, final=final)
            log("DEBUG: FINAL THOUGHTS",final)
            thoughts = thoughts
            final = final
            
            response = final

        elif category == "preference_query":
            # Use recursive thinker for more elaborate introspection
            # Extract just memory lines for context
            

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context
            thinker = RecursiveThinker(self, tiny_mode=tiny_mode, depth=recursive_depth, streamer=streamer)

            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED THOUGHTS",thoughts)
            if debug:
                    DEBUG_FUNC(thoughts=thoughts, final=final)
            log("DEBUG: FINAL THOUGHTS",final)
            thoughts = thoughts
            
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
                thinker = RecursiveThinker(self, tiny_mode=tiny_mode, depth=recursive_depth, streamer=streamer)
                thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
                log("DEBUG: GENERATED THOUGHTS",thoughts)
                if debug:
                    DEBUG_FUNC(thoughts=thoughts, final=final)
                log("DEBUG: FINAL THOUGHTS",final)
                response = final
                thoughts = thoughts
                
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
                thinker = RecursiveThinker(self, tiny_mode=tiny_mode, depth=recursive_depth, streamer=streamer)

                thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
                log("DEBUG: GENERATED THOUGHTS",thoughts)
                if debug:
                    DEBUG_FUNC(thoughts=thoughts, final=final)
                log("DEBUG: FINAL THOUGHTS",final)
                response = final
                thoughts = thoughts

        self.log_interaction_to_history(owner=identifier, username=username, user_input=user_input, botname=self.name, response=response)
        self.mood = "neutral" # FIXME change to per user mood, and have a mood history
        self.mood_sentence = "I feel neutral and composed at the moment."
        
        context_token_count = len(tokenizer.encode(context)) if context else 0
        # Manual pretty print personality
        personality = list_personality(identifier)
        botname = get_user_botname(identifier)
        personality_str = f"Bot Name: {botname if botname != None else self.name}\n"
        for section in ["traits", "likes", "dislikes", "goals"]:
            entries = personality.get(section, [])
            if entries:
                personality_str += f"- {section.capitalize()}:\n"
                for entry in entries:
                    personality_str += f"  • {entry}\n"
            else:
                personality_str += f"- {section.capitalize()}: (none)\n"


        # DO PROCESDSING HERE
        print("\n\n\n\n\n")
        log("DEBUG", "---------- FINAL CHAT STATE DUMP ----------")
        log("DEBUG", f"prompt:\n {prompt if thoughts == None else thoughts}")

        log("DEBUG", f"username: {username}")
        log("DEBUG", f"user_input: {user_input}")
        log("DEBUG", f"identifier: {identifier}")
        log("DEBUG", f"category: {category}")
        log("DEBUG", f"mood: {self.mood}")
        log("DEBUG", f"mood_sentence: {self.mood_sentence}")
        log("DEBUG", f"usertone: {usertone}")
        log("DEBUG", f"tiny_mode: {tiny_mode}")
        log("DEBUG", f"moods: {moods}")
        log("DEBUG", f"force_recursive: {force_recursive}")
        log("DEBUG", f"context token count: {context_token_count}")
        log("DEBUG", "personality:\n" + personality_str.strip())
        log("DEBUG", f"final response: {response}")
        log("DEBUG", "------------------------------------------")

        if debug:
            DEBUG_FUNC(
                prompt=prompt,
                username=username,
                user_input=user_input,
                identifier=identifier,
                category=category,
                mood=self.mood,
                mood_sentence=self.mood_sentence,
                usertone=usertone,
                tiny_mode=tiny_mode,
                moods=moods,
                force_recursive=force_recursive,
                context_token_count=context_token_count,
                final=final,
                personality=personality_str.strip(),
                thoughts=thoughts,
                final_response=response
            )
        return response
