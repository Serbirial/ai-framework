#from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
#from transformers import StoppingCriteria, StoppingCriteriaList

#import torch


from llama_cpp import Llama

import requests
import tiny_prompts, custom_gpt2_prompts
import json
import time
import os
import asyncio, re
import sqlite3
from .recursive import RecursiveThinker
from .stepped_work import RecursiveWork

from . import classify
from . import grouped_preprocessing
from . import static_prompts
from utils import openai
from utils.helpers import get_mem_tokens_n

from log import log
from .static import StopOnSpeakerChange, DB_PATH, WORKER_IP_PORT, DummyTokenizer, DEBUG_FUNC, Config

CONFIG_VAR = Config()

tokenizer = DummyTokenizer() # FiXME



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
    def __init__(self, name="ayokdaeno", db_path=DB_PATH, model = None):
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
        if model:
            self.model = model
        else:
            self.model = Llama(
                model_path=CONFIG_VAR.general["main_llm_path"],
                n_ctx=16000,              # TODO use CTX setter 
                n_threads=7,             # tune to setup
                use_mlock=True,          # locks model in RAM to avoid swap on Pi (turn off if not running from a Pi)
                logits_all=False,
                verbose=False,
                use_mmap=True,
                n_gpu_layers=32,
                low_vram=False,
                n_batch=64
                #numa=False
            )




    def get_mood_based_on_like_dislike(self, classification):

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

    
        
    def build_prompt(self, persona_prompt, username, user_input, identifier, usertone, context, cnn_output=None):
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
            f"{task_section}"
            f"{rules_section}"
            f"{self_capabilities}"

        ).strip()
        if cnn_output:
            system_prompt += cnn_output

        prompt = (
            #"<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            "<|eot_id|>"
            f"{history_section}"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_input.strip()}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        return prompt



    def _straightforward_generate(self, prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, _prompt_for_cut):
        stop_criteria.line_count = 0  # reset for this generation
        stop_criteria.buffer = ""
        stop_criteria.output = ""
        stop_criteria.stopped = False
        if streamer:
            streamer.buffer = ""
            streamer.special_buffer = []
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
                streamer(text_chunk)

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
    
    def run_classifiers(self, tokenizer, user_input, category_override, identifier, history):
        """Returns usertone, moods, mood, persona prompt, category
        """
        
        personality = list_personality(identifier)

        usertone, category, like_or_dislike, mood_sentence = grouped_preprocessing.basic_preprocessing(self.model, user_input, personality["likes"], personality["dislikes"], history)
        persona_prompt = self.get_persona_prompt(identifier)
        if category_override:
            category = category_override
        def get_moods():
            return {
                "Like/Dislike Mood Factor": { 
                    "prompt": "This is the mood factor based on if your likes, or dislikes, were mentioned in the input.",
                    "mood": self.get_mood_primitive(user_input),
                    },
                "General Input Mood Factor": {
                    "prompt": "This is the mood factor based on if the input as a whole is liked, e.g: Did the user compliment/insult, did they talk about one of your likes/dislikes, etc.",
                    "mood": self.get_mood_based_on_like_dislike(like_or_dislike),
                    },
                "Social Intents Mood Factor": {
                    "prompt": "These are the moods based on the detected social intents from the input, e.g: user intent, user attitude, user tone.",
                    "mood": self.get_moods_social(usertone)
                }
            } # TODO Set mood based on all moods

        moods = get_moods()

        # Set the base mood based on highest score social mood
        social_moods = moods["Social Intents Mood Factor"]["mood"]
        mood = social_moods[0] if social_moods else "uncertain (api error)"


        return usertone, moods, mood, mood_sentence, persona_prompt, category

    def chat(self, username, user_input, identifier, tier, max_new_tokens=None, temperature=0.7, top_p=0.9, context = None, debug=False, streamer = None, force_recursive=False, recursive_depth=3, category_override=None, tiny_mode=False, cnn_file_path=None):
        cnn_output = None
        cnn_output_formatted = None

        if not max_new_tokens:
            max_new_tokens = CONFIG_VAR.token_config[tier]["BASE_MAX_TOKENS"]

        max_memory_tokens = CONFIG_VAR.token_config[tier]["MAX_MEMORY_TOKENS"]

        if cnn_file_path:
            if streamer:
                streamer.add_special(f"Image Detetcted, starting processing (may take a while)...")
            
            try:
                with open(cnn_file_path, "rb") as f:
                    cnn_response = requests.post(
                        f"http://localhost:6006/describe_image",  # or your actual CNN endpoint
                        files={"image": f},
                        timeout=500
                    )
                if cnn_response.status_code == 200:
                    cnn_output = cnn_response.json().get("description", None)
                else:
                    if streamer:
                        streamer.add_special(f"Major Error")
                    cnn_output = f"### ERROR: CNN API returned status {cnn_response.status_code}"
            except Exception as e:
                if streamer:
                    streamer.add_special(f"Major Error")
                cnn_output = f"### ERROR: CNN API request failed: {str(e)}"
                
        if cnn_output != None:            
            category_override = "other" # temp
            cnn_output_formatted = static_prompts.build_cnn_input_prompt(cnn_output)
            if streamer:
                streamer.add_special(f"Image processed.")
        if streamer:
            streamer.add_special("Pre-processing input")
        usertone, moods, mood, mood_sentence, persona_prompt, category = self.run_classifiers(tokenizer, user_input, category_override, identifier, context)
        
        self.mood = mood
        
        
        self.mood_sentence = mood_sentence
        if streamer:
            streamer.add_special(f"Setting mood to {self.mood}\n")
            streamer.add_special(f"Setting mood sentence to {self.mood_sentence}\n")
            
            
            
        if tiny_mode:
            prompt = tiny_prompts.build_base_prompt_tiny(self, username, user_input, identifier, usertone, context)
        elif CONFIG_VAR.general["custom_gpt2"]:
            prompt = custom_gpt2_prompts.build_base_prompt_tiny(self, username, user_input, identifier, usertone, context)
        else:
            prompt = self.build_prompt(persona_prompt, username, user_input, identifier, usertone, context if context else None, cnn_output=cnn_output_formatted)

        #inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        #log("DEBUG: DEFAULT PROMPT TOKENS", inputs.input_ids.size(1))

        custom_stops = [f"<|{username}|>", f"<|{self.name}|>"]
        stop_criteria = StopOnSpeakerChange(bot_name=self.name, custom_stops=custom_stops, min_lines=1)  # NO tokenizer argument
        if streamer:
            streamer.add_special(f"I have classified your message as `{category}` and im routing your response accordingly...")
        
        thoughts = None
        final = "blank final string"
        response = "This is the default blank response, you should never see this."
        if category == "instruction_memory":
            if streamer:
                streamer.add_special(f"Trying to save to memory...")

            if get_mem_tokens_n(identifier, max_memory_tokens) > max_memory_tokens:
                return "I cant store anything in my memory right now. (AT LIMIT)"
            memory_data = classify.interpret_memory_instruction(user_input, self.model)
            if memory_data:
                raw_text = memory_data  # make sure this is a string
                
                self.add_to_remember(identifier, raw_text)
                if streamer:
                    streamer.add_special(f"Saved to memory.")
                
                prompt = classify.build_memory_confirmation_prompt(raw_text)

                # add data from helper function into prompt before responding
                response = self._straightforward_generate(prompt, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
                if debug:
                    DEBUG_FUNC(prompt=prompt, response=response, memory_data=memory_data)
            else:
                if streamer:
                    streamer.add_special(f"Major error")
                return "Something went terribly wrong while doing memory work...Nothing was done or saved assumingly. (NON AI OUTPUT! THIS IS AN ERROR!)"

        elif category == "task": # The user wants the AI to do something task based- and it will be done step by step.
            if streamer:
                streamer.add_special(f"Trying to complete the task...")

            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context
            thinker = RecursiveWork(self, config=CONFIG_VAR, persona_prompt=persona_prompt, depth=recursive_depth, streamer=streamer)
            thoughts, final = thinker.think(tier=tier, question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
            log("DEBUG: GENERATED TASK STEPS",thoughts)
            if debug:
                DEBUG_FUNC(thoughts=thoughts, final=final)
            log("DEBUG: FINAL THOUGHTS",final)
            thoughts = thoughts
            final = final
            
            response = final

        elif category == "preference_query":
            if streamer:
                streamer.add_special(f"Internally reasoning...")
            
            # Use recursive thinker for more elaborate introspection
            # Extract just memory lines for context
            

            # Join last 5 pairs (user + bot responses) into context
            if not context:
                short_context = self.get_recent_history(identifier, limit=10)
            else:
                short_context = context
            thinker = RecursiveThinker(self, CONFIG_VAR, persona_prompt, tiny_mode=tiny_mode, depth=recursive_depth, streamer=streamer)

            thoughts, final = thinker.think(question=user_input, tier=tier, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
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
                if streamer:
                    streamer.add_special(f"Forcing recursive (will take longer).")
                
                thinker = RecursiveThinker(self, CONFIG_VAR, persona_prompt, tiny_mode=tiny_mode, depth=recursive_depth, streamer=streamer)
                thoughts, final = thinker.think(question=user_input, tier=tier, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
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
                if streamer:
                    streamer.add_special(f"Forcing recursive (will take longer).")
                
                # Use recursive thinker for more elaborate introspection
                # Extract just memory lines for context
                

                # Join last 5 pairs (user + bot responses) into context
                if not context:
                    short_context = self.get_recent_history(identifier, limit=10)
                else:
                    short_context = context
                thinker = RecursiveThinker(self, CONFIG_VAR, persona_prompt, tiny_mode=tiny_mode, depth=recursive_depth, streamer=streamer)

                thoughts, final = thinker.think(question=user_input, tier=tier, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
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
