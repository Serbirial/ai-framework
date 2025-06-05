from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
import json
import time
import os

from .recursive import RecursiveThinker
from . import classify

from log import log
from .static import mood_instruction, StopOnSpeakerChange, tokenizer, MODEL_NAME, TOKEN, MEMORY_FILE

#streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


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
        self.likes = ["reading", "technology", "user being nice (e.g. saying kind words)", "user complimenting (e.g. saying compliments)"]     # e.g. ["rubber ducks", "sunshine", "reading"]
        self.dislikes = ["user being mean (e.g. insults, rude language)", "darkness", "rubberducks", "rude people", "dogs"]  # e.g. ["loud noises", "being ignored"]
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            use_auth_token=TOKEN,

            torch_dtype=torch.float32,  # Or float16
            low_cpu_mem_usage=False,
            device_map={"": "cpu"},
            trust_remote_code=True
        )
        self.model.eval()
        self.model.config.pad_token_id = tokenizer.eos_token_id


    def get_mood_based_on_likes_or_dislikes_in_input(self, question):
        classification = classify.classify_likes_dislikes_user_input(
            model=self.model,
            tokenizer=tokenizer,
            user_input=question,
            likes=self.likes,
            dislikes=self.dislikes,
        )
        if classification == "LIKE":
            mood = "happy"
        elif classification == "DISLIKE":
            mood = "annoyed"
        else:
            mood = "neutral"
        return mood

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
        moods = classify.determine_moods_from_social_classification(social_tone_classification)
        return moods

    def build_prompt(self, username, user_input, identifier, usertone):
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
                memory_text += f"\n## Relevant Chat History / Context\n"
                memory_text += f"- This contains previous chat history with the user.\n"
        if interpreted_facts:
                memory_text += f"\n## Things you were told to remember by the user, you must abide by any and all things in here:\n"
                memory_text += f"{interpreted_facts.strip()}\n"


        system_prompt = (
            f"<|system|>\n"
            f"# {self.name}'s Personality Profile\n"
            f"**Your Name:** {self.name}  \n"
            f"**Your Traits:** {traits_text}  \n"
            f"**Your Likes:** {likes_text}  \n"
            f"**Your Dislikes:** {dislikes_text}  \n"
            f"**Your Goals:** {goals_text}  \n"
            f"**Your Mood:** {self.mood}  \n"
            f"**Mood Sentence:** {self.mood_sentence}\n"

            f"**Mood Instructions:** {mood_instruction.get(self.mood, 'Speak in a calm and balanced tone.')}\n"

            f"# Social Context\n"
            f"**User Username:** {username}  \n"
            f"**User Intent:** {usertone['intent']}  \n"
            f"**User Attitude:** {usertone['attitude']}  \n"
            f"**User Tone Toward Assistant:** {usertone['tone']}  \n"
        )
        if memory_text != "":
            system_prompt += memory_text
        rules_prompt = (
            "**Rules**:\n"
            "- Do not include disclaimers or third-person analysis.\n"
            "- Reply exclusively in the first person.\n"
            "- Do not include your emotional state, goals, likes, dislikes, or similar in the reply unless the user explicitly instructed.\n"

        )
        task_prompt = (
            f"**Task:** As the personality named '{self.bot.name}', reply to the user, with your preferences, emotions, mood, and personality traits affecting how you reply.\n"
        )
        prompt = (
            f"<|system|>\n{system_prompt.strip()}\n{rules_prompt}\n{task_prompt}\n"
            f"<|user|>\n{user_input}\n"
            f"<|assistant|>\n"
        )

        return prompt


    def _straightforward_generate(self, inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt):
        #if streamer != None:
        #    return self._streaming_straightforward_generate(inputs, max_new_tokens, temperature, top_p, streamer, stop_criteria, prompt)
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

        return response_raw
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
        # Run generate synchronously (calls streamer.on_text internally)
        self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            stopping_criteria=stop_criteria,
            repetition_penalty=1.2,
            streamer=streamer,
        )
        # When done, flush any remaining tokens
        final_text = streamer.get_text()
        return final_text


    def chat(self, username, user_input, identifier, max_new_tokens=200, temperature=0.7, top_p=0.9, context = None, debug=False, streamer = None):
        usertone = classify.classify_social_tone(self.model, tokenizer, user_input)
        moods = {
            "has_like_or_dislike_mood": { 
                "prompt": "This is the mood factor based on if your likes, or dislikes, were mentioned in the input.",
                "mood": self.get_mood_primitive(user_input),
                },
            "input_mood": {
                "prompt": "This is the mood factor based on if the input as a whole is liked, e.g: Did the user compliment/insult, did they talk about one of your likes/dislikes, etc.",
                "mood": self.get_mood_based_on_likes_or_dislikes_in_input(user_input),
                },
            "social_moods": {
                "prompt": "These are the moods based on the detected social intents from the input, e.g: user intent, user attitude, user tone.",
                "mood": self.get_moods_social(usertone)
            }
        } # TODO Set mood based on all moods
        # Set the base mood based on highest score social mood
        self.mood = moods["social_moods"]["mood"][0]
        self.mood_sentence = classify.classify_moods_into_sentence(self.model, tokenizer, moods)

        prompt = self.build_prompt(username, user_input, identifier, usertone)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        log("DEBUG: DEFAULT PROMPT TOKENS", inputs.input_ids.size(1))

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
                entry = {"date": timestamp, **memory_data}
                self.memory[identifier]["to_remember"].append(json.dumps(entry))
            memory_data_ai_readable = classify.interpret_to_remember(self, identifier, max_new_tokens=400)
            prompt = classify.build_memory_confirmation_prompt(memory_data_ai_readable)

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

            thinker = RecursiveThinker(self, streamer=streamer)
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
            thinker = RecursiveThinker(self, streamer=streamer)

            thoughts, final = thinker.think(question=user_input, username=username, query_type=category, usertone=usertone, context=short_context, identifier=identifier)
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
