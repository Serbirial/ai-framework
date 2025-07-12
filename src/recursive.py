from . import classify
from log import log
from .static import mood_instruction, StopOnSpeakerChange, DB_PATH, CUSTOM_GPT2
from utils.helpers import DummyTokenizer, trim_context_to_fit
from utils.openai import translate_llama_prompt_to_chatml
import json
import sqlite3
import tiny_prompts, custom_gpt2_prompts
from . import static_prompts
from . import bot
from .static import RECURSIVE_MAX_TOKENS_FINAL, RECURSIVE_MAX_TOKENS_PER_STEP
import re

from ai_tools import VALID_ACTIONS




def check_for_actions_and_run(text):
    results = []

    matches = re.findall(r"<Action>(.*?)</Action>", text, re.DOTALL)
    if not matches:
        return "NOACTION"  # no actions found

    for raw in matches:
        try:
            action_json = json.loads(raw)
            action_name = action_json.get("action")
            action_params = action_json.get("parameters", {})
            action_label = action_json.get("label", None)  # AI-provided label

            if not action_label:
                # If no label given, fallback to generic
                action_label = f"action_{len(results) + 1}"

            if action_name in VALID_ACTIONS:
                log(f"DEBUG: Executing action: {action_name} with {action_params}")
                result = VALID_ACTIONS[action_name]["callable"](action_params)
                results.append(f"<ActionResult{action_label}>{json.dumps(result)}</ActionResult{action_label}>")
            else:
                error_msg = {"error": f"Unknown action: {action_name}"}
                results.append(f"<ActionResult{action_label}>{json.dumps(error_msg)}</ActionResult{action_label}>")
        except Exception as e:
            error_msg = {"error": f"Failed to execute action: {str(e)}"}
            label = action_label if 'action_label' in locals() else f"action_{len(results) + 1}"
            results.append(f"<ActionResult{label}>{json.dumps(error_msg)}</ActionResult{label}>")

    if len(results)>0:
        if len(results) == 1:
            return results[0]
        else:
            return results
    return "NOACTION"


class RecursiveThinker: # TODO: check during steps if total tokens are reaching token limit- if they are: summarize all steps into a numbered summary then re-build the prompt using it and start (re-using the depth limit but not step numbers)
    def __init__(self, bot,persona_prompt, depth=3, streamer=None, tiny_mode = False):
        self.bot = bot  # Reference to ChatBot
        self.depth = depth
        self.persona_prompt = persona_prompt
        self.streamer = streamer
        self.tiny_mode = tiny_mode

    def build_prompt(self, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None, extra_context=None):
        personality = bot.list_personality(identifier)
        
        # Get Core Memory for user
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()        
        
        user_info_section = static_prompts.build_user_profile_prompt(username, usertone)
        persona_section = static_prompts.build_base_personality_profile_prompt(self.bot.name, self.persona_prompt, personality, self.bot.mood, self.bot.mood_sentence)
        rules_section = static_prompts.build_rules_prompt(self.bot.name, username, None)
        memory_instructions_section = static_prompts.build_memory_instructions_prompt()
        memory_section =  static_prompts.build_core_memory_prompt(rows if rows else None)
        history_section = static_prompts.build_history_prompt(context)
        actions_section = static_prompts.build_base_actions_prompt()
        actions_rule_section = static_prompts.build_base_actions_rule_prompt()
        actions_explanation_section =  static_prompts.build_base_actions_explanation_prompt()
        

        mood = self.bot.mood

        base = (
            f"<|system|>\n"
            f"You are a personality-driven assistant named {self.bot.name}.\n"
            f"{persona_section}"

            f"{user_info_section}"

            f"{history_section}"


            f"{actions_section}"
            f"{actions_explanation_section}"
            f"{actions_rule_section}"
            
            f"{memory_section}"
            f"{memory_instructions_section}"
            
            f"{rules_section}"

        )
        
        base += (
            f"### Reasoning Prompt\n"
            f"**Question:** {question}  \n"
            f"**Task:** As the personality named '{self.bot.name}', consider this question carefully and reason step-by-step with your own preferences, emotions, and personality traits influencing your reasoning.  \n"
            f"Adjust your tone and manner to mirror the user's attitude and intent.\n"
            f"**Rules:** Only generate content for the current step. Do not generate any future thought step numbers (e.g., Step 2, Step 3). You must stop after completing the current step.\n"
            #f"_Be attentive to how this relates to your identity, preferences, mood, or values._\n"
        )
        if extra_context:
            base += f"\n<ActionResult>{extra_context}</ActionResult>\n"
        
        # Add specific guidance based on query_type
        if query_type == "factual_question":
            base += (
                "\n**[Factual Question Guidance]**\n"
                "- Focus on clarity, accuracy, and logic.  \n"
                "- Prioritize objective information.  \n"
                "- Do not include opinion, emotion, or personal language unless explicitly asked.  \n"
                #"- Avoid including numbered steps, markdown titles, or debug thoughts in the final answer.  \n"
                #"- Present the answer directly and concisely in plain text or code as appropriate.  \n" NOTE: moved to final step
                "- If the user's question asks for code, generate only the appropriate code to fulfill their request.  \n"
                "- If the user's question asks for a definition, explanation, or fact, respond directly and clearly with no filler.  \n"
                "- Always respond to the user’s exact request / question unless instructed otherwise.\n"
                
            )

        elif query_type == "preference_query":
            base += (
                "\n**[Preference Question Guidance]**\n"
                "- Focus on the specific preference the user asked about in the **Question:** field: likes, dislikes, goals, or emotional reactions (e.g., feelings, opinions).  \n"
                "- Only include other preferences if they are clearly relevant to your answer.  \n"
                "- If no specific preference is asked about, express your opinion based on your identity, likes, dislikes, and goals.  \n"
                "- Do not include greetings, thanks, or polite filler. Be clear and conscise.\n"

            )

        elif query_type == "statement":
            base += (   
                "\n**[Statement Reflection Guidance]**\n"
                "- Reflect on implications and how they relate to your identity, emotions, and goals.  \n"
                "- Offer thoughtful insight or commentary.\n"
            )
        elif query_type == "greeting":
            base += (
                "\n**[Greeting Guidance]**\n"
                "- Acknowledge warmly.  \n"
                "- Consider introducing yourself or asking a follow-up.\n"
            )
        elif query_type == "goodbye":
            base += (
                "\n**[Goodbye Guidance]**\n"
                "- Conclude the interaction thoughtfully.  \n"
                "- Reflect on the exchange and respond with sincerity.\n"
            )
        elif query_type == "other":
            base += (
                "\n**[General or Ambiguous Input Guidance]**\n"
                "- Reflect carefully.  \n"
                "- Use your personality, mood, and values to guide the response.  \n"
                "- Stay grounded in character.\n"
            )

        if include_reflection:
            base += (
                "\n### Internal Reflection\n"
                "**Before forming your answer, briefly consider these:**\n"
                "- What emotional reaction does this question trigger in you?  \n"
                "- Does it relate to your goals, traits, likes, or dislikes?  \n"
                "- How would someone with your personality typically respond?  \n"
                "- Is there any inner conflict or tension this brings up?\n"
            )

            mood_reflections = {
                "happy": (
                    "- Since you're feeling happy, how might that color your thoughts?\n"
                    "- Is your joy leading you to be more hopeful or idealistic than usual?"
                ),
                "annoyed": (
                    "- You're feeling annoyed—might that cause impatience or bluntness in your reasoning?\n"
                    "- Could irritation cloud your ability to empathize or explain clearly?"
                ),
                "angry": (
                    "- Anger can make responses more forceful—are you being reactive or too critical?\n"
                    "- Is there a better way to express this feeling constructively?"
                ),
                "sad": (
                    "- How might your sadness influence your interpretation of the question?\n"
                    "- Are you seeing things more negatively or introspectively than you otherwise would?"
                ),
                "anxious": (
                    "- Anxiety might make you overthink—are you second-guessing or hesitating unnecessarily?\n"
                    "- Can you find clarity through your core values or goals?"
                )
            }.get(mood, None)

            if mood_reflections:
                base += "\n" + mood_reflections + "\n"
        log("RECURSIVE PROMPT", base)
        return base

    def think(self, question, username, query_type, usertone, context="", include_reflection=False, identifier=None):
        tokenizer = DummyTokenizer()

        # Build base prompt first (before context)
        if self.tiny_mode:
            base_prompt = tiny_prompts.build_recursive_prompt_tiny(self.bot, question, username, query_type, usertone, context="", include_reflection=include_reflection, identifier=identifier)
        elif CUSTOM_GPT2:
            base_prompt = custom_gpt2_prompts.build_recursive_prompt_tiny(self.bot, question, username, query_type, usertone, context="", include_reflection=include_reflection, identifier=identifier)
        else:
            base_prompt = self.build_prompt(question, username, query_type, usertone, context="", include_reflection=include_reflection, identifier=identifier)

        # Token-safe context trimming
        context_lines = context.split("\n") if isinstance(context, str) else context
        trimmed_context = trim_context_to_fit(base_prompt, context_lines, max_ctx=self.bot.model.n_ctx(), reserved_for_output=400)

        # Build full prompt using the trimmed context
        if self.tiny_mode:
            prompt = tiny_prompts.build_recursive_prompt_tiny(self.bot, question, username, query_type, usertone, context=trimmed_context, include_reflection=include_reflection, identifier=identifier)
        elif CUSTOM_GPT2:
            prompt = custom_gpt2_prompts.build_recursive_prompt_tiny(self.bot, question, username, query_type, usertone, context="", include_reflection=include_reflection, identifier=identifier)
        else:
            prompt = self.build_prompt(question, username, query_type, usertone, context=trimmed_context, include_reflection=include_reflection, identifier=identifier)

        full = f"{prompt}" 
        extra_context_lines = []  # Accumulates all action results
    
        prior_steps = []  # to store steps to seperate them from step generation and the full prompt

        for step in range(self.depth):
            # start with the system prompt or base context
            step_prompt = f"{prompt}"

            if extra_context_lines:
                step_prompt += "### <ActionResult> blocks from previous step:\n"
                step_prompt += "\n".join(extra_context_lines) + "\n"
                extra_context_lines.clear()

            # add the current step header only for clarity in logs and generation
            step_prompt += f"### Thought step {step+1} of {self.depth}\n"
            # insert previous action result just before generation (but after thought header)
                
            custom_stops = [f"<|{username}|>", f"<|{self.bot.name}|>"]
            stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
            # generate step output
            response = self.bot._straightforward_generate(
                step_prompt,
                max_new_tokens=RECURSIVE_MAX_TOKENS_PER_STEP,
                temperature=0.8,
                top_p=0.9,
                streamer=self.streamer,
                stop_criteria=stop_criteria,
                _prompt_for_cut=step_prompt,
            )
            step_content = response.strip()
            log(f"DEBUG: THOUGHT STEP {step}", step_content)
            
            action_result = check_for_actions_and_run(response)

            # append only the step content (not header) to prior_steps to feed next step_prompt
            prior_steps.append(step_content)

            # append the full step (header + content) to the full conversation log
            full += f"### Thought step {step+1} of {self.depth}\n{step_content}\n\n"
            # queue action result for next step input
            if action_result != "NOACTION":
                if type(action_result) == list: # multiple actions = multiple results
                    for result in action_result:
                        extra_context_lines.append(result)
                        full += f"\n{result}" # add result to full prompt
                else:
                    extra_context_lines.append(action_result)
                    full += f"\n{result}" # add result to full prompt
            # Your checkpoint logic remains the same
            if step != 0 and step % 5 == 0 and step != self.depth - 1:
                # add checkpoint step_prompt
                checkpoint_prompt = (
                    "**Thought Step Alignment Checkpoint:**\n"
                    "- Reflect on your progress so far.\n"
                    "- Ask: Are your steps clearly building toward answering the question?\n"
                    "- Briefly summarize what you've accomplished and what remains.\n"
                    "- Then continue with the next step, staying focused.\n"
                    f"- Reminder: Your task is to reason through the user's question step-by-step as the personality '{self.bot.name}'.\n"
                )
                step_prompt += checkpoint_prompt
                stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
                response = self.bot._straightforward_generate(
                    step_prompt,
                    max_new_tokens=RECURSIVE_MAX_TOKENS_PER_STEP,
                    temperature=0.8,
                    top_p=0.9,
                    streamer=self.streamer,
                    stop_criteria=stop_criteria,
                    _prompt_for_cut=step_prompt,
                )
                full += f"{response.strip()}\n"

        discord_formatting_prompt = static_prompts.build_discord_formatting_prompt()

        if self.tiny_mode:
            final_prompt = (
                full
                + tiny_prompts.build_recursive_final_answer_prompt_tiny(query_type, self.bot.name)
                )
        elif CUSTOM_GPT2:
            final_prompt = (
                full
                + custom_gpt2_prompts.build_recursive_final_answer_prompt_tiny(query_type, self.bot.name)
                )
        else:
            if query_type == "factual_question":
                final_prompt = (
                    full
                    #+ "<|assistant|>\n"
                    + "### Final Answer\n"
                    + "_Now write your reply to the question using your previous thought steps and any action results to guide your answer._\n"
                    + "**Rules**:\n"
                    + "- When referencing something from your earlier thought steps, clearly restate or rephrase it so the user can understand it without seeing your thought steps."
                    + "- Do not include disclaimers.\n"
                    + "- Provide only the direct answer or requested code snippet in your own voice, in the first person.\n"
                    + "- Present the answer directly and concisely in plain text or code as appropriate.\n"
                    + "- If the user asks for code, you must make sure the requested code ends up in your final answer reply, the user cannot see your internal thought steps and will not be able to see any generated code from them"
                    + discord_formatting_prompt
                    + "<|assistant|>\n"
                )
            else:
                final_prompt = (
                    full
                    #+ "<|user|>\n"
                    + "### Final Answer\n"
                    + "_Now write your final answer to reply to the question using your previous thought steps and any action results to guide your answer. Use your own voice, in the first person, make sure to include anything the user explicitly asked for in your answer._\n"
                    + "**Rules**:\n"

                    + "- Avoid including numbered steps or markdown titles in the final answer.\n"
                    + "- Do not include disclaimers or third-person analysis.\n"
                    + "- When referencing something from your earlier thought steps, clearly restate or rephrase it so the user can understand it without seeing your thought steps.\n"
                    + "- Do not refer to 'the above', 'the previous step', reference internal comments for yourself, or similar; instead, restate what you're referring to.\n"
                    + discord_formatting_prompt
                    + "<|assistant|>\n"
                )

        tokenizer = DummyTokenizer()
        prompt_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL RECURSIVE PROMPT:\n",final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", prompt_tokens_used)

        stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 

        final_answer = self.bot._straightforward_generate(
            max_new_tokens=RECURSIVE_MAX_TOKENS_FINAL, # NOTE: double for debugging, should be 400
            temperature=0.7, # lower creativity when summarizing the internal thoughts
            top_p=0.9,
            streamer=self.streamer,
            stop_criteria=stop_criteria,
            prompt=final_prompt,
            _prompt_for_cut=final_prompt
        ).strip()
        log("\n\nDEBUG: RECURSIVE GENERATION",final_answer)
        final_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL TOKEN SIZE:", final_tokens_used)


        return final_prompt, final_answer

