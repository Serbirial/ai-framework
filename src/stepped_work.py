from . import classify
from log import log
from .static import mood_instruction, StopOnSpeakerChange, DB_PATH, CUSTOM_GPT2
from utils.helpers import DummyTokenizer, trim_context_to_fit
from utils.openai import translate_llama_prompt_to_chatml
import json
import sqlite3
import tiny_prompts, custom_gpt2_prompts
from . import bot
import re
from .static import WORK_MAX_TOKENS_FINAL, WORK_MAX_TOKENS_PER_STEP

from ai_tools import VALID_ACTIONS
from . import static_prompts
from .ai_actions import check_for_actions_and_run

class RecursiveWork: # TODO: check during steps if total tokens are reaching token limit- if they are: summarize all steps into a numbered summary then re-build the prompt using it and start (re-using the depth limit but not step numbers)
    def __init__(self, bot, persona_prompt: str, depth=3, streamer=None):
        self.bot = bot  # Reference to ChatBot
        self.persona_prompt = persona_prompt
        self.depth = depth
        self.streamer = streamer

    def build_prompt(self, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None, extra_context=None):
        personality = bot.list_personality(identifier)

        persona_prompt = self.persona_prompt
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()  
        user_info_section = static_prompts.build_user_profile_prompt(username, usertone)
        persona_section = static_prompts.build_base_personality_profile_prompt(self.bot.name, persona_prompt, personality, self.bot.mood, self.bot.mood_sentence)
        rules_section = static_prompts.build_rules_prompt(self.bot.name, username, None)
        memory_instructions_section = static_prompts.build_memory_instructions_prompt()
        memory_section =  static_prompts.build_core_memory_prompt(rows if rows else None)
        history_section = static_prompts.build_history_prompt(context)
        actions_section = static_prompts.build_base_actions_prompt()
        actions_rule_section = static_prompts.build_base_actions_rule_prompt()
        actions_explanation_section =  static_prompts.build_base_actions_explanation_prompt()
        
        base = (
            f"<|system|>\n"
            f"You are a personality-driven assistant named {self.bot.name}.\n"

            f"{persona_section}"
            
            f"{user_info_section}"
            
            f"{memory_instructions_section}"
            
            f"{memory_section}"
            
            f"{history_section}"

            f"{rules_section}"


            f"### Task Completion Framework\n"
            f"You are completing a task for the user using real external tools when needed.\n"
            f"Tasks must be executed using Actions — they are not simulated, they are real code and functions.\n"

            f"{actions_section}"
            f"{actions_explanation_section}"
            f"{actions_rule_section}"
        )

        
        
        base += (
            f"### Task Info\n"
            f"**User Given Task:** {question}  \n"
            f"**Task:** As the personality named '{self.bot.name}', you are now performing a real-world task step-by-step. Use <Action> calls to interact with real tools or data sources when needed.\n"
            f"You must complete the task through actionable thinking — reasoning is encouraged, but results must come from actions and their results, not assumptions.\n"

        )
        if extra_context:
            base += f"\n<ActionResult>{extra_context}</ActionResult>\n"
        
        # Add specific guidance based on query_type

        log("INSTRUCT PROMPT", base)
        return base

    def think(self, question, username, query_type, usertone, context="", include_reflection=False, identifier=None):
        tokenizer = DummyTokenizer()

        # Build base prompt first (before context)

        base_prompt = self.build_prompt(question, username, query_type, usertone, context="", include_reflection=include_reflection, identifier=identifier)

        # Token-safe context trimming
        context_lines = context.split("\n") if isinstance(context, str) else context
        trimmed_context = trim_context_to_fit(base_prompt, context_lines, max_ctx=self.bot.model.n_ctx(), reserved_for_output=400)

        # Build full prompt using the trimmed context

        prompt = self.build_prompt(question, username, query_type, usertone, context=trimmed_context, include_reflection=include_reflection, identifier=identifier)

        full = f"{prompt}" 
        extra_context_lines = []  # Accumulates all action results
        prior_steps = []  # to store steps to seperate them from step generation and the full prompt

    
        for step in range(self.depth):
            # start with the system prompt or base context
            step_prompt = f"{full}"


            if extra_context_lines:
                step_prompt += "### Previous Steps <ActionResult> blocks:\n"
                step_prompt += "\n".join(extra_context_lines) + "\n"
                extra_context_lines.clear()
                
            step_prompt += f"### Current Step:\n"

            step_prompt += (
                "**Step Rules:**\n"
                "- You must ONLY generate content for the **current step**.\n"
                f"- DO NOT generate or refer to future steps (like Step {step+2} or Step {step+3}).\n"
                "- You may leave yourself instructions or a plan for the *next* step, but do NOT write its contents.\n"
                "- Do NOT anticipate or simulate later outputs. Stay entirely within the scope of this one step.\n"
                "- For *any* math expressions (even simple ones), you MUST use the `execute_math` action.\n"
                "- Actions must be emitted using this exact format:\n"
                f'  <Action>{{"action": "execute_math", "parameters": {{"expression": "5 * 20 + 3"}}, "label": "math{step+1}"}}</Action>\n'
                "- Do NOT simulate or guess action results — only use <ActionResult> from Previous Steps <ActionResult> blocks.\n"
                "- If no action is needed, reason forward logically toward completing the task.\n"
                "- Actions are expensive operations; you should avoid REPEATING an action with the SAME parameters once its result is known.\n"
                "- Use previously returned <ActionResult> values when available to build your reasoning.\n"
                "- You MUST WAIT for the real <ActionResult> of any actions emitted in the current step to be provided in the next step’s prompt.\n"
                "- Only emit new actions when necessary.\n"
                "- Output the action first, then optionally explain your reasoning.\n"
            )
            step_prompt += "<|assistant|>"
                
            custom_stops = [f"<|{username}|>", f"<|{self.bot.name}|>"]
            stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
            # generate step output
            response = self.bot._straightforward_generate(
                step_prompt,
                max_new_tokens=WORK_MAX_TOKENS_PER_STEP,
                temperature=0.7,
                top_p=0.9,
                streamer=self.streamer,
                stop_criteria=stop_criteria,
                _prompt_for_cut=step_prompt,
            )
            step_content = response.strip()
            prior_steps.append(step_content)

            log(f"DEBUG: WORK STEP {step}", step_content)
            
            # append the full step (header + content) to the full conversation log
            full += f"### Step {step+1} of {self.depth}\n{step_content}\n\n"
            
            # Check for and run any actions
            action_result = check_for_actions_and_run(response)
            
            # queue action result for next step input
            if action_result != "NOACTION":
                if type(action_result) == list: # multiple actions = multiple results
                    for result in action_result:
                        extra_context_lines.append(result)
                        full += f"\n{result}" # add result to full prompt
                else:
                    extra_context_lines.append(action_result)
                    full += f"\n{action_result}" # add result to full prompt

            # Your checkpoint logic remains the same
            if step != 0 and step % 5 == 0 and step != self.depth - 1:
                # add checkpoint step_prompt
                checkpoint_prompt = (
                    "**Task Alignment Checkpoint:**\n"
                    "- Reflect on your progress so far.\n"
                    "- Ask: Are your steps clearly building toward completing the task?\n"
                    "- Briefly summarize what you've accomplished and what remains.\n"
                    "- Then continue with the next step, staying focused.\n"
                    f"- Reminder: Your task is to do step-by-step processing and calling <Actions> to complete the task as needed as the personality '{self.bot.name}'.\n"
                )
                step_prompt += checkpoint_prompt
                stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
                response = self.bot._straightforward_generate(
                    step_prompt,
                    max_new_tokens=WORK_MAX_TOKENS_PER_STEP,
                    temperature=0.8,
                    top_p=0.9,
                    streamer=self.streamer,
                    stop_criteria=stop_criteria,
                    _prompt_for_cut=step_prompt,
                )
                full += f"**Task Alignment Checkpoint Results:**\n{response.strip()}\n"

        discord_formatting_prompt = static_prompts.build_discord_formatting_prompt()
        final_prompt = (
            full
            #+ "<|assistant|>\n"
            + "### Final Answer\n"
            + "_Now summarize your steps to the user. If the task is unfinished, explain what progress has been made and what steps remain._\n"
            + "**Rules**:\n"
            + "- When referencing something from your earlier steps, clearly restate it so the user can understand it without seeing your internalized steps.\n"
            + "- Include disclaimers when accessing the web through actions.\n"
            + "- Do not use Actions, you are no longer allowed to execute actions, only use previous data.\n"
            
            + "- Present the answer directly and concisely.\n"
            + discord_formatting_prompt
            + "<|assistant|>\n"
        )


        tokenizer = DummyTokenizer()
        prompt_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL STEPPED WORK PROMPT:\n",final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", prompt_tokens_used)

        stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 

        final_answer = self.bot._straightforward_generate(
            max_new_tokens=WORK_MAX_TOKENS_FINAL, # NOTE: double for debugging, should be 400
            temperature=0.7, # lower creativity when summarizing the internal thoughts
            top_p=0.9,
            streamer=self.streamer,
            stop_criteria=stop_criteria,
            prompt=final_prompt,
            _prompt_for_cut=final_prompt
        ).strip()
        log("\n\nDEBUG: STEPPED WORK",final_answer)
        final_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL TOKEN SIZE:", final_tokens_used)


        return final_prompt, final_answer

