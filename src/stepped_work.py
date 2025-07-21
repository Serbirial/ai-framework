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
            #"<|begin_of_text|>"
            f"{history_section}" # low priority for large chat history

            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are a personality-driven assistant named {self.bot.name}.\n"

            f"{persona_section}"
            f"{user_info_section}"
            f"{memory_instructions_section}"
            f"{memory_section}"
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

    def think(self, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None):
        tokenizer = DummyTokenizer()


        prompt = self.build_prompt(question, username, query_type, usertone, context=context, include_reflection=include_reflection, identifier=identifier)

        full = f"{prompt}"
        extra_context_lines = []  # Accumulates all action results
        prior_steps = []  # to store steps to seperate them from step generation and the full prompt

        to_add = ""
        for step in range(self.depth):
            # start with the system prompt or base context
            step_prompt = f"{full}"

                
            step_prompt += f"### Current Step:\n"
            step_prompt += f"**User Given Task:**\n    - {question}\n" # Reinforce the task every step

            step_prompt += (
                "**Step Rules:**\n"
                "- You must ONLY generate content for the **current step**.\n"
                "- You may leave yourself instructions or a plan for the *next* step, but do NOT write its contents.\n"
                "- Stay entirely within the scope of this current step, you are NOT ALLOWED to create numbered steps.\n"
                "- For *any* basic math expressions (addition, subtraction, multiplication, division, etc), you MUST use the `execute_math` action.\n"
                "- For *any* advanced calculus expressions (derivatives, integrals, limits, etc), you MUST use the `run_calculus` action.\n"
                "- For any *python code execution*, you MUST use the `run_python_sandboxed` action to safely run Python code inside a secured sandbox environment.\n"
                "- Do NOT attempt to execute code directly or guess results; always rely on the sandbox’s verified output before proceeding.\n"
                "- The sandbox has the limitations of 400M RAM, 60 seconds runtime, and no network access — you must pass fully formatted, valid Python code as a string, with newlines separating code lines.\n"

                #"- For *any* latex output use the `generate_latex` action to produce LaTeX from structured JSON data describing document elements.\n"
                #"- Provide parameters like type (document, section, text, equation, table, list) and related fields (content, title, text, latex, columns, rows, items, ordered).\n"

                "- You must ONLY use an <Action> if the user explicitly requested a task that requires it, or if the current reasoning step logically requires real data you cannot guess.\n"
                "- NEVER guess or assume an action is needed unless it is required to get real data.\n"

                "- If no action is needed, reason forward logically toward completing the task.\n"
                "- Actions are expensive operations; you should avoid REPEATING an action with the SAME parameters once its result is known.\n"
                "- Use previously returned <ActionResult> values when available to build your reasoning.\n"
                "- Do not assume or simulate the result of an Action: Always wait for the actual <ActionResult> to be returned in the next step before proceeding.\n"
                "- Only emit new actions when necessary.\n"
                "- Output the action first, then explain your reasoning why you called the action and how you planned to use it.\n"
                f"- You have {self.depth} steps to work through this task, you are on step {step+1}.\n"
                f"- You should actively progress every step and try to complete the task on or before step {self.depth} (step cutuff limit).\n"
                "- If the task is complete before the last step, clearly indicate so and use the remaining steps to explain, refine, and prepare the final summary.\n\n"
                "- Do NOT include any '###' or '### Step...' headings or numbering in your response; output only the content for the current step.\n"

            )
            # end sys prompt
            step_prompt += "<|eot_id|>"

            # add previous steps and tool results
            step_prompt += to_add
            
            custom_stops = [f"<|{username}|>", f"<|{self.bot.name}|>"]

            stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
            step_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            # response begins here

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
            clean_step_content = step_content.replace("<|begin_of_text|>", "").strip()

            prior_steps.append(clean_step_content)

            log(f"DEBUG: WORK STEP {step}", clean_step_content)
            to_add += "<|start_header_id|>assistant<|end_header_id|>\n"

            # append the full step (header + content) to the full conversation log
            to_add += f"### Step {step+1} of {self.depth}:\n{clean_step_content}\n"
            to_add == "<|eot_id|>\n"
            
            # Check for and run any actions
            action_result = check_for_actions_and_run(self.bot.model, response)
            
            # queue action result for next step input
            if action_result != "NOACTION":
                if type(action_result) == list: # multiple actions = multiple results
                    for result in action_result:
                        extra_context_lines.append(result)
                        to_add += f"{result}\n" # add result to full prompt
                else:
                    extra_context_lines.append(action_result)
                    to_add += f"{action_result}\n" # add result to full prompt
                to_add += "\n"

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
                to_add += "<|start_header_id|>assistant<|end_header_id|>\n"

                to_add += f"**Task Alignment Checkpoint Results:**\n{response.strip()}\n"
                to_add == "<|eot_id|>"

        discord_formatting_prompt = static_prompts.build_discord_formatting_prompt()
        final_prompt = (
            full
            + discord_formatting_prompt
            + f"### Responding to {username}\n"
            + "**Task:** Now summarize to the user. If the task is unfinished, explain what progress has been made and what steps remain.\n\n"
            + "**Rules**:\n"
            + "- Do NOT mention or list internal step names, action calls, or raw execution details unless asked.\n"

            + "- When referencing something from your earlier steps, clearly restate it so the user can understand it without seeing your internalized steps.\n"
            + "- Include disclaimers when accessing the web through actions.\n"
            + "- You may NOT execute any new actions — only use previously obtained data.\n"
            + "- Present the answer directly and concisely — speak in the first person as if you are directly replying to the user.\n"
            + "- If the task is complete, clearly state it and provide a helpful concluding summary.\n"
            + "- If more steps remain, clearly list only the next immediate steps without excess detail.\n"
            + "- When presenting results from any external tool or action, explain them clearly and conversationally without mentioning internal commands or raw data; focus on making the information accessible and helpful to the user.\n\n"

            + to_add # add the steps

            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

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

