from log import log
from .static import StopOnSpeakerChange, DB_PATH
from utils.helpers import DummyTokenizer, trim_context_to_fit
import json
import sqlite3
from . import bot

from . import static_prompts
from .ai_actions import check_for_actions_and_run

class RecursiveWork: # TODO: check during steps if total tokens are reaching token limit- if they are: summarize all steps into a numbered summary then re-build the prompt using it and start (re-using the depth limit but not step numbers)
    def __init__(self, bot, config, persona_prompt: str, depth=3, streamer=None):
        self.bot = bot  # Reference to ChatBot
        self.persona_prompt = persona_prompt
        self.config = config
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
        actions_section = static_prompts.build_base_actions_prompt()
        actions_rule_section = static_prompts.build_base_actions_rule_prompt()
        actions_explanation_section =  static_prompts.build_base_actions_explanation_prompt()
        
        base = (
            #"<|begin_of_text|>"

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
            f"**{username} asked you:** {question}  \n"
            f"**Task:** As '{self.bot.name}', you are now performing a real-world task step-by-step. Use <Action> calls to interact with real tools or data sources when needed.\n"
            f"You must complete the task through actionable thinking — reasoning is encouraged, but results must come from actions and their results, not assumptions.\n"
        )

        if extra_context:
            base += f"\n<|ipython|>{extra_context}</|ipython|>\n"


        # Add specific guidance based on query_type

        log("TASK PROMPT", base)
        return base
    
    def build_final_prompt(self, identifier, username, usertone, question, steps_and_tools):
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
        discord_formatting_prompt = static_prompts.build_discord_formatting_prompt()
        
        
        base = (
            #"<|begin_of_text|>"

            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are '{self.bot.name}'.\n"

            f"{persona_section}"
            f"{user_info_section}"
            f"{memory_instructions_section}"
            f"{memory_section}"
            f"{rules_section}"
            f"{discord_formatting_prompt}"
            
            "**Task:**\n"
            f"- {username} asked you: {question}\n"
            "- Summarize your internal task steps to the user, if the task is unfinished, explain what progress has been made and what steps remain.\n\n"
            "**Rules:**\n"
            "- DO NOT mention or list raw ipython tool output unless the user explicitly asks.\n"
            "- The user cannot see your internal thoughts or steps — always restate anything important from earlier as if explaining it from scratch.\n"
            "- Explain your reasoning and what has been done so far in plain, conversational language.\n"
            "- Include clear disclaimers if your response includes web data, scraped content, or summaries from tools.\n"
            "- DO NOT execute new actions — only summarize based on information you already gathered.\n"
            "- Speak naturally in the first person, as if you’re talking directly to the user.\n"
            "- If the task is still in progress, list ONLY the next immediate steps in as few words as possible while still clearly communicating to both the user and yourself when reviewing chat history.\n"
            "- When sharing Results from a Action, restate what was found in your own words.\n"
            "- The user can only see this reply, they cant see ANY previous steps- only what you reply with below- so make sure to re-state anything when referencing steps.\n\n"
            
            f"<|eot_id|>\n" # end system prompt
            
        )
        
        
        # add the previous output (pairs of steps and their tool outputs)
        base += steps_and_tools



        log("FINAL TASK PROMPT", base)
        return base

    def think(self, question, username, query_type, usertone, tier, context=None, include_reflection=False, identifier=None):
        tokenizer = DummyTokenizer()


        prompt = self.build_prompt(question, username, query_type, usertone, context=context, include_reflection=include_reflection, identifier=identifier)

        full = f"{prompt}"
        extra_context_lines = []  # Accumulates all action results
        prior_steps = []  # to store steps to seperate them from step generation and the full prompt
        history_section = static_prompts.build_history_prompt(context)

        to_add = ""
        for step in range(self.depth):
            # start with the system prompt or base context
            step_prompt = f"{full}"

                
            step_prompt += f"### Current Step:\n"
            step_prompt += f"**Task Reminder:** {question}\n\n" # Reinforce the task every step

            step_prompt += (
                "**Step Rules:**\n"
                "- You must ONLY generate content for the **current step**.\n"
                "- You may leave yourself instructions or a plan for the *next* step, but do NOT write its contents.\n"
                "- Stay entirely within the scope of this current step, you are NOT ALLOWED to create numbered steps.\n"
                "- For *any* basic math expressions (addition, subtraction, multiplication, division, etc), you MUST use the `execute_math` action.\n"
                "- For *any* advanced calculus expressions (derivatives, integrals, limits, etc), you MUST use the `run_calculus` action.\n"
                "- For any *python code execution*, you MUST use the `run_python_sandboxed` action to safely run Python code inside a secured sandbox environment.\n"
                "- NEVER attempt to execute code directly or guess results; always rely on the sandbox’s verified output before proceeding.\n"

                #"- For *any* latex output use the `generate_latex` action to produce LaTeX from structured JSON data describing document elements.\n"
                #"- Provide parameters like type (document, section, text, equation, table, list) and related fields (content, title, text, latex, columns, rows, items, ordered).\n"

                "- You must ONLY use an <Action> if the user explicitly requested a task that requires it, or if the current step logically requires real data you cannot guess.\n"
                "- Dont use actions that wont explicitly progress to solving the user given task.\n"

                "- If no action is needed, reason forward logically toward completing the user given task.\n"
                "- Actions are expensive operations; you should avoid REPEATING an action with the SAME parameters once its result is known.\n"
                "- Use previously returned ipython tokens when available to build your reasoning.\n"
                "- Do not assume or simulate the result of an Action: Always wait for the next step before proceeding.\n"
                "- Only emit new actions when necessary.\n"
                "- Output the action first, then explain your reasoning why you called the action and how you planned to use it.\n"
                f"- You have {self.depth} steps to work through this task, you are on step {step+1}.\n"
                f"- You should actively progress every step and try to complete the task on or before step {self.depth} (step cutuff limit).\n"
                "- If the task is complete before the last step, clearly indicate so and use the remaining steps to explain, refine, and prepare for the last step.\n\n"
                #"- Do NOT output any '###' or '### Step...' headings.\n\n"

            )
            # end sys prompt
            step_prompt += "<|eot_id|>"


            # add previous steps and tool results
            step_prompt += to_add
            
            custom_stops = [f"<|{username}|>", f"<|{self.bot.name}|>"]

            stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
            step_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            # response begins here
            log(f"STEP {step+1} PROMPT\n", step_prompt)
            self.streamer.add_special(f"Moving on to step {step+1} of the task!")
            
            response = self.bot._straightforward_generate(
                step_prompt,
                max_new_tokens=self.config.token_config[tier]["WORK_MAX_TOKENS_PER_STEP"],
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
            token_window = self.config.token_config[tier]["BASE_TOKEN_WINDOW"]
            chat_window = self.config.token_config[tier]["BASE_TOKEN_WINDOW"]
            prompt_window = self.config.token_config[tier]["PROMPT_RESERVATION"]
            
            
            action_result = check_for_actions_and_run(self.bot.model, response, max_token_window=token_window, max_chat_window=chat_window, prompt_size=prompt_window)
            
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
                    f"- Reminder: The user asked: {question}\n"
                )
                step_prompt += checkpoint_prompt
                stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops)
                response = self.bot._straightforward_generate(
                    step_prompt,
                    max_new_tokens=self.config.token_config[tier]["WORK_MAX_TOKENS_PER_STEP"],
                    temperature=0.8,
                    top_p=0.9,
                    streamer=self.streamer,
                    stop_criteria=stop_criteria,
                    _prompt_for_cut=step_prompt,
                )
                to_add += "<|start_header_id|>assistant<|end_header_id|>\n"

                to_add += f"**Task Alignment Checkpoint Results:**\n{response.strip()}\n"
                to_add == "<|eot_id|>"

        final_prompt = self.build_final_prompt(identifier, username, usertone, question, steps_and_tools=to_add)
        final_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"



        tokenizer = DummyTokenizer()
        prompt_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL STEPPED WORK PROMPT:\n",final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", prompt_tokens_used)

        stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
        self.streamer.add_special(f"Finalizing the task response!")
        
        final_answer = self.bot._straightforward_generate(
            max_new_tokens=self.config.token_config[tier]["WORK_MAX_TOKENS_FINAL"], # NOTE: double for debugging, should be 400
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

