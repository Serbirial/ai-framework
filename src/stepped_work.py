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


class RecursiveWork: # TODO: check during steps if total tokens are reaching token limit- if they are: summarize all steps into a numbered summary then re-build the prompt using it and start (re-using the depth limit but not step numbers)
    def __init__(self, bot, persona_prompt: str, depth=3, streamer=None):
        self.bot = bot  # Reference to ChatBot
        self.persona_prompt = persona_prompt
        self.depth = depth
        self.streamer = streamer

    def build_prompt(self, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None, extra_context=None):
        personality = bot.list_personality(identifier)

        traits = "\n- " + "\n- ".join(personality.get("traits", [])) if personality.get("traits") else "None"
        goals = "\n- " + "\n- ".join(personality.get("goals", [])) if personality.get("goals") else "None"
        likes = "\n- " + "\n- ".join(personality.get("likes", [])) if personality.get("likes") else "None"
        dislikes = "\n- " + "\n- ".join(personality.get("dislikes", [])) if personality.get("dislikes") else "None"


        mood = self.bot.mood
        persona_prompt = self.persona_prompt

        base = (
            f"<|system|>\n"
            f"You are a personality-driven assistant named {self.bot.name}.\n"
            f"# {self.bot.name}'s Personality Profile\n"
            f"{persona_prompt}\n\n"
            f"**Your Traits:** {traits}  \n"
            f"**Your Goals:** {goals}  \n"
            f"**Your Mood:** {mood}  \n"
            f"**Mood Sentence:** {self.bot.mood_sentence}\n"
            f"**Mood Instructions:** {mood_instruction.get(mood, 'Speak in a calm and balanced tone.')}\n"
            f"# Base User Info\n"
            f"**Username:** {username}  \n\n"
            
            f"# Task Completion Framework\n"
            f"You are completing a task for the user using real external tools when needed.\n"
            f"Tasks must be executed using Actions — they are not simulated, they are real code and functions.\n"

            "## Action Usage Rules"

            "- You may never perform arithmetic yourself."
            "- For *any* math expressions (even simple ones), you must use the \"execute_math\" action."
            "- Math must be executed with the following action:"
            "  - \"execute_math\": Use this to run math using +, -, *, /, %, //, or ** only."
            "    Example: <Action>{\"action\": \"execute_math\", \"parameters\": {\"expression\": \"10 * 11 + 3\"}, \"label\": \"math1\"}</Action>"

            f"## Action Execution Format\n"
            f"You may output up to THREE <Action> JSON blocks per step.\n"
            f"Each <Action> must use this format:\n"
            '<Action>{ "action": "<action_name>", "parameters": { ... }, "label": "<unique_label>" }</Action>\n'
            f"Where:\n"
            + "\n".join(
                f'  - "{k}": {v["help"]}\n'
                f'    Example: <Action>{{"action": "{k}", "parameters": {json.dumps(v["params"])}, "label": "{k}_example1"}}</Action>'
                for k, v in VALID_ACTIONS.items()
            )
            + "\n"
            
            "- Use <ActionResult<label>> results in the next step — never guess them.\n"
            "- If no action is needed, reason forward logically toward the task goal.\n"
        )


        # Get interpreted to_remember facts for the user
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM MEMORY WHERE userid = ? ORDER BY timestamp ASC",
            (identifier,)
        )
        rows = cursor.fetchall()
        conn.close()        
        
        memory_text = ""
        if context:
            memory_text += "\n## Chat History\n"
            memory_text += "- Use chat history only to understand recurring topics, context, or prior misunderstandings.\n"
            memory_text += context

        if rows:
            memory_text += "\n## **Binding Instructions / Memory:**\n"
            memory_text += "\n".join(f"- {row[0].strip()}" for row in rows)
            memory_text += "\n"

        base += memory_text
        
        base += (
            #f"\n<|user|>\n"
            f"### Task Prompt\n"
            f"**User Given Task:** {question}  \n"
            f"**Task:** As the personality named '{self.bot.name}', you are now performing a real-world task step-by-step. Use <Action> calls to interact with real tools or data sources when needed.\n"
            f"You must complete the task through actionable thinking — reasoning is encouraged, but results must come from actions and their results, not assumptions.\n"

            f"**Rules:** Only generate content for the current step. Do not generate any future step numbers. You must stop after completing the current step.\n"
            f"# Note: In the question and personality profile, 'you' or '{self.bot.name}' always refers to the named personality '{self.bot.name}' (assistant), never the user, and '{self.bot.name}' will always refer to the assistant, never the user.\n" # BUG: the AI is referring to its own likes/dislikes as the users
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
            # include prior steps content only (no "### Thought step" headers)
            if prior_steps:
                step_prompt += "**Prior Steps:**\n"
                step_prompt += "\n".join(prior_steps) + "\n\n"
            # add the current step header for clarity when doing tasks
            step_prompt += f"### Step {step+1} of {self.depth}\n"
            step_prompt += "You must either:\n"
            step_prompt += "- Emit one or more <Action> blocks to continue the task, OR\n"
            step_prompt += "- Reason forward with what has been learned to complete the task.\n"
            step_prompt += "If all needed data is already available, begin constructing your final result logic now.\n\n"
            # insert previous action result just before generation (but after thought header)
            if extra_context_lines:
                step_prompt += "\n".join(extra_context_lines) + "\n"
                extra_context_lines.clear()
                
            step_prompt += "<|assistant|>"
                
            custom_stops = [f"<|{username}|>", f"<|{self.bot.name}|>"]
            stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 
            # generate step output
            response = self.bot._straightforward_generate(
                step_prompt,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                streamer=self.streamer,
                stop_criteria=stop_criteria,
                _prompt_for_cut=step_prompt,
            )
            step_content = response.strip()
            prior_steps.append(step_content)

            full += step_content
            log(f"DEBUG: WORK STEP {step}", step_content)
            
            action_result = check_for_actions_and_run(response)
            
            # queue action result for next step input
            if action_result != "NOACTION":
                if type(action_result) == list: # multiple actions = multiple results
                    for result in action_result:
                        extra_context_lines.append(result)
                else:
                    extra_context_lines.append(action_result)


            # append the full step (header + content) to the full conversation log
            full += f"### Step {step+1} of {self.depth}\n{step_content}\n\n"

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
                    max_new_tokens=150,
                    temperature=0.8,
                    top_p=0.9,
                    streamer=self.streamer,
                    stop_criteria=stop_criteria,
                    _prompt_for_cut=step_prompt,
                )
                full += f"{response.strip()}\n"


        final_prompt = (
            full
            #+ "<|assistant|>\n"
            + "### Final Answer\n"
            + "_Now summarize your steps to the user. If the task is unfinished, explain what progress has been made and what steps remain._\n"
            + "**Rules**:\n"
            + "- When referencing something from your earlier steps, clearly restate it so the user can understand it without seeing your internalized steps.\n"
            + "- Include disclaimers when accessing the web through actions.\n"
            + "- Present the answer directly and concisely.\n"
            + "<|assistant|>\n"
        )


        tokenizer = DummyTokenizer()
        prompt_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL STEPPED WORK PROMPT:\n",final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", prompt_tokens_used)

        stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=custom_stops) 

        final_answer = self.bot._straightforward_generate(
            max_new_tokens=350, # NOTE: double for debugging, should be 400
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

