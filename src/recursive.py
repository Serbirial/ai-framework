from . import classify
from log import log
from .static import mood_instruction, StopOnSpeakerChange, DB_PATH
from utils.helpers import DummyTokenizer, trim_context_to_fit
from utils.openai import translate_llama_prompt_to_chatml
import json
import sqlite3
from . import bot

class RecursiveThinker:
    def __init__(self, bot, depth=3, streamer=None):
        self.bot = bot  # Reference to ChatBot
        self.depth = depth
        self.streamer = streamer

    def build_prompt(self, question, username, query_type, usertone, context=None, include_reflection=False, identifier=None, extra_context=""):
        personality = bot.list_personality(identifier)

        traits = "\n- " + "\n- ".join(personality.get("traits", [])) if personality.get("traits") else "None"
        goals = "\n- " + "\n- ".join(personality.get("goals", [])) if personality.get("goals") else "None"
        likes = "\n- " + "\n- ".join(personality.get("likes", [])) if personality.get("likes") else "None"
        dislikes = "\n- " + "\n- ".join(personality.get("dislikes", [])) if personality.get("dislikes") else "None"


        mood = self.bot.mood

        base = (
            f"<|system|>\n"
            f"# {self.bot.name}'s Personality Profile\n"
            f"**Your Name:** {self.bot.name}  \n"
            f"**Your Traits:** {traits}  \n"
            f"**Your Likes:** {likes}  \n"
            f"**Your Dislikes:** {dislikes}  \n"
            f"**Your Goals:** {goals}  \n"
            f"**Your Mood:** {mood}  \n"
            f"**Mood Sentence:** {self.bot.mood_sentence}\n"
            f"**Mood Instructions:** {mood_instruction.get(mood, 'Speak in a calm and balanced tone.')}\n"

            f"# Social Context\n"
            f"**User Username:** {username}  \n"
            f"**User Intent:** {usertone['intent']}  \n"
            f"**User Attitude:** {usertone['attitude']}  \n"
            f"**User Tone Toward Assistant:** {usertone['tone']}  \n"
            #f"# Info"
            #"You may optionally output ONE <Action> JSON block per step:\n"
            #'<Action>{"action":"action_name","parameters":{...}}</Action>\n'
            #"If no action needed, respond with reasoning only.\n"
            #"Results of actions will be given next step in <ActionResult> block.\n"
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
            memory_text += f"\n## Relevant Chat History / Context\n"
            memory_text += f"- This contains previous chat history with the user (or users, if it's an open-ended chat).\n"
            memory_text += context

        if rows:
            memory_text += "\n## User-Stored Facts (These are things the user explicitly told you to remember. Treat them as binding instructions. You MUST obey them unless otherwise told.):\n"
            memory_text += "\n".join(f"- **{row[0].strip()}**" for row in rows)
            memory_text += "\n"
        base += memory_text
        
        base += (
            f"\n<|user|>\n"
            f"### Reasoning Prompt\n"
            f"**Question:** {question}  \n"
            f"**Task:** As the personality named '{self.bot.name}', consider this question carefully and reason step-by-step with your own preferences, emotions, and personality traits influencing your reasoning.  \n"
            f"Adjust your tone and manner to mirror the user's attitude and intent.\n"
            f"**Rules:** Only generate content for the current thought step. Do not generate content for any future thought step numbers (e.g., Step 2, Step 3). You must stop after completing the current step.\n"
            #f"_Be attentive to how this relates to your identity, preferences, mood, or values._\n"
            f"# Note: In the question and personality profile, 'you' or '{self.bot.name}' always refers to the named personality '{self.bot.name}' (assistant), never the user, and '{self.bot.name}' will always refer to the assistant, never the user.\n" # BUG: the AI is referring to its own likes/dislikes as the users
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
        base_prompt = self.build_prompt(question, username, query_type, usertone, context="", include_reflection=include_reflection, identifier=identifier)

        # Token-safe context trimming
        context_lines = context.split("\n") if isinstance(context, str) else context
        trimmed_context = trim_context_to_fit(base_prompt, context_lines, max_ctx=self.bot.model.n_ctx(), reserved_for_output=400)

        # Build full prompt using the trimmed context
        prompt = self.build_prompt(question, username, query_type, usertone, context=trimmed_context, include_reflection=include_reflection, identifier=identifier)
        full = prompt
        extra_context_lines = []  # Accumulates all action results

        for step in range(self.depth):
            full += (
                f"<|assistant|>\n### Thought step {step+1}:\n"
            )

            if extra_context_lines:
                for result_line in extra_context_lines:
                    full += f"{result_line}\n"
            # convert it over 
            #full = translate_llama_prompt_to_chatml(full)

            stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name)
            response = self.bot._straightforward_generate(
                full,
                max_new_tokens=120,
                temperature=0.8,
                top_p=0.9,
                streamer=self.streamer,
                stop_criteria=stop_criteria,
                _prompt_for_cut=full
            )

            log("DEBUG: THOUGHT STEP", response.strip())

            #lines = response.strip().splitlines()
            #new_lines = []

            #for line in lines:
            #    if line.strip().lower().startswith("action:"):
            #        action_key = line.split(":", 1)[1].strip().lower()
            #        result = self.perform_action(action_key)
            #        result_json = json.dumps(result)
            #        log(f"DEBUG: Action result for '{action_key}': {result_json}")
            #        action_result_str = f"<ActionResult>{result_json}</ActionResult>"
            #        extra_context_lines.append(action_result_str)
            #        new_lines.append(action_result_str)
            #    else:
            #        new_lines.append(line)

            #response = "\n".join(new_lines)
            full += f"{response.strip()}\n"



        if query_type == "factual_question":
            final_prompt = (
                full
                + "<|user|>\n"
                + "### Final Answer\n"
                + "_Now write your reply to the question using your previous thought steps and any action results to guide your answer._\n"
                + "**Rules**:\n"
                + "- When referencing something from your earlier thought steps, clearly restate or rephrase it so the user can understand it without seeing your thought steps."
                + "- Do not include disclaimers.\n"
                + "- Provide only the direct answer or requested code snippet in your own voice, in the first person.\n"
                + "- Present the answer directly and concisely in plain text or code as appropriate.\n"
                + "- If the user asks for code, you must make sure the requested code ends up in your final answer reply, the user cannot see your internal thought steps and will not be able to see any generated code from them"
                + "<|assistant|>\n"
            )
        else:
            final_prompt = (
                full
                + "<|user|>\n"
                + "### Final Answer\n"
                + "_Now write your final answer to reply to the question using your previous thought steps and any action results to guide your answer. Use your own voice, in the first person, make sure to include anything the user explicitly asked for in your answer._\n"
                + "**Rules**:\n"

                + "- Avoid including numbered steps or markdown titles in the final answer.\n"
                + "- Do not include disclaimers or third-person analysis.\n"
                + "- When referencing something from your earlier thought steps, clearly restate or rephrase it so the user can understand it without seeing your thought steps.\n"
                + "- Do not refer to 'the above', 'the previous step', reference internal comments for yourself, or similar; instead, restate what you're referring to.\n"
                + "<|assistant|>\n"
            )

        tokenizer = DummyTokenizer()
        prompt_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL RECURSIVE PROMPT:\n",final_prompt)
        log(f"\nDEBUG: FINAL PROMPT TOKENS", prompt_tokens_used)


        final_answer = self.bot._straightforward_generate(
            final_prompt,
            max_new_tokens=400, # NOTE: double for debugging, should be 400
            temperature=0.7, # lower creativity when summarizing the internal thoughts
            top_p=0.9,
            streamer=self.streamer,
            stop_criteria=stop_criteria,
            prompt=final_prompt
        ).strip()
        log("DEBUG: RECURSIVE GENERATION",final_answer)
        final_tokens_used = tokenizer.count_tokens(final_prompt)

        log(f"DEBUG: FINAL TOKEN SIZE: {final_tokens_used}")


        return full, final_answer

