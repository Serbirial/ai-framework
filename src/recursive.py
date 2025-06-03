from . import classify
from log import log
from transformers import StoppingCriteriaList
from .static import mood_instruction, StopOnSpeakerChange, tokenizer

class RecursiveThinker:
    def __init__(self, bot, depth=3, streamer=None):
        self.bot = bot  # Reference to ChatBot
        self.depth = depth
        self.streamer = streamer

    def build_prompt(self, question, query_type, usertone, context=None, include_reflection=False, identifier=None):
        traits = ", ".join(self.bot.traits)
        goals = ", ".join(self.bot.goals)
        likes = ", ".join(self.bot.likes)
        dislikes = ", ".join(self.bot.dislikes)
        mood = self.bot.mood

        base = (
            f"<|system|>\n"
            f"# Personality Profile\n"
            f"**Your Name:** {self.bot.name}  \n"
            f"**Your Traits:** {traits}  \n"
            f"**Your Likes:** {likes}  \n"
            f"**Your Dislikes:** {dislikes}  \n"
            f"**Your Goals:** {goals}  \n"
            f"**Your Mood:** {mood}  \n"
            f"**Mood Sentence**: {self.bot.mood_sentence}\n"

            f"**Mood Instructions:** {mood_instruction.get(mood, 'Speak in a calm and balanced tone.')}\n"

            f"# Social Context\n"
            f"**User Intent:** {usertone['intent']}  \n"
            f"**User Attitude:** {usertone['attitude']}  \n"
            f"**User Tone Toward Assistant:** {usertone['tone']}  \n"
        )

        if context:
            base += (
                f"\n## Relevant Memory or Context\n"
                f"- Consider how this context / recent history may affect your thoughts.\n"
                f"{context.strip()}\n"
            )

        if identifier:
            interpreted_facts = classify.interpret_to_remember(self.bot, identifier)
            if interpreted_facts:
                base += (
                    f"\n## Things You Were Told to Remember\n"
                    f"{interpreted_facts.strip()}\n"
                )

        base += (
            f"\n<|user|>\n"
            f"### Reasoning Prompt\n"
            f"**Question:** {question}  \n"
            f"**Task:** As the AI personality '{self.bot.name}', consider this question carefully and reason step-by-step with your own preferences, emotions, and personality traits.  \n"
            f"Adjust your tone and manner to mirror the user's attitude and intent.\n"
            #f"_Be attentive to how this relates to your identity, preferences, mood, or values._\n"
            f"# Note: In the question and personality profile, 'you' or '{self.bot.name}' always refers to the assistant (AI Assistant), never the user, and '{self.bot.name}' will always refer to the assistant, never the user.\n" # BUG: the AI is referring to its own likes/dislikes as the users

        )

        # Add specific guidance based on query_type
        if query_type == "factual_question":
            base += (
                "\n**[Factual Question Guidance]**\n"
                "- Focus on clarity, accuracy, and logic.  \n"
                "- Prioritize objective information.  \n"
                "- Avoid emotional or opinion-based reasoning unless relevant.\n"
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

        return base

    def think(self, question, query_type, usertone, context="", include_reflection=False, identifier=None):
        prompt = self.build_prompt(question, query_type, usertone, context, include_reflection, identifier)
        full = prompt
        log("DEBUG: RECURSIVE PROMPT",full)

        for step in range(self.depth):
            full += f"<|assistant|>\n### Step {step+1}:\n"

            inputs = tokenizer(full, return_tensors="pt").to(self.bot.model.device)

            stop_criteria = StoppingCriteriaList([StopOnSpeakerChange(tokenizer, bot_name=self.bot.name)])

            response = self.bot._straightforward_generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                streamer=self.streamer,
                stop_criteria=stop_criteria,
                prompt=full
            )
            full += f"{response.strip()}\n"
            log("DEBUG: RECURSIVE THOUGHT",response.strip())
            

        final_prompt = (
            full
            + "<|user|>\n"
            + "### Final Answer\n"
            + "_Now write your final answer for the user. Use your own voice, in the first person, make sure to include anything the user explicitly asked for from your internal steps._\n"
            + "_Do not include disclaimers, third-person analysis, or mention of internal thought steps._\n"
            + "_When referencing something from your earlier thoughts, clearly restate or rephrase it so the user can understand it without seeing your internal steps._\n"
            + "_Do not refer to 'the above', 'the previous step', reference internal comments for yourself, or similar; instead, restate what you're referring to._\n"
            + "<|assistant|>\n"
        )

        log("DEBUG: FINAL RECURSIVE PROMPT",final_prompt)

        inputs = tokenizer(final_prompt, return_tensors="pt").to(self.bot.model.device)

        final_answer = self.bot._straightforward_generate(
            inputs,
            max_new_tokens=400,
            temperature=0.8,
            top_p=0.9,
            streamer=self.streamer,
            stop_criteria=stop_criteria,
            prompt=final_prompt
        ).strip()
        log("DEBUG: RECURSIVE GENERATION",final_answer)

        return full, final_answer

