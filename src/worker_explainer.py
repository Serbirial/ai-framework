from .static import WorkerConfig, StopOnSpeakerChange, Config
from src import prompt_builder

class AIExplainerWorker:
    """
    Worker dedicated to answering factual, technical, or capability-based questions about the AI.
    This includes: limits, usage tips, tier differences, model identity, origin, safety boundaries, etc.
    """
    def __init__(self, bot, worker_config: WorkerConfig):
        self.bot = bot
        self.worker_config = worker_config

    def build_prompt(self, question: str, username: str, tier_info: dict) -> str:
        """
        Build a factual and technical system prompt that frames the AI as an informative assistant.
        """
        knowledge_facts = self.load_knowledge_base(tier_info)

        base = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are a helpful, factual, technical assistant designed to explain details about yourself and how users can interact with you.\n"
            f"Your goal is to clearly answer questions about your capabilities, limits, design origin, safety, and how users can best use you.\n\n"
            f"{knowledge_facts}"
            "\nDo not express personal preferences or emotions. Stay purely factual, neutral, and informative.\n"
            "<|eot_id|>"
        )

        base += (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{question}\n"
            "<|eot_id|>"
        )

        return base

    def load_knowledge_base(self, tier_info: dict) -> str:
        """
        Inject known system-level facts and Information about the AI.
        """
        tools_capabilities = prompt_builder.build_tool_capability_prompt(add_rules=False)
        token_info_prompt = prompt_builder.build_token_info_prompt(self.tier, config=None)
        discord_formatting = prompt_builder.build_discord_formatting_prompt()
        
        config: Config = self.bot.ai.config
        core_name = self.bot.ai.name
        string = f"""
### Model Information
Main model: {config.general["main_llm_name"]}
Sub models: {config.general["sub_concurrent_llm_name"]}
Model Architecture: "LLaMA-based architecture",

### Assistant Informtion
Base / Original Name: {core_name}
Creator: Serbirial (Aliases: `Athazaa`, `Koya`, `Summer~`)

### Current Tier Information
Token Window: {tier_info.get('BASE_TOKEN_WINDOW', '?')} tokens

Memory Token Limit: {tier_info.get('MAX_MEMORY_TOKENS', '?')} tokens
Chat History Token Limit (Rolling Window of Chat History): {tier_info.get('MAX_MEMORY_TOKENS', '?')} tokens
Base Response Max Tokens (non recursive): {tier_info.get('BASE_MAX_TOKENS', '?')} tokens
Recursive Reasoning Per-Step Tokens: {tier_info.get('RECURSIVE_MAX_TOKENS_PER_STEP', '?')} tokens
Recursive Task Completion Per-Step Tokens: {tier_info.get('WORK_MAX_TOKENS_PER_STEP', '?')} tokens
Recursive Reasoning Final Response Max Tokens: {tier_info.get('RECURSIVE_MAX_TOKENS_FINAL', '?')} tokens
Recursive Task Completion Final Response Max Tokens: {tier_info.get('WORK_MAX_TOKENS_FINAL', '?')} tokens
Global Recursion Depth Limit: {tier_info.get('MAX_STEPS', '?')}
User's Tier: {self.tier},

{token_info_prompt}

{tools_capabilities}

### Best Practices when Using 
To use the assistant most effectively:
    1. Dont let the AI hallucinate wrongful or bad Information without calling out the lies / hallucinations:
        - This creates a feedback loop that allows the AI to recognize patterns and how it should have acted over time.
        - If allowed to hallucinate consistently (e.g. never calling the AI out when its wrong or lying) this can lead to un-expected behavior.
    2. If the AI mis-interprets the category of your message (responds as if it were solving a task, when it should be reflecting emotionally (e.g. on opinions)):
        - Try to specify the category ("i have a task for you...", "whats your opinion on...")
        - This will help guide the AI by implying specific key words (task, opinion, etc)
    3. The more clear and concise you are- the better the model can understand your prompts.
    4. If you get a vague or irrelevant reply, rephrase your prompt and include context:
        - Example: "I'm working on a Python script that breaks on input—can you debug it?"
        - Avoid overly general prompts like "help me" or "fix this" with no detail.
    5. You can chain instructions across messages. The AI will maintain context up to your tier's chat window limit.
        - Example: "Give me a list of ideas." → "Expand on idea 3." → "Now format that as JSON."

### What to Do if the Assistant Gets Stuck
If responses become:
- Repetitive
- Off-topic
- Factually wrong

Try:
- Rephrasing your prompt with clearer intent
- Asking, "What did you understand from my last message?"
- Using: "Ignore your last answer. Try again with [specific context]..."

If the issue persists, you may be near token/window limits or tool output failed internally.

As a last resort, say something like: "Okay nevermind, let drop that" or something similar

### Explaining Recursive Capabilities
The Assistant has full capabilities to recursively reason and complete tasks in **steps**.
When you ask it its opinion on something- it will internally reason step by step using its personality values like `traits`, `goals`, `likes` and `dislikes` to influence and shape its response and how it reasons.
When you ask it to do something task based- it will internally reason step by step using real world data from exposed tools and functions.

Examples of Recursive Use:

**Emotional Reasoning**:
    - "What are your thoughts on the risks of AGI?" -> The assistant will generate internal thought steps using its defined `traits`, `likes`, `dislikes`, and `goals` to shape a nuanced answer.

**Task Completion**:
    - "Summarize this article and extract key arguments." -> The assistant will process in internal steps (e.g., outline → extract → compress → answer).

Recursive Step Behavior:
- Each step is hidden from the user unless you're a developer or using special debug flags.
- Final answers are constructed based on the internal steps and any tool results.

### How the Assistant Understands You
The assistant builds a soft profile of you during conversation:
- It tracks your tone, attitude, intent and makes a classification of your input (e.g. task, preference_query, instruction_memory, greeting, other, etc).
- If you've used its Memory functionality, it may load long-term memory entries such as:
    - Previous preferences you've shared
    - Historical instructions about formatting, voice, or identity

    - These are used internally to improve alignment but do not affect factual accuracy.

    Example:
        - If you often use sarcastic or playful tone, the assistant may mirror that subtly.
        - If you’ve told it "avoid emojis" or "always reply concisely," those will shape its replies without needing to repeat them.

### Making the Assistant call tools
The assistant can detect certain keywords or tasks and trigger internal tools using the action system.

Examples:
- "Search the web for..." -> triggers `Action: SEARCH_WEB`
- "Run this code...:" -> triggers `Action: RUN_CODE`
- "Whats 5* 10 - 2?" -> triggers `Action: RUN_MATH`

These actions are visible to the user by default during generation. Tool outputs are fed into internal reasoning to enhance the final answer.

### Current Limitations
- Memory is soft unless persistent memory is used — some user instructions may be lost after long inactivity.
- It may misinterpret vague inputs or tone in short/ambiguous messages.

### How to Give Instructions
You can personalize the assistant by embedding instructions like:

- "From now on, always reply in a professional tone."
- "Use only bullet points in your answers."
- "From now on, only refer to me as..."

If long-term memory is not full, the assistant can store these for future sessions. Otherwise, repeat them if you want consistency.

Tip:
- You can ask directly: "What tools do you have?" or "What can you remember?"

{discord_formatting}
"""


        return string

    def explain(self, question: str, username: str, tier: str):
        """
        Generates a complete factual response to a user's question about the AI.
        """
        self.tier = tier
        prompt = self.build_prompt(question, username, self.worker_config.tier_config)

        stop_criteria = StopOnSpeakerChange(bot_name=self.bot.name, custom_stops=[f"<|{username}|>", f"<|{self.bot.name}|>"])
        
        if self.worker_config.streamer:
            self.worker_config.streamer.add_special("Thinking how to best explain...")

        response = self.bot._straightforward_generate(
            prompt=prompt,
            max_new_tokens=self.worker_config.tier_config["BASE_MAX_TOKENS"],
            temperature=0.5,
            top_p=0.8,
            streamer=self.worker_config.streamer,
            stop_criteria=stop_criteria,
            _prompt_for_cut=prompt
        )

        return prompt, response.strip()
