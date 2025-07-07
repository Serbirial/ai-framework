import random
import discord
import asyncio
from concurrent.futures import ProcessPoolExecutor
import re
import aiohttp
from src import bot
from src import static

import asyncio

import time

async def generate_and_stream(self, message, processed_input, history):
    streammsg = await message.reply("Generating...")

    streamer = object()
    
    # Run generation in thread
    loop = asyncio.get_running_loop()
    gen_task = loop.run_in_executor(
        None,
        lambda: self.ai.chat(
            username=message.author.display_name,
            user_input=processed_input,
            identifier=message.guild.id,
            context=history,
            debug=False,
            streamer=streamer
        )
    )

    # While the generation runs, keep updating the message with partial text
    while not gen_task.done():
        await asyncio.sleep(3)  # update every 3 seconds 
        current_text = streamer.get_text()
        if current_text:
            await streammsg.edit(content=current_text)
    
    # When done, update message with full text
    final_text = await gen_task
    await streammsg.edit(content=final_text)


AiChatBot = bot.ChatBot

#intents = discord.Intents.all()
import threading


class ChatBot(discord.Client):
    """ChatBot handles discord communication. This class runs its own thread that
    persistently watches for new messages, then acts on them when the bots username
    is mentioned. It will use the ChatAI class to generate messages then send them
    back to the configured server channel.

    ChatBot inherits the discord.Client class from discord.py
    """

    def __init__(self) -> None:
        #self.set_response_chance()
        super().__init__()
        #super().__init__(intents=intents)
        self.ai = AiChatBot(memory_file="memory.json")
        self.is_generating = False
        self.generate_lock = asyncio.Lock()
        self.chat_contexts = {} #userID:Object


    async def get_chat_context(self, message):
        channel: discord.TextChannel = self.get_channel(message.channel.id)
        context = []
        last_author = None
        last_lines = []

        async for msg in channel.history(limit=10, oldest_first=False, before=message.created_at):
            if msg.id == message.id:
                continue

            content = msg.content.strip()
            if not content:
                continue  # skip empty messages

            if msg.author == last_author:
                last_lines.insert(0, content)
            else:
                if last_lines:
                    grouped = f"{last_author.display_name}: " + "\n".join(last_lines)
                    context.append(grouped)
                last_author = msg.author
                last_lines = [content]

        if last_lines:
            grouped = f"{last_author.display_name}: " + "\n".join(last_lines)
            context.append(grouped)

        context.reverse()
        return context


    async def on_ready(self) -> None:
        """ Initializes the GPT2 AI on bot startup """
        print("Logged on as", self.user)



    def parse_command_flags(self, content: str):
        """
        Parses command-style flags from the start of a message.
        Supported: !recursive [depth], !depth [N], !memstore, !debug, !help, etc.
        Returns: (flags: dict, result: str)
        - If help flag is set, result is help text.
        - Otherwise, result is the cleaned input string.
        """
        flags = {
            "recursive": False,
            "depth": 3,
            "memstore": False,
            "debug": False,
            "help": False,
        }

        tokens = content.strip().split()
        remaining = []

        i = 0
        while i < len(tokens):
            token = tokens[i].lower()

            if token == "!help":
                flags["help"] = True
                break  # stop parsing further flags
            elif token == "!recursive":
                flags["recursive"] = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    flags["depth"] = int(tokens[i + 1])
                    i += 1
            elif token == "!depth":
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    flags["depth"] = int(tokens[i + 1])
                    i += 1
            elif token == "!memstore":
                flags["memstore"] = True
            elif token == "!debug":
                flags["debug"] = True
            else:
                remaining.append(tokens[i])
            i += 1

        if flags["help"]:
            help_text = (
                "**Available Command Flags:**\n"
                "`!recursive [N]` - Forces the bot to use recursive reasoning (default depth = 3, or use a number).\n"
                "`!depth N`       - Sets the recursion depth manually (used with or without !recursive).\n"
                "`!memstore`      - Forces the bot to treat this as a memory instruction.\n"
                "`!debug`         - Enables debug mode, useful for testing prompt contents or reasoning.\n"
                "`!help`          - Shows this help message.\n"
                "**YOU CAN USE MULTIPLE FLAGS AT THE SAME TIME!"
            )
            return flags, help_text

        clean_input = " ".join(remaining)
        return flags, clean_input


    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        tokenizer = static.DummyTokenizer()
        if message.author.id not in self.chat_contexts:
            context = self.chat_contexts[message.author.id] = static.ChatContext(tokenizer, 2048, 800)
            history = context.get_context_text()
        elif message.author.id in self.chat_contexts:
            context = self.chat_contexts[message.author.id]
            history = context.get_context_text()
            

        has_mentioned = any(str(mention) == f"{self.user.name}#{self.user.discriminator}" for mention in message.mentions)
        if not has_mentioned:
            return

        processed_input = self.process_input(message.content)
        flags, user_msg = self.parse_command_flags(processed_input)
        if flags["help"]:
            await message.reply(user_msg)
            return

        async with self.generate_lock:  # ✅ Thread-safe section
            async with message.channel.typing():
                try:
                    if flags["recursive"]:
                        response = await asyncio.to_thread(
                            self.ai.recursive_think,
                            username=message.author.display_name,
                            user_input=user_msg,
                            identifier=message.guild.id,
                            context=history,
                            force_recursive=True,
                            recursive_depth=flags["depth"],
                            debug=flags["debug"]
                        )
                    if flags["memstore"]:
                        response = await asyncio.to_thread(
                            self.ai.recursive_think,
                            username=message.author.display_name,
                            user_input=user_msg,
                            identifier=message.guild.id,
                            context=history,
                            category_override="instruction_memory",
                            debug=flags["debug"]
                        )
                    else:
                        response = await asyncio.to_thread(
                            self.ai.chat,
                            max_new_tokens=500,
                            username=message.author.display_name,
                            user_input=processed_input,
                            temperature=0.8,
                            identifier=message.guild.id,
                            context=history,
                            debug=flags["debug"]
                        )
                        await message.reply(response)
                    context.add_line(f"{message.author.display_name}: {processed_input}")
                    context.add_line(f"{self.ai.name}: {response}")


                except aiohttp.client_exceptions.ClientConnectorError:
                    pass

    def process_input(self, message):
        """ Process the input message """
        if type(message) == type(list):
            toreturn = []
            for msg in message:
                toreturn.append(msg.replace(f"<@1065772573331312650>", "ayokadeno"))
            return toreturn
        processed_input = message.replace(f"<@1065772573331312650>", "ayokadeno")
        return processed_input

    def process_context(self, messagelist):
        """ Process the context """
        toreturn = []
        for msg in messagelist:
            toreturn.append(msg.replace(f"<@1065772573331312650>", "ayokadeno"))
        return toreturn


    def check_if_should_respond(self, has_been_mentioned) -> bool:
        """ Check if the bot should respond to a message """
        should_respond = random.random() < self.response_chance

        return should_respond


    def set_response_chance(self, response_chance: float = 0.25) -> None:
        """ Set the response rate """
        self.response_chance = response_chance


    def set_model_name(self, model_name: str = "355M") -> None:
        """ Set the GPT2 model name """
        self.model_name = model_name
        
if __name__ == "__main__":
    bot = ChatBot()
    bot.run(open("token.txt", 'r').readlines()[0])