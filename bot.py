import random
import discord
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
from src import bot
from src import static

import asyncio

import time

async def generate_and_stream(self, message, processed_input, processed_context):
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
            context=processed_context,
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


    async def on_message(self, message: discord.Message) -> None:
        """ Handle new messages sent to the server channels this bot is watching """

        if message.author == self.user:
            # Skip any messages sent by ourselves so that we don't get stuck in any loops
            return

        # Check to see if bot has been mentioned
        has_mentioned = False
        for mention in message.mentions:
            if str(mention) == self.user.name+"#"+self.user.discriminator:
                has_mentioned = True
                break

        # Only respond randomly (or when mentioned), not to every message
        #if random.random() > float(self.response_chance) and has_mentioned == False:
        #    return
        if has_mentioned:
            processed_input = self.process_input(message.content)

            context = await self.get_chat_context(message)

            processed_context = self.process_context(context)


            async with message.channel.typing():
                try:
                    if processed_input.lower().startswith("!stream"):
                        processed_input = processed_input.split("!stream", 1)[1]
                        await generate_and_stream(self, message, processed_input, processed_context)
                    else:
                        response = await asyncio.to_thread(
                            self.ai.chat,
                            username=message.author.display_name,
                            user_input=processed_input,
                            temperature=0.8,
                            identifier=message.guild.id,
                            context=processed_context,
                            debug=False,
                            streamer=None
                        )
                        await message.reply(response)



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