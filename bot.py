import random
import discord
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
from src import bot
from src import static

from transformers import TextStreamer
from transformers import TextStreamer

class DiscordStreamer(TextStreamer):
    def __init__(self, tokenizer, message, initial_text="", delay=3, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.message = message
        self.delay = delay
        self.buffer = initial_text
        self.queue = asyncio.Queue()
        self.token_buffer = ""
        self.updater_task = asyncio.create_task(self.update_loop())

    def on_text(self, text: str, **kwargs):
        # This is called for every new token chunk
        asyncio.get_event_loop().call_soon_threadsafe(self.queue.put_nowait, text)

    async def update_loop(self):
        while True:
            token = await self.queue.get()
            if token is None:  # End signal
                break
            self.token_buffer += token
            if len(self.token_buffer) >= 10 or token.endswith(('.', '!', '?')):
                self.buffer += self.token_buffer
                self.token_buffer = ""
                try:
                    await self.message.edit(content=self.buffer)
                except Exception:
                    pass  # Might be deleted or fail; safe to ignore
                await asyncio.sleep(self.delay)
        # Final flush
        self.buffer += self.token_buffer
        await self.message.edit(content=self.buffer)


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
        async for msg in channel.history(limit=10, oldest_first=False, before=message.created_at):
            if msg.id == message:
                pass
            else:
                context.append(f"{msg.author.display_name}: {msg.content}\n")
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
                        streammsg = await message.reply("Hmm...") 
                        streamer = DiscordStreamer(static.tokenizer, streammsg, "Thinking...")

                        processed_input = processed_input.split("!stream", 1)[1]

                        def blocking_chat():
                            self.ai.chat(
                                username=message.author.display_name,
                                user_input=processed_input,
                                identifier=message.guild.id,
                                context=processed_context,
                                debug=False,
                                streamer=streamer
                            )

                        # Offload entire chat() call to thread to prevent blocking the event loop
                        await asyncio.to_thread(blocking_chat)

                        await streamer.queue.put(None)
                    else:
                        response = await asyncio.to_thread(
                            self.ai.chat,
                            username=message.author.display_name,
                            user_input=processed_input,
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