from .interactive_tool_class import InteractiveTool
import discord
import io
import discord
from src.static import WorkerConfig
from datetime import datetime

class DiscordInteractiveTool(InteractiveTool):
    """
    Discord tool allowing:
    - SET_FILE <filename>  (set a file in memory to send)
    - SEND_MESSAGE <content>  (send one message with that file)
    - EDIT_MESSAGE <new_content> (edit sent message content)
    """

    def __init__(self, discord_client: discord.Client, WorkerConfig: WorkerConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.client = discord_client
        self.config = WorkerConfig
        if self.config:
            self.op_member_id = self.config.identifier # OPs id
            self.guild_id = self.config.guild_id # guild
            self.channel_id = self.config.channel_id # Channel
            self.op_message_id = self.config.message_id # OPs MID
        
        self._file_name = None
        self._file_path = None
        self._sent_message = None  # discord.Message object once sent
        self._sent_once = False

        self.detailed_logs = []

    def log_step(self, command, result):
        self.detailed_logs.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "command": command,
            "result": result
        })

    def set_file(self, filename: str, file_path: bytes):
        self._file_name = filename
        self._file_path = file_path
        self.log_step("SET_FILE", f"File set: {filename} ({len(file_path)} bytes)")

    async def send_message(self, channel_id: int):
        if self._sent_once:
            result = "Message already sent. Only one message allowed, editing message (DONT USE SEND_MESSAGE, USE EDIT_MESSAGE)."
            self.log_step("SEND_MESSAGE", result)
            result = await self.edit_message()
            return result

        channel = self.client.get_channel(channel_id)
        if channel is None:
            result = f"Channel ID {channel_id} not found."
            self.log_step("SEND_MESSAGE", result)
            return result

        file = None
        if self._file_path and self._file_name:
            file = discord.File(io.BytesIO(self._file_path), filename=self._file_name)

        try:
            msg = await channel.send(content=None, file=file)
            self._sent_message = msg
            self._sent_once = True
            result = f"Message sent in channel {channel_id} with ID {msg.id}."
        except Exception as e:
            result = f"Error sending message: {e}"

        self.log_step("SEND_MESSAGE", result)
        return result

    async def edit_message(self, new_content: str):
        if not self._sent_message:
            result = "No message has been sent yet."
            self.log_step("EDIT_MESSAGE", result)
            return result

        try:
            await self._sent_message.edit(content=new_content)
            result = "Message content edited."
        except Exception as e:
            result = f"Error editing message: {e}"

        self.log_step("EDIT_MESSAGE", result)
        return result


    async def reply_to_message(bot, data):
        channel = await bot.fetch_channel(int(data["channel_id"]))
        message = await channel.fetch_message(int(data["message_id"]))
        await message.reply(data["content"])
        return "Replied to the message."

    async def react_to_message(bot, data):
        channel = await bot.fetch_channel(int(data["channel_id"]))
        message = await channel.fetch_message(int(data["message_id"]))
        await message.add_reaction(data["emoji"])
        return "Replied to the message."

    async def voice_speak(bot, data): # BROKEN
        return "This is currently stubbed"
    
        guild = bot.get_guild(int(data["guild_id"]))
        vc_channel = guild.get_channel(int(data["channel_id"]))
        vc = await vc_channel.connect()

        tts_path = "/tmp/tts.mp3"
        subprocess.run(["edge-tts", "--text", data["text"], "--write-media", tts_path])

        vc.play(discord.FFmpegPCMAudio(tts_path))
        while vc.is_playing():
            await asyncio.sleep(1)

        await vc.disconnect()
        return {"status": "ok", "action": "VoiceSpeak"}

    async def tag_users(bot, data):
        max_mentions = 3
        channel = await bot.fetch_channel(int(data["channel_id"]))
        
        # Limit number of users to tag
        limited_users = data["user_ids"][:max_mentions]
        mentions = " ".join([f"<@{uid}>" for uid in limited_users])
        
        await channel.send(f"{mentions} {data['message']}")
        return f"Tagged {len(limited_users)} user(s)."


    async def create_poll(bot, data):
        channel = await bot.fetch_channel(int(data["channel_id"]))
        message = await channel.send(f"**{data['question']}**\n" + "\n".join(data["options"]))
        for option in data["options"]:
            emoji = option.strip().split(" ")[0]
            await message.add_reaction(emoji)
        return f"Created Poll At Message ID {message}"

    async def fetch_chat_history(bot, data):
        channel = await bot.fetch_channel(int(data["channel_id"]))
        messages = [msg async for msg in channel.history(limit=int(data.get("limit", 10)))]
        result = [{"author": str(m.author), "content": m.content} for m in messages[::-1]]
        return {
            "status": "ok",
            "action": "FetchChatHistory",
            "messages": result
        }

    async def receive_output(self, input_data: str):
        """
        Accepts a command string and executes the corresponding Discord interaction.

        Supported commands:
        - SET_FILE <filename> <filepath>
        - SEND_MESSAGE <content>
        - EDIT_MESSAGE <new content>
        - SEND_IMAGE <filename> <filepath>
        - REPLY_TO <channel_id> <message_id> <content>
        - REACT_TO <channel_id> <message_id> <emoji>
        - TAG_USERS <channel_id> <user_ids> <message>
        - CREATE_POLL <channel_id> <question> <options>
        - FETCH_CHAT_HISTORY <channel_id> <limit>

        All actions operate strictly within the self-identified context (pre-injected by config).
        """
        parts = input_data.split(maxsplit=2)
        if not parts:
            return "No command provided."

        command = parts[0].upper()

        if command == "SET_FILE":
            if len(parts) < 3:
                return "Usage: SET_FILE <filename> <filepath>"
            filename = parts[1]
            try:
                file_path = parts[2]
            except Exception as e:
                return f"Failed to get file path: {e}"

            self.set_file(filename, file_path)
            return f"File '{filename}' set successfully."
        elif command == "SEND_IMAGE":
            if len(parts) < 3:
                return "Usage: SEND_IMAGE <filename> <filepath>"
            filename = parts[1]
            filepath = parts[2]

            channel = self.client.get_channel(self.channel_id)
            if channel is None:
                return f"Channel ID {self.channel_id} not found."

            try:
                file = discord.File(io.BytesIO(filepath), filename=filename)
                msg = await channel.send(content=filename, file=file)
                self.log_step("SEND_IMAGE", f"Sent image as message ID {msg.id}")
                return f"Image sent successfully with message ID {msg.id}."
            except Exception as e:
                return f"Error sending image: {e}"

        elif command == "SEND_MESSAGE":
            if len(parts) < 2:
                return "Usage: SEND_MESSAGE <content>"

            return await self.send_message(self.channel_id)

        elif command == "EDIT_MESSAGE":
            if len(parts) < 2:
                return "Usage: EDIT_MESSAGE <new_content>"
            new_content = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
            return await self.edit_message(new_content)

        else:
            return f"Unknown command '{command}'."

    def describe(self):
        return {
            "name": "DiscordInteractiveTool",
            "description": (
                "Interactive Discord tool to communicate with a user in a specific channel using pre-injected configuration. "
                "Supports a single message with optional file, edits, replies, tagging users, reactions, polls, and fetching history."
            ),
            "commands": [
                "SET_FILE <filename> <filepath> — Load a file into memory to attach to a message",
                "SEND_MESSAGE <content> — Sends a message with the previously set file (only once per session)",
                "EDIT_MESSAGE <new_content> — Edit the previously sent message",
                "SEND_IMAGE <filename> <filepath> — Sends an image to the channel immediately",
                "REPLY_TO <channel_id> <message_id> <content> — Replies to a specific message",
                "REACT_TO <channel_id> <message_id> <emoji> — Adds a reaction emoji to a message",
                "TAG_USERS <channel_id> <user_ids> <message> — Tags up to 3 users with a message",
                "CREATE_POLL <channel_id> <question> <options> — Creates a poll with reactions as options",
                "FETCH_CHAT_HISTORY <channel_id> <limit> — Fetches the latest N messages"
            ]
        }
