from .interactive_tool_class import InteractiveTool
import discord
import io
import asyncio
from datetime import datetime

class DiscordInteractiveTool(InteractiveTool):
    """
    Discord tool allowing:
    - SET_FILE <filename>  (set a file in memory to send)
    - SEND_MESSAGE <content>  (send one message with that file)
    - EDIT_MESSAGE <new_content> (edit sent message content)
    """

    def __init__(self, discord_client: discord.Client, channel_id: int, **kwargs):
        super().__init__(**kwargs)
        self.client = discord_client
        self.channel_id = channel_id
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

    async def receive_output(self, input_data: str):
        """
        Commands:
        SET_FILE <filename> <filepath>
        SEND_MESSAGE <content>
        EDIT_MESSAGE <new content>
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
                "Allows sending a single Discord message with an in-memory file attachment, "
                "and editing its content later.\n"
                "Only one message can be sent per session."
            ),
            "commands": [
                "SET_FILE <filename> <filepath>",
                "SEND_MESSAGE <content>",
                "EDIT_MESSAGE <new_content>"
            ]
        }
