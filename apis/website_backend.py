import time
import threading
from queue import Queue, Empty
from flask import Flask, request, Response, stream_with_context
import asyncio
import functools


from ..src import bot

from ..src import static

app = Flask(__name__)

chatbot_ai = bot.ChatBot(db_path=static.DB_PATH)
generate_lock = asyncio.Lock()  # single global lock until using sanic

class FlaskStreamer:
    def __init__(self, cooldown=0.3, max_chars=500):
        self.cooldown = cooldown
        self.max_chars = max_chars
        self.buffer = ""
        self.lock = threading.Lock()
        self.queue = Queue()
        self.last_emit = 0
        self.closed = False

    def __call__(self, new_text_chunk):
        with self.lock:
            self.buffer += new_text_chunk
            if len(self.buffer) > self.max_chars:
                excess = len(self.buffer) - self.max_chars
                self.buffer = self.buffer[excess:]

            now = time.monotonic()
            elapsed = now - self.last_emit

            if elapsed >= self.cooldown:
                self._emit_buffer()
                self.last_emit = now
            else:
                delay = self.cooldown - elapsed
                timer = threading.Timer(delay, self._emit_buffer)
                timer.daemon = True
                timer.start()

    def _emit_buffer(self):
        with self.lock:
            if self.buffer:
                self.queue.put(self.buffer)
                self.buffer = ""

    def close(self):
        self._emit_buffer()
        self.closed = True
        self.queue.put(None)

    def generator(self):
        while True:
            try:
                chunk = self.queue.get(timeout=5)
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
            except Empty:
                if self.closed:
                    break



async def generate_ai_response(user_input, streamer):
    """
    Calls your bot.ChatBot.chat synchronously in a thread,
    passing streamer for chunked output.
    """
    async with generate_lock:

        loop = asyncio.get_running_loop()
        partial_chat = functools.partial(
            chatbot_ai.chat,
            username="web_user",
            user_input=user_input,
            identifier="web_session_1",
            tier="free",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            context=None,
            debug=False,
            streamer=streamer,
            force_recursive=False,
            recursive_depth=3,
            category_override=None,
            tiny_mode=False,
            cnn_file_path=None,
        )

        await loop.run_in_executor(None, partial_chat)

        streamer.close()



def run_async_loop(coro):
    """Run an async coroutine from sync context (Flask route)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    streamer = FlaskStreamer(cooldown=0.3, max_chars=100)

    def stream_response():
        def worker():
            run_async_loop(generate_ai_response(user_input, streamer))

        threading.Thread(target=worker, daemon=True).start()

        yield from streamer.generator()

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return Response(stream_with_context(stream_response()), headers=headers)

import time
import threading
from queue import Queue, Empty
from flask import Flask, request, Response, stream_with_context
import asyncio
import functools

from src import bot
from src import static

app = Flask(__name__)

chatbot_ai = bot.ChatBot(db_path=static.DB_PATH)
generate_lock = asyncio.Lock()  # single global lock until using sanic

class FlaskStreamer:
    def __init__(self, cooldown=0.3, max_chars=500):
        self.cooldown = cooldown
        self.max_chars = max_chars
        self.buffer = ""
        self.lock = threading.Lock()
        self.queue = Queue()
        self.last_emit = 0
        self.closed = False

    def __call__(self, new_text_chunk):
        with self.lock:
            self.buffer += new_text_chunk
            if len(self.buffer) > self.max_chars:
                excess = len(self.buffer) - self.max_chars
                self.buffer = self.buffer[excess:]

            now = time.monotonic()
            elapsed = now - self.last_emit

            if elapsed >= self.cooldown:
                self._emit_buffer()
                self.last_emit = now
            else:
                delay = self.cooldown - elapsed
                timer = threading.Timer(delay, self._emit_buffer)
                timer.daemon = True
                timer.start()

    def _emit_buffer(self):
        with self.lock:
            if self.buffer:
                self.queue.put(self.buffer)
                self.buffer = ""

    def close(self):
        self._emit_buffer()
        self.closed = True
        self.queue.put(None)

    def generator(self):
        while True:
            try:
                chunk = self.queue.get(timeout=5)
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
            except Empty:
                if self.closed:
                    break

async def generate_ai_response(user_input, streamer):
    """
    Calls your bot.ChatBot.chat synchronously in a thread,
    passing streamer for chunked output.
    """
    async with generate_lock:

        loop = asyncio.get_running_loop()
        partial_chat = functools.partial(
            chatbot_ai.chat,
            username="web_user",
            user_input=user_input,
            identifier="web_session_1",
            tier="free",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            context=None,
            debug=False,
            streamer=streamer,
            force_recursive=False,
            recursive_depth=3,
            category_override=None,
            tiny_mode=False,
            cnn_file_path=None,
        )

        await loop.run_in_executor(None, partial_chat)

        streamer.close()

def run_async_loop(coro):
    """Run an async coroutine from sync context (Flask route)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    streamer = FlaskStreamer(cooldown=0.3, max_chars=100)

    def stream_response():
        def worker():
            run_async_loop(generate_ai_response(user_input, streamer))

        threading.Thread(target=worker, daemon=True).start()

        yield from streamer.generator()

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return Response(stream_with_context(stream_response()), headers=headers)


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Stream Debug Console</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', sans-serif;
      background-color: #0d1117;
      color: #c9d1d9;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background-color: #161b22;
      padding: 1rem;
      text-align: center;
      font-size: 1.5rem;
      border-bottom: 1px solid #30363d;
    }
    #stream {
      flex-grow: 1;
      padding: 1rem;
      overflow-y: auto;
      white-space: pre-wrap;
      font-family: Consolas, monospace;
      border-bottom: 1px solid #30363d;
    }
    #controls {
      background-color: #161b22;
      padding: 1rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    input[type="text"] {
      flex-grow: 1;
      padding: 0.5rem;
      background-color: #0d1117;
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 4px;
    }
    button {
      padding: 0.5rem 1rem;
      background-color: #238636;
      border: none;
      color: white;
      font-weight: bold;
      border-radius: 4px;
      cursor: pointer;
    }
    button:disabled {
      background-color: #2ea04355;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <header>Chat with Ayok.</header>
  <div id="stream">[output appears here]</div>
  <div id="controls">
    <input id="userInput" type="text" placeholder="Type a message to send to the AI...">
    <button id="sendBtn">Send</button>
  </div>

  <script>
    const streamDiv = document.getElementById('stream');
    const inputBox = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    sendBtn.onclick = async () => {
      const msg = inputBox.value.trim();
      if (!msg) return;
      inputBox.value = '';
      streamDiv.textContent += `\\n> ${msg}\\n`;
      sendBtn.disabled = true;

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // Strip SSE "data: " prefix and trailing newlines
        const lines = buffer.split('\\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            streamDiv.textContent += line.substring(6);
          }
        }
        streamDiv.scrollTop = streamDiv.scrollHeight;
        buffer = '';
      }

      sendBtn.disabled = false;
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, threaded=True)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
