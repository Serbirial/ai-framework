import time
import threading
from queue import Queue, Empty
from flask import Flask, request, Response, stream_with_context
import asyncio
import functools

from src import bot
from src import static

app = Flask(__name__)

context = static.ChatContext(static.DummyTokenizer(), 1024)
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
        self.special_buffer = []

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
        """
        """
        while True:
            try:
                chunk = self.queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
            except Empty:
                pass

            while True:
                with self.lock:
                    if not self.special_buffer:
                        break
                    special_line = self.special_buffer.pop(0)

                yield f"data: SPECIAL:{special_line}\n\n"

            if self.closed and self.queue.empty() and not self.special_buffer:
                break

    def add_special(self, data):
        with self.lock:
            self.special_buffer.append(data)


async def generate_ai_response(
    user_input,
    streamer,
    username="web_user",
    tier="t0",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    debug=False,
    force_recursive=False,
    recursive_depth=3,
    category_override=None,
    tiny_mode=False,
    cnn_file_path=None,
):
    """
    Calls your bot.ChatBot.chat synchronously in a thread,
    passing streamer for chunked output.
    """
    async with generate_lock:

        loop = asyncio.get_running_loop()
        partial_chat = functools.partial(
            chatbot_ai.chat,
            username=username,
            user_input=user_input,
            identifier="web_session_1",
            tier=tier,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            context=context.get_context_text(),
            debug=debug,
            streamer=streamer,
            force_recursive=force_recursive,
            recursive_depth=recursive_depth,
            category_override=category_override,
            tiny_mode=tiny_mode,
            cnn_file_path=cnn_file_path,
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
    data = request.json or {}

    user_input = data.get("message", "")
    username = data.get("username", "web_user")
    tier = data.get("tier", "t0")
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    recursive_depth = data.get("recursive_depth", 3)
    category_override = data.get("category_override", None)
    debug = bool(data.get("debug", False))
    force_recursive = bool(data.get("force_recursive", False))

    try:
        recursive_depth = int(recursive_depth)
    except Exception:
        recursive_depth = 3

    streamer = FlaskStreamer(cooldown=0.3, max_chars=100)

    def stream_response():
        def worker():
            run_async_loop(
                generate_ai_response(
                    user_input=user_input,
                    streamer=streamer,
                    username=username,
                    tier=tier,
                    temperature=temperature,
                    top_p=top_p,
                    recursive_depth=recursive_depth,
                    category_override=category_override,
                    debug=debug,
                    force_recursive=force_recursive,
                )
            )

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
    with open("./index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, threaded=False)
