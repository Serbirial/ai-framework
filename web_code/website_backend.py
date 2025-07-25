import time
import threading
from queue import Queue, Empty
import asyncio
import functools

from sanic import Sanic, response
from sanic_ext import Extend

from src import bot
from src import static

app = Sanic("AyokPT")
Extend(app)

context = static.ChatContext(static.DummyTokenizer(), 1024)
generate_lock = asyncio.Lock()
model_lock = threading.Lock()

class SanicStreamer:
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

    def add_special(self, data):
        with self.lock:
            self.special_buffer.append(data)

    async def generator(self):
        last_heartbeat = time.monotonic()
        heartbeat_interval = 15  # seconds

        while True:
            try:
                chunk = self.queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield f"{chunk}"
                last_heartbeat = time.monotonic()
            except Empty:
                now = time.monotonic()
                if now - last_heartbeat > heartbeat_interval:
                    yield "<|heartbeat|>"
                    last_heartbeat = now

            while True:
                with self.lock:
                    if not self.special_buffer:
                        break
                    special_line = self.special_buffer.pop(0)

                yield f"SPECIAL:{special_line}\n"

            if self.closed and self.queue.empty() and not self.special_buffer:
                break

@app.before_server_start
async def setup_chatbot(app, _):
    # Initialize the chatbot once here
    app.ctx.chatbot = bot.ChatBot(db_path=static.DB_PATH)

async def generate_ai_response(
    user_input,
    streamer,
    ip,
    username="web_user",
    tier="t0",
    max_new_tokens=226,
    temperature=0.7,
    top_p=0.9,
    debug=False,
    force_recursive=False,
    recursive_depth=3,
    category_override=None,
    tiny_mode=False,
    cnn_file_path=None,
):
    async with generate_lock:
        return  app.ctx.chatbot.chat(username=username,
            user_input=user_input,
            identifier=ip,
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
            cnn_file_path=cnn_file_path)

@app.post("/api/chat")
async def chat(request):
    data = request.json or {}

    user_input = data.get("message", "")
    username = data.get("username", "web_user")
    tier = data.get("tier", "t0")
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    recursive_depth = int(data.get("recursive_depth", 3))
    category_override = data.get("category_override", None)
    debug = bool(data.get("debug", False))
    force_recursive = bool(data.get("force_recursive", False))

    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()

    streamer = SanicStreamer(cooldown=0.3, max_chars=100)

    async def stream_response(_):
        # Use asyncio.to_thread to run blocking code safely
        async def worker():
            await asyncio.to_thread(
                generate_ai_response,
                user_input=user_input,
                streamer=streamer,
                ip=ip,
                username=username,
                tier=tier,
                temperature=temperature,
                top_p=top_p,
                recursive_depth=recursive_depth,
                category_override=category_override,
                debug=debug,
                force_recursive=force_recursive,
            )

        # Run the blocking call in background async task
        task = asyncio.create_task(worker())

        async for chunk in streamer.generator():
            yield chunk

        await task  # ensure task finishes

    return response.stream(
        stream_response,
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.get("/")
async def index(_):
    with open("./index.html", "r", encoding="utf-8") as f:
        return response.html(f.read())

if __name__ == "__main__":
    # IMPORTANT: Run with 1 worker to ensure single model instance
    app.run(host="0.0.0.0", port=8000, debug=False, workers=1)
