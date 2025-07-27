import time
import asyncio
import functools
from queue import Queue, Empty
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from threading import Lock

from src import agent, static

app = FastAPI()

context = static.ChatContext(static.DummyTokenizer(), 1024)

# Load the chatbot once here
print("Loading chatbot AI once...")
chatbot_ai = agent.AgentInstance(db_path=static.DB_PATH)

generate_lock = asyncio.Lock()
model_lock = Lock()


class Streamer:
    def __init__(self, cooldown=0.3, max_chars=500): # ignoring max chars
        self.cooldown = cooldown
        self.max_chars = max_chars
        self.buffer = ""
        self.lock = Lock()
        self.queue = Queue()
        self.last_emit = 0
        self.closed = False
        self.special_buffer = []

    def __call__(self, new_text_chunk):
        with self.lock:
            self.buffer += new_text_chunk

            now = time.monotonic()
            elapsed = now - self.last_emit

            if elapsed >= self.cooldown:
                self._emit_buffer()
                self.last_emit = now
            else:
                delay = self.cooldown - elapsed
                threading.Timer(delay, self._emit_buffer).start()

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

    def generator(self):
        last_heartbeat = time.monotonic()
        heartbeat_interval = 15  # seconds

        while True:
            try:
                chunk = self.queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield chunk.encode("utf-8")
                last_heartbeat = time.monotonic()
            except Empty:
                now = time.monotonic()
                if now - last_heartbeat > heartbeat_interval:
                    yield b"<|heartbeat|>"
                    last_heartbeat = now

            while True:
                with self.lock:
                    if not self.special_buffer:
                        break
                    try:
                        special_line = self.special_buffer.pop(0)
                    except IndexError:
                        break
                yield f"SPECIAL:{special_line}\n".encode("utf-8")

            if self.closed and self.queue.empty() and not self.special_buffer:
                break


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
        loop = asyncio.get_running_loop()
        partial_chat = functools.partial(
            chatbot_ai.chat,
            username=username,
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
            cnn_file_path=cnn_file_path,
        )
        with model_lock:
            await loop.run_in_executor(None, partial_chat)
        streamer.close()


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()

    user_input = data.get("message", "")
    username = data.get("username", "web_user")
    tier = data.get("tier", "t0")
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    recursive_depth = int(data.get("recursive_depth", 3))
    category_override = data.get("category_override", None)
    debug = bool(data.get("debug", False))
    force_recursive = bool(data.get("force_recursive", False))

    ip = request.client.host

    streamer = Streamer(cooldown=0.3, max_chars=100)

    async def run_generation():
        await generate_ai_response(
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

    # Schedule generation in background async task
    generation_task = asyncio.create_task(run_generation())

    return StreamingResponse(
        streamer.generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


#if __name__ == "__main__":
    # IMPORTANT: Run with 1 worker to ensure single model instance
#    app.run(host="0.0.0.0", port=8000, debug=False, workers=1)