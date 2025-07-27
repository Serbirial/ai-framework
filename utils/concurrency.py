from llama_cpp import Llama
from src.static import Config
from multiprocessing import Process, Pipe
from src.agent import AgentInstance
from worker_passive_autonomy import AutonomousPassiveThinker

config = Config()

def _model_worker(conn, worker_config, model_path, username, botname, core_ids, n_threads, token_window, identifier, tier):
    import psutil, os
    psutil.Process(os.getpid()).cpu_affinity(core_ids)

    model = AgentInstance(name=botname, model=Llama(
        model_path=model_path,
        n_ctx=token_window,
        n_threads=n_threads,
        use_mlock=False,
        logits_all=False,
        verbose=False,
        use_mmap=False,
        n_gpu_layers=0,
        low_vram=True,
        n_batch=64
    ))

    try:
        persona_prompt = AgentInstance.get_persona_prompt(AgentInstance, identifier)
        thinker = AutonomousPassiveThinker(worker_config, config, persona_prompt)
        result, send_message, message = thinker.think(username, identifier, tier)
        conn.send({"done": True, "final": result, "send_message": send_message, "message": message})

    except Exception as e:
        conn.send({"error": str(e)})
    finally:
        conn.close()

class BackgroundThinkerProcess:
    def __init__(self, worker_config, model_path: str, tier: str, identifier: int, username: str, bot: ..., token_window: int, n_threads=2, core_ids=[0, 1]):
        self.model_path = model_path
        self.bot = bot
        self.username = username
        self.tier = tier
        self.token_window = token_window
        self.identifier = identifier
        self.n_threads = n_threads
        self.core_ids = core_ids

        parent_conn, child_conn = Pipe()
        self.conn = parent_conn
        self.proc = Process(
            target=_model_worker,
            args=(child_conn, worker_config, model_path, username, self.bot.name, core_ids, n_threads, token_window, identifier, tier)
        )
        self.proc.start()

    async def _communicate(self):
        output = None
        while True:
            if self.conn.poll(0.05):
                msg = self.conn.recv()
                if "done" in msg:
                    output = msg
                    if msg["send_message"] == True:
                        if msg["message"] != None:
                            user = await self.bot.fetch_user(self.identifier)
                            await user.send(msg["message"])
                    break
                elif "error" in msg:
                    output = f"[ERROR: {msg['error']}]"
                    
                    break
        return output

    async def run(self, **kwargs):
        return self._communicate()
