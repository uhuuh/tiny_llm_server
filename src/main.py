import asyncio

import multiprocessing as mp
import threading
from typing import List


from src.base import Config
from src.engine import Engine
from src.utils import get_config, ProcessExecutor
from src.server import app, ChatCompletionRequestResult, ChatCompletionRequest


class EngineProc:
    def __init__(self, config: Config, in_queue: mp.Queue, out_queue: mp.Queue):
        self.config = config
        self.engine = Engine(config)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.out_queue.put("engine_init_finished")

    def add_req_loop(self):
        while True:
            reqs = [self.in_queue.get()]
            while req := self.in_queue.get_nowait():
                reqs.append(req)
            self.engine.add_requests(reqs)

    def step_req_loop(self):
        while True:
            reqs = self.engine.step()
            self.out_queue.put(reqs)

class EngineClient:
    def __init__(self, config):
        self.config = config
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        asyncio.set_event_loop(loop)
        self.loop = loop
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.req_queues_lock = threading.Lock()
        self.req_queues = {}
        self.engine_proc = ProcessExecutor(
            EngineProc,
            cls_kwargs={
                "config": self.config,
                "in_queue": self.in_queue,
                "out_queue": self.out_queue
            },
            io_methods=["add_req_loop", "step_req_loop"],
        )
        self.engine_proc.start()
        engine_init_msg = self.out_queue.get()

    def ret_req_loop(self):
        def _add_req_res(r):
            self.req_queues[r.id].put(r)

        while True:
            reqs: List[ChatCompletionRequestResult] = self.in_queue.get()
            with self.req_queues_lock:
                for r in reqs:
                    assert r.id in self.req_queues
                    self.loop.call_soon_threadsafe(_add_req_res)

    async def chat_completions_handler(self, req: ChatCompletionRequest):
        q = asyncio.Queue()
        with self.req_queues_lock:
            self.req_queues[req.id] = q
        response = await q.get() # TODO when is None
        return response

if __name__ == "__main__":
    # TODO 如何抢占端口
    config = get_config()
    client = EngineClient(config)
    app.state.chat_completions_handler = client.chat_completions_handler

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

