import asyncio

import multiprocessing as mp
import queue
import threading
from typing import List
from loguru import logger
from src.base import Config
from src.engine import Engine
from src.utils import get_config, ProcessExecutor
from src.server import app, ChatCompletionRequestResult, ChatCompletionRequest

INIT_MSG = "engine_init_finished"

class EngineProc:
    def __init__(self, config: Config, in_queue: mp.Queue, out_queue: mp.Queue):
        self.config = config
        self.engine = Engine(config)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.init_text = INIT_MSG
        self.out_queue.put(self.init_text)

    def add_req_loop(self):
        while True:
            reqs = [self.in_queue.get()]
            try:
                while True:
                    reqs.append(self.in_queue.get_nowait())
            except queue.Empty:
                pass
            self.engine.add_requests(reqs)
            reqs.clear()

    def step_req_loop(self):
        while True:
            reqs = self.engine.step()
            logger.info("engine_proc_finish_req {}", [r.id for r in reqs])
            self.out_queue.put_nowait(reqs)

class EngineClient:
    def __init__(self, config):
        self.config = config
        self.loop = None
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
        assert self.out_queue.get() == INIT_MSG

        self.enable_recv_loop = True
        self.recv_loop_thread = threading.Thread(target=self.ret_req_loop)
        self.recv_loop_thread.start()

    def ret_req_loop(self):
        def _add_req_res(r):
            logger.info("before server_recv_req {}", r.id)
            self.req_queues[r.id].put_nowait(r)

        while self.enable_recv_loop:
            reqs: List[ChatCompletionRequestResult] = self.out_queue.get()
            logger.info("client_finish_req {}", [r.id for r in reqs])
            with self.req_queues_lock:
                for r in reqs:
                    assert r.id in self.req_queues
                    self.loop.call_soon_threadsafe(_add_req_res, r)

    async def chat_completions_handler(self, req: ChatCompletionRequest):
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
            self.loop.set_debug(True)

        q = asyncio.Queue()
        with self.req_queues_lock:
            self.req_queues[req.id] = q

        self.in_queue.put_nowait(req)

        response = await q.get()
        logger.info("server_recv_req {}", response.id)
        return response

if __name__ == "__main__":
    # TODO 如何抢占端口
    config = get_config()
    client = EngineClient(config)
    app.state.chat_completions_handler = client.chat_completions_handler

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

