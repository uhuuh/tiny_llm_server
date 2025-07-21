import asyncio
import os
import threading

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sympy import asinh

from engine import Engine, RequestFactory
import multiprocessing as mp
import threading as th
from loguru import logger

from src.base import SampleConfig, InferConfig, Config, RequestParam, request_callback, MessageType, Listener


class EngineRunner:
    def __init__(self, config):
        pass

class EngineManager:
    def __init__(self, config: Config, out_queue):
        self.config = config
        self.dp = config.infer_config.data_parallel

        self.engine_ok_nun = 0
        self.cond = th.Condition()

        self.out_queue = out_queue
        self.in_queues = [mp.Queue() for _ in range(self.dp)]

        self.send_req_num = np.zeros(self.dp)
        self.finish_req_num = np.zeros(self.dp)

    def add_request(self, req):
        target_idx = np.argmin(self.send_req_num - self.finish_req_num)
        self.send_req_num[target_idx] += 1
        self.in_queues[target_idx].put_nowait(req)

    def start(self):
        def fun(config, in_queue, out_queue):
            e = Engine(config, in_queue, out_queue)
            while True:
                e.step()

        engine_procs = []
        for a in range(self.dp):
            p = mp.Process(target=fun, args=(self.config, self.in_queues[a], self.out_queue))
            p.start()
            engine_procs.append(p)

        with self.cond:
            while self.engine_ok_nun < self.dp:
                self.cond.wait()

    def handle_engine_start(self, msg):
        with self.cond:
            self.engine_ok_nun += 1
            if self.engine_ok_nun == self.dp:
                self.cond.notify_all()

    def handle_response(self, msg):
        engine_id = msg.engine_id
        self.finish_req_num[engine_id] += 1

class Client:
    def __init__(self, config):
        self.config = config
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.now_loop = asyncio.get_running_loop()
        self.response_queues = dict()
        self.request_factory = RequestFactory(config)

        self.engine_manager = EngineManager(self.config, self.out_queue)
        self.engine_manager.start()

        handlers = {
            MessageType.engine_start: [
                (self.engine_manager, self.engine_manager.handle_engine_start)
            ],
            MessageType.response: [
                (self.engine_manager, self.engine_manager.handle_response),
                (self, self.handle_presonse)
            ],
        }
        self.listener = Listener(self.out_queue, handlers)

    async def generate_response(self, param: RequestParam):
        logger.info(f"recv param={param}")
        sample_config = SampleConfig(
            max_new_token_new=param.max_tokens,
            temperature=param.temperature,
        )
        req = self.request_factory.create(
            tokens=param.prompt,
            sample_config=sample_config,
            step_fun=request_callback
        )
        logger.info(f">>> debug tokens={type(req.tokens)}")
        logger.info(f"create req={req}")
        self.in_queue.put_nowait(req)
        logger.info("req enter in_queue")

        assert self.now_loop == asyncio.get_running_loop()
        response_queue = asyncio.Queue()
        self.response_queues[req.id] = response_queue
        logger.info("req register response_queues, que_id={}, loop_id={}",
                    id(response_queue), id(asyncio.get_event_loop()))

        while True:
            response = await response_queue.get() # TODO when is None
            if response is None:
                break

            del self.response_queues[req.id]
            logger.info("req response")
            return response

    def handle_presonse(self, req):
        # 如果从out_queue收到，loop不为空
        if req.id in self.response_queues:
            self.now_loop.call_soon_threadsafe(
                self.response_queues[req.id].put_nowait, req.tokens)
            logger.info("req enter response_queue, tokens={}, que_id={}, loop_id={}",
                        self.response_queues[req.id], id(self.response_queues[req.id]),
                        id(asyncio.get_event_loop()))
        else:
            raise

import signal
import sys
def shutdown_handler(sig, frame):
    print("Shutting down gracefully...")
    sys.exit(0)
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


infer_config = InferConfig(
    model_path="/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct",
    max_req_num=256,
    max_batch_token_num=4096,
    gpu_memory_utilization=0.5,
    block_size=16,
    enable_prefix_cache=True,
    enable_debug=True,
    max_prefill_len=32,
    enable_chunked_prefill=True,
)
config = Config(
    infer_config=infer_config,
)
client = Client(config)
app = FastAPI()

@app.post("/")
async def generate(request: RequestParam):
    return await client.generate_response(request)

if __name__ == "__main__":
    # TODO 如何抢占端口
    os.system("kill -9 $(lsof -t -i:8001)")
    os.system("kill -9 $(lsof -t -i:8001)")
    os.system("kill -9 $(lsof -t -i:8001)")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

