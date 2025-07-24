import asyncio
import os
import threading
from ftplib import MSG_OOB

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from engine import Engine
from src.squence import RequestFactory, Sequence
import multiprocessing as mp
import threading as th
from loguru import logger

from src.base import SampleConfig, InferConfig, Config, RequestParam, MessageType, Listener


def request_callback(req: Sequence, engine: Engine):
    # NOTE 回调在另外一个进程被调用，使用mp而非asyncio的queue传递输出
    if req.is_finish():
        req.engine_id = engine.id
        engine.out_queue.put_nowait((MessageType.response, req))
        logger.info("req enter out_queue")

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
        target_idx = np.argmin(self.send_req_num)
        self.send_req_num[target_idx] += 1
        self.in_queues[target_idx].put_nowait((MessageType.request, req))
        logger.info("req enter in_queue, engine_id={}, send_req_nm={}, finish_req_num={}",
                    target_idx, self.send_req_num, self.finish_req_num)

    def start(self):
        def fun(id, config, in_queue, out_queue):
            e = Engine(id, config)
            e.run(in_queue, out_queue)

        engine_procs = []
        for a in range(self.dp):
            p = mp.Process(target=fun, args=(a, self.config, self.in_queues[a], self.out_queue))
            p.start()
            engine_procs.append(p)

        with self.cond:
            while self.engine_ok_nun < self.dp:
                self.cond.wait()

    @staticmethod
    def handle_engine_start(self, msg):
        with self.cond:
            self.engine_ok_nun += 1
            logger.info("engine_ok_nun={}", msg)
            if self.engine_ok_nun == self.dp:
                self.cond.notify_all()

    @staticmethod
    def handle_response(self, msg):
        engine_id = msg.engine_id
        self.finish_req_num[engine_id] += 1
        logger.info("engine_id={} send_req_num={} finish_req_num={}",
                    engine_id, self.send_req_num, self.finish_req_num)

class Client:
    def __init__(self, config):
        self.config = config
        self.out_queue = mp.Queue()
        self.loop = None
        self.response_queues = dict()
        self.request_factory = RequestFactory(config)
        self.engine_manager = EngineManager(self.config, self.out_queue)

        handlers = {
            MessageType.engine_start: [
                (self.engine_manager, self.engine_manager.handle_engine_start)
            ],
            MessageType.response: [
                (self.engine_manager, self.engine_manager.handle_response),
                (self, self.handle_response)
            ],
        }
        self.listener = Listener(self.out_queue, handlers)

        self.engine_manager.start()

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
        logger.info("req start add_request")
        self.engine_manager.add_request(req)

        now_loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = now_loop
        else:
            assert self.loop == now_loop

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

    @staticmethod
    def handle_response(self, req):
        # 说明还没有获取请求
        if self.loop is None:
            return

        if req.id in self.response_queues:
            self.loop.call_soon_threadsafe(
                self.response_queues[req.id].put_nowait, req.tokens)
            logger.info("req enter response_queue, tokens={}, que_id={}, loop_id={} engine_id={}",
                        self.response_queues[req.id], id(self.response_queues[req.id]),
                        id(self.loop), req.engine_id)
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
    data_parallel=2,
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

