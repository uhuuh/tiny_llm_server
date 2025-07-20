import asyncio
import os
from queue import Empty
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sympy import asinh

from engine import Engine, RequestFactory, Request
import multiprocessing as mp
from loguru import logger

from src.base import SampleConfig, InferConfig, Config


class RequestParam(BaseModel):
    model: str
    prompt: List[int] # TODO should str
    temperature: float
    max_tokens: int
    stream: bool

def request_callback(req: Request, out_queue: mp.Queue):
    # NOTE 回调在另外一个进程被调用，使用mp而非asyncio的queue传递输出
    if req.is_finish():
        out_queue.put_nowait(req)
        logger.info("req enter out_queue")

class Client:
    def __init__(self, config):
        self.config = config
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.response_queues = dict()
        self.request_factory = RequestFactory(config)

        self.schedule_proc = mp.Process(target=self.run_engine)
        self.schedule_proc.start()
        # TODO 等待engine启动ok之后，再初始化完成
        self.has_handle_output_loop = False

    def run_engine(self):
        engine = Engine(self.config, self.in_queue, self.out_queue)
        engine.run()

    async def generate_response(self, param: RequestParam):
        if not self.has_handle_output_loop:
            # NOTE 需要在任意一个async def中添加task
            self.has_handle_output_loop = True
            asyncio.create_task(self.handle_output_loop())
            logger.info("create handle_output_loop ok")

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

    async def handle_output_loop(self):
        # TODO need opt
        while True:
            try:
                req = self.out_queue.get_nowait()
                if req.id in self.response_queues:
                    await self.response_queues[req.id].put(req.tokens)
                    logger.info("req enter response_queue, tokens={}, que_id={}, loop_id={}",
                                self.response_queues[req.id], id(self.response_queues[req.id]),
                                id(asyncio.get_event_loop()))
                else:
                    raise
            except Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                raise e


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

