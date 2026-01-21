import asyncio
import os

from fastapi import FastAPI

from src.engine import request_callback, Engine
from src.squence import RequestFactory
import multiprocessing as mp
from loguru import logger

from src.base import SampleConfig, InferConfig, Config, RequestParam, MessageType, Listener
from src.utils import get_config
from src.server import app


class Client:
    def __init__(self, config):
        self.config = config
        self.out_queue = mp.Queue()
        self.loop = None
        self.response_queues = dict()
        self.request_factory = RequestFactory(config)
        # self.engine_manager = Engine(self.config, self.out_queue)

        # handlers = {
        #     MessageType.scheduler_init_end: [
        #         (self.engine_manager, self.engine_manager.handle_engine_start)
        #     ],
        #     MessageType.scheduler_req_finish: [
        #         (self.engine_manager, self.engine_manager.handle_response),
        #         (self, self.handle_response)
        #     ],
        # }
        # self.listener = Listener(self.out_queue, handlers)
        #
        # self.engine_manager.start()

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

# import signal
# import sys
# def shutdown_handler(sig, frame):
#     print("Shutting down gracefully...")
#     sys.exit(0)
# signal.signal(signal.SIGINT, shutdown_handler)
# signal.signal(signal.SIGTERM, shutdown_handler)
#
#
# config = get_config()
# client = Client(config)
# app = FastAPI()

# @app.post("/")
# async def generate(request: RequestParam):
#     return await client.generate_response(request)

if __name__ == "__main__":
    # # TODO 如何抢占端口
    # os.system("kill -9 $(lsof -t -i:8001)")
    # os.system("kill -9 $(lsof -t -i:8001)")
    # os.system("kill -9 $(lsof -t -i:8001)")
    config = get_config()


    import uvicorn
    uvicorn.step_loop(app, host="0.0.0.0", port=8001)

