import multiprocessing as mp
import threading as th

import numpy as np
from loguru import logger

from src.base import MessageType, Config
from src.scheduler import Scheduler
from src.squence import Sequence
from src.utils import ProcessExecutor
from src.worker import Worker
from transformers import AutoTokenizer
from server import ChatCompletionRequest


def request_callback(req: Sequence, engine: Scheduler):
    # NOTE 回调在另外一个进程被调用，使用mp而非asyncio的queue传递输出
    if req.is_finish():
        req.engine_id = engine.id
        engine.out_queue.put_nowait((MessageType.scheduler_req_finish, req))
        logger.info("req enter out_queue")

class Engine:
    def __init__(self, config: Config):
        self.config = config

        self.send_req_num = np.zeros(self.dp)
        self.finish_req_num = np.zeros(self.dp)
        self.tokenizer = AutoTokenizer.from_pretrained(config.infer_config.model_path, use_fast=True)

        self._init_other()

    def add_request(self, req: ChatCompletionRequest):
        target_idx = np.argmin(self.send_req_num)
        self.send_req_num[target_idx] += 1
        self.scheduler_in_queues[target_idx].put_nowait((MessageType.scheduler_req_recv, req))
        logger.info("req enter in_queue, engine_id={}, send_req_nm={}, finish_req_num={}",
                    target_idx, self.send_req_num, self.finish_req_num)

    def _init_other(self):
        self.dp = self.config.infer_config.data_parallel
        self.tp = self.config.infer_config.tensor_parallel

        self.scheduler_in_queues = [mp.Queue() for _ in range(self.dp)]
        self.scheduler_out_queues = mp.Queue()
        self.worker_in_queues = [mp.Queue() for _ in range(self.dp * self.tp)]
        self.worker_out_queues = [mp.Queue() for _ in range(self.dp)]
        self.procs = []

        for dp_i in range(self.dp):
            for tp_i in range(self.tp):
                p = ProcessExecutor(
                    Worker,
                    cls_kwargs={
                        "id": f"worker-{dp_i}-{tp_i}",
                        "in_queue": self.worker_in_queues[dp_i * self.tp + tp_i],
                        "out_queue": self.worker_out_queues[dp_i]
                    },
                    io_methods=["step_loop"]
                )
                p.start()
                self.procs.append(p)
            p = ProcessExecutor(
                Scheduler,
                cls_kwargs={
                    "id": f"scheduler-{dp_i}",
                    "in_queue": self.scheduler_in_queues[dp_i],
                    "out_queue": self.scheduler_out_queues,
                    "worker_in_queue": self.worker_in_queues[dp_i * self.tp: dp_i * self.tp + self.tp],
                    "worker_out_queue": self.worker_out_queues[dp_i],
                },
                io_methods=["recv_req_loop", "step_loop"]
            )
            p.start()
            self.procs.append(p)

        self.engine_ok_nun = 0
        while self.engine_ok_nun < self.dp:
            msg = self.scheduler_out_queues.get()
            # TODO 检查消息类型
            self.engine_ok_nun += 1

    def step(self):
        msg, res = self.scheduler_out_queues.get()
        # TODO 维护一个统计信息, 哪一个调度器处理了请求, 以便发送请求时负载均衡
        return res
