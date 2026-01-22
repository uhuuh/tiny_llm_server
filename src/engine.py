import multiprocessing as mp
import threading as th
from typing import List

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

from src.base import Config, SampleConfig
from src.scheduler import Scheduler
from src.utils import ProcessExecutor
from src.worker import Worker
from src.server import ChatCompletionRequest, ChatCompletionRequestResult
from src.base import SchedulerInitEndMessage, SchedulerReqFinishMessage, SchedulerReqRecvMessage


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.send_req_num = np.zeros(config.infer_config.data_parallel)
        self.finish_req_num = np.zeros(config.infer_config.data_parallel)
        self.tokenizer = AutoTokenizer.from_pretrained(config.infer_config.model_path, use_fast=True)

        self._init_other()

    def add_requests(self, reqs: List[ChatCompletionRequest]):
        all_prompt_text = []
        for r in reqs:
            t = self.tokenizer.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
            all_prompt_text.append(t)
        all_prompt_tokens = self.tokenizer.batch_encode(all_prompt_text)

        msg = SchedulerReqRecvMessage(requests=[])
        for i, r in enumerate(reqs):
            msg.requests.append(SchedulerReqRecvMessage.RequestInputInfo(
                request_id=r.request_id,
                prompt_tokens=all_prompt_tokens[i],
                sample_config=SampleConfig(
                    top_p=r.top_p,
                    temperature=r.temperature,
                    max_tokens=r.max_tokens,
                ),
            ))

        target_idx = np.argmin(self.send_req_num) # TODO 是否应该选择正在运行最小的scheduler
        self.send_req_num[target_idx] += len(reqs)
        self.scheduler_in_queues[target_idx].put_nowait(msg)

    def get_scheduler_id(self, dp_idx: int) -> str:
        return f"scheduler-{dp_idx}"

    def get_worker_id(self, dp_idx: int, tp_idx: int) -> str:
        return f"worker-{dp_idx}-{tp_idx}"

    def parse_scheduler_id(self, scheduler_id) -> int:
        return int(scheduler_id.split('-')[1])

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
                        "id": self.get_worker_id(tp_i, tp_i),
                        "config": self.config,
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
                    "id": self.get_scheduler_id(dp_i),
                    "config": self.config,
                    "in_queue": self.scheduler_in_queues[dp_i],
                    "out_queue": self.scheduler_out_queues,
                    "worker_in_queues": self.worker_in_queues[dp_i * self.tp: dp_i * self.tp + self.tp],
                    "worker_out_queue": self.worker_out_queues[dp_i],
                },
                io_methods=["recv_req_loop", "step_loop"]
            )
            p.start()
            self.procs.append(p)

        self.engine_ok_nun = 0
        while self.engine_ok_nun < self.dp:
            msg: SchedulerInitEndMessage = self.scheduler_out_queues.get()
            logger.info("recv scheduler {} init finished", msg.scheduler_id)
            self.engine_ok_nun += 1
        logger.info("all scheduler init finished")

    def step(self) -> List[ChatCompletionRequestResult]:
        msg: SchedulerReqFinishMessage = self.scheduler_out_queues.get()
        self.finish_req_num[self.parse_scheduler_id(msg.scheduler_id)] += 1

        msg2 = []
        for r in msg.requests:
            msg2.append(ChatCompletionRequestResult(
                id=r.request_id,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.output_tokens,
                output_text=r.output_text,
            ))

        return msg2
