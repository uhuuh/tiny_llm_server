import os

import torch
from transformers import AutoTokenizer

from src.base import SampleConfig, InferConfig, Config
from src.engine import Engine
from src.squence import RequestFactory, Sequence
from loguru import logger
from torch import distributed as dist
import numpy as np


if __name__ == '__main__':
    def init_dist(config):
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        tp = config.parallel_config.tp_size
        dp = config.parallel_config.dp_size
        groups = np.arange(world_size).reshape(-1, dp, tp)
        for group in groups.transpose((0, 2, 1)).reshape(-1, dp):
            if rank in group:
                config.parallel_config.dp_rank = np.where(group == rank)[0].item()
                config.parallel_config.dp_group = dist.new_group(group)
        for group in groups.reshape(-1, tp):
            if rank in group:
                config.parallel_config.tp_rank = np.where(group == rank)[0].item()
                config.parallel_config.tp_group = dist.new_group(group)
        logger.info(f"rank={rank} world_size={world_size} parallel_context={config.parallel_config}")

    sample_config = SampleConfig()
    infer_config = InferConfig(
        model_path="/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct",
        max_req_num=254,
        max_batch_token_num=4094,
        gpu_memory_utilization=0.3,
        block_size=16,
        enable_prefix_cache=True,
        enable_debug=True,
        max_prefill_len=30,
        enable_chunked_prefill=True,
        data_parallel=1,
        tensor_parallel=1,
    )
    config = Config(
        infer_config=infer_config,
    )

    init_dist(config)

    scheduler = Engine(0, config, None, None)
    tokenizer = AutoTokenizer.from_pretrained(config.infer_config.model_path)
    inputs = [
        [19808, 279, 13458, 88, 9104, 323, 279, 6783, 11980, 4889, 11, 1340, 8570, 5290, 1059, 6930, 22531, 11514, 553,
         277, 39411, 11, 274, 5654, 4017, 17931, 13970, 11, 22930, 2770, 8974, 518, 279, 11174, 33019, 7741, 3241, 11,
         5556, 304, 3381, 11, 6587, 41001, 304, 279, 1879, 315, 71830, 323, 27799, 13],
        [19808, 279, 13458, 88, 9104, 323, 279, 6783, 11980, 4889, 11, 1340, 8570, 5290, 1059, 6930, 22531, 11514, 553,
         277, 39411, 11, 274, 5654, 4017, 17931, 13970, 11, 22930, 2770, 8974, 518, 279, 11174, 33019, 7741, 3241, 11,
         5556, 304, 3381, 11, 6587, 41001, 304, 279, 1879, 315, 71830, 323, 27799, 13],
    ]
    req_factory = RequestFactory(config)

    def step_fun(req: Sequence, context):
        if req.is_finish():
            print(f"req_id={req.id} output={tokenizer.decode(req.tokens)}")


    for inp in inputs:
        scheduler.add_wait_request(req_factory.create(inp, sample_config, step_fun=step_fun))

    # NOTE: because chunked prefill, has some step not generate new token
    for _ in range(sample_config.max_new_token_new + 3):
        scheduler.step()
