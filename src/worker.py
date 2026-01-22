import glob
import os
from collections import deque
from math import floor
from typing import List
import multiprocessing as mp
import threading as th

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file

from src.base import Config, ScheduleInfo, SampleConfig, ceil_div, WorkerInitEndMessage
from src.model import ModelInput, RotaryPositionalEmbedding
from src.sequence import Sequence


class Worker:
    def __init__(self, id, config: Config, in_queue: mp.Queue, out_queue: mp.Queue):
        self.id = id
        self.config = config
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.model = self.get_model()
        self.pos_emb_manager = RotaryPositionalEmbedding(
            param_dtype=config.model_config.get_param_dtype(),
            dim=config.model_config.hidden_size // config.model_config.num_attention_heads,
            max_seq_len=config.model_config.max_position_embeddings
        )
        # worker 确实应该持有cache_storager，一个是worker是每张卡都有，而engine是每dp域有,
        # 二是所有分配cuda显存应该在一个进程中
        self.cache_storager = CacheStorager(config)
        self.warm_up()
        logger.info("worker {} init finished", self.id)
        self.out_queue.put_nowait(WorkerInitEndMessage(worker_id=self.id, block_num=self.cache_storager.block_num))

    def get_model(self):
        # TODO below should clear
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.bfloat16)

        from src.model import Qwen2

        if self.config.infer_config.enable_debug:
            logger.info("enable debug mode")
            self.config.model_config.num_hidden_layers = 2
        model = Qwen2(self.config)
        model = model.to("cuda").eval()

        if not self.config.infer_config.enable_debug:
            self.load_weight(model, self.config.infer_config.model_path)

        return model

    def load_weight(self, model, dir_path: str, device="cuda"):
        param_dict = {}
        pattern = os.path.join(dir_path, "*.safetensors")
        for path in sorted(glob.glob(pattern)):
            tensors = load_file(path, device=device)
            param_dict.update(tensors)

        tp_size = self.config.parallel_config.tp_size
        tp_rank = self.config.parallel_config.tp_rank # NOTE
        split_strategy = {
            "q_proj.weight": (1, tp_size),
            "k_proj.weight": (1, tp_size),
            "v_proj.weight": (1, tp_size),
            "o_proj.weight": (tp_size, 1),
            "down_proj.weight": (1, tp_size),
            "gate_proj.weight": (1, tp_size),
            "up_proj.weight": (tp_size, 1),
        }

        for name, param in param_dict.items():
            if name in split_strategy:
                splits = split_strategy[name]
                for a, s in enumerate(splits):
                    if s != 1:
                        param = param.chunk(s, dim=a)[tp_rank]

            param_dict[name] = param

        missing_keys, unexpected_keys = model.load_state_dict(param_dict, strict=False)
        logger.info("load weight missing_keys: {} unexpected_keys: {}", missing_keys, unexpected_keys)

    def warm_up(self):
        max_batch_token_num = self.config.infer_config.max_batch_token_num
        max_req_num = self.config.infer_config.max_req_num
        dummy_block_num = max_batch_token_num

        def get_dummy_reqs():
            schedule_info = ScheduleInfo()
            dummy_reqs = []
            sample_config = SampleConfig()
            req_id = 0
            all_tokens = list(range(max_batch_token_num))
            pre_step = 0
            for a in range(max_req_num):
                step = ((max_batch_token_num // max_req_num) +
                        int(a < (max_batch_token_num % max_req_num)))
                tokens = all_tokens[pre_step: pre_step + step]
                pre_step += step

                schedule_info.req_new_token_nums[req_id] = step
                schedule_info.now_req_num += 1
                schedule_info.req_new_token_nums[req_id] = step

                req = Sequence(req_id, tokens, self.config.infer_config.block_size, sample_config, None)
                req_id += 1
                dummy_reqs.append(req)
            return dummy_reqs, schedule_info

        def set_dummy_cache(dummy_reqs, dummy_schedule_info):
            dummy_block_num = max_batch_token_num
            self.cache_storager.set_cache(dummy_block_num)
            free_blocks = deque(list(range(max_batch_token_num)))
            for req in dummy_reqs:
                new_token_num = dummy_schedule_info.req_new_token_nums[req.id]
                need_block_num = ceil_div(new_token_num, self.config.infer_config.block_size)
                for _ in range(need_block_num):
                    block_id = free_blocks.popleft()
                    req.block_table.append(block_id)
                    start_slot_id = block_id * self.config.infer_config.block_size
                    req.slot_mapping.extend(range(start_slot_id, start_slot_id + self.config.infer_config.block_size))

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        param_gpu_memory = torch.cuda.memory_allocated()

        dummy_reqs, dummy_schedule_info = get_dummy_reqs()
        set_dummy_cache(dummy_reqs, dummy_schedule_info)

        before_allocated_gpu_memory = torch.cuda.memory_allocated()
        self.step(dummy_reqs, dummy_schedule_info)
        torch.cuda.synchronize()
        after_allocated_gpu_memory = torch.cuda.memory_allocated()
        assert before_allocated_gpu_memory == after_allocated_gpu_memory

        activation_gpu_memory = torch.cuda.max_memory_allocated() - after_allocated_gpu_memory
        _, total_gpu_memory = torch.cuda.mem_get_info()
        cache_gpu_memory = (floor(total_gpu_memory * self.config.infer_config.gpu_memory_utilization) -
                            activation_gpu_memory - param_gpu_memory)
        dummy_cache_gpu_memory = dummy_block_num * self.cache_storager.block_byte_num()
        assert cache_gpu_memory >= dummy_cache_gpu_memory

        block_num = ceil_div(cache_gpu_memory, self.cache_storager.block_byte_num())
        self.cache_storager.set_cache(block_num)
        logger.info("param_gpu_memory={:,} activation_gpu_memory={:,} cache_gpu_memory={:,} "
                    "block_num={:,} block_size={}",
                    param_gpu_memory, activation_gpu_memory, cache_gpu_memory,
                    block_num, self.config.infer_config.block_size)


    def step(self, request_list: List[Sequence], info: ScheduleInfo):
        # NOTE: prefill req must in head
        request_list.sort(key=lambda req: req.computed_len)

        prefill_token_num = 0
        input_ids = []
        position_ids = []
        slot_mapping = []
        block_tables = []
        seq_lens = []
        max_seq_q_len = 0
        cu_seq_q_lens = [0]
        decode_max_seq_len = 0
        decode_max_seq_q_len = 0
        decode_cu_seq_q_lens = [0]

        for req in request_list:
            seq_q_len = info.req_new_token_nums[req.id]
            input_ids.extend(req.tokens[req.computed_len: req.computed_len + seq_q_len])
            position_ids.extend(range(req.computed_len, req.computed_len + seq_q_len))
            slot_mapping.extend(req.slot_mapping[req.computed_len: req.computed_len + seq_q_len])

            if req.computed_len == 0:
                prefill_token_num += seq_q_len

                max_seq_q_len = max(max_seq_q_len, seq_q_len)
                cu_seq_q_lens.append(cu_seq_q_lens[-1] + seq_q_len)
            else:
                # TODO computed_len + seq_q_len = real seq_len
                block_tables.append(req.block_table)
                seq_lens.append(req.computed_len + seq_q_len)
                decode_max_seq_len = max(decode_max_seq_len, req.computed_len + seq_q_len)
                decode_max_seq_q_len = max(decode_max_seq_q_len, seq_q_len)
                decode_cu_seq_q_lens.append(decode_cu_seq_q_lens[-1] + seq_q_len)

        logger.info(f">>> debug input_ids={input_ids}")
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        position_ids = torch.tensor(position_ids, dtype=torch.int64)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        cos, sin = self.pos_emb_manager.get_cos_sin(position_ids)

        prefill_cu_seqlens_q = torch.tensor(cu_seq_q_lens, dtype=torch.int32)
        prefill_max_seq_len_q = torch.tensor(max_seq_q_len, dtype=torch.int32)

        max_block_num_per_seq = max(map(len, block_tables)) if block_tables else 0
        decode_block_table = torch.tensor([t + [-1] * (max_block_num_per_seq - len(t)) for t in block_tables]
                                   , dtype=torch.int32)
        decode_seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        decode_max_seq_len = torch.tensor(decode_max_seq_len, dtype=torch.int32)
        decode_max_seq_q_len = torch.tensor(decode_max_seq_q_len, dtype=torch.int32)
        decode_cu_seq_q_lens = torch.tensor(decode_cu_seq_q_lens, dtype=torch.int32)

        inp = ModelInput(
            input_ids=input_ids,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            num_prefill_tokens=prefill_token_num,
            k_cache=self.cache_storager.k_cache,
            v_cache=self.cache_storager.v_cache,
            slot_mapping=slot_mapping,
            prefill_cu_seqlens_q=prefill_cu_seqlens_q,
            prefill_max_seqlen_q=prefill_max_seq_len_q,
            decode_block_table=decode_block_table,
            decode_seq_lens=decode_seq_lens,
            decode_max_seq_len=decode_max_seq_len,
            decode_max_seq_q_len=decode_max_seq_q_len,
            decode_cu_seq_q_lens=decode_cu_seq_q_lens,
        )
        logger.info("model_input={}", inp)
        output = self.model(inp)

        finish_requests = []
        unfinish_requests = []

        start_idx = 0
        for req in request_list:
            seq_q_len = info.req_new_token_nums[req.id]
            if req.computed_len + seq_q_len >= req.seq_len:
                logits = output[start_idx: start_idx + seq_q_len][-1]
                next_token = torch.argmax(logits, dim=-1).tolist()
                req.append_token(seq_q_len, next_token)
            else:
                req.append_token(seq_q_len, None)

            if req.is_finish():
                finish_requests.append(req)
            else:
                unfinish_requests.append(req)

            start_idx += seq_q_len

        logger.info("step finished_req={} unfinished_req={}",
                    [req.tokens for req in finish_requests], [req.tokens for req in unfinish_requests])

        return finish_requests, unfinish_requests

    def step_loop(self):
        while True:
            msg, args = self.in_queue.get()
            assert msg == MessageType.worker_step_start
            ret = self.step(*args)
            self.out_queue.put((MessageType.worker_step_end, ret))

class CacheStorager:
    def __init__(self, config: Config):
        self.param_dtype = config.model_config.get_param_dtype()
        self.layer_num = config.model_config.num_hidden_layers
        head_dim = config.model_config.hidden_size // config.model_config.num_attention_heads
        head_kv_num = config.model_config.num_key_value_heads
        block_size = config.infer_config.block_size
        self.block_shape = [block_size, head_kv_num, head_dim]
        self.k_cache = None
        self.v_cache = None

    def block_byte_num(self):
        single = torch.empty((), dtype=self.param_dtype).element_size()
        # 2 is k and v
        return 2 * self.layer_num * np.prod(self.block_shape) * single

    def set_cache(self, block_num):
        del self.k_cache
        del self.v_cache

        self.block_num = block_num
        kv_cache_shape = [block_num] + self.block_shape
        self.k_cache = [torch.empty(kv_cache_shape, dtype=self.param_dtype, device="cuda")
                        for _ in range(self.layer_num)]
        self.v_cache = [torch.empty(kv_cache_shape, dtype=self.param_dtype, device="cuda")
                        for _ in range(self.layer_num)]
