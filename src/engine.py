import gc
from collections import deque
from queue import Empty
from threading import Thread
from typing import Dict

import numpy as np
from numba.tests.test_gil import sleep
from transformers import AutoTokenizer

from base import *
import torch
from loguru import logger
from math import floor

class RequestFactory:
    def __init__(self, config):
        self.id = 0
        self.config = config

    def create(self, tokens, sample_config, step_fun):
        req = Request(self.id, tokens, self.config.infer_config.block_size, sample_config, step_fun)
        self.id += 1
        return req

class Request:
    def __init__(self, id, tokens, block_size, sample_config: SampleConfig, step_fun):
        self.id = id
        self.tokens = tokens
        self.slot_mapping = []
        self.block_table = []

        self.in_seq_len = len(tokens)
        self.out_seq_len = 0
        self.seq_len = self.in_seq_len + self.out_seq_len
        self.computed_len = 0
        self.cached_len = 0

        self.sample_config = sample_config
        self.block_size = block_size
        self.step_fun = step_fun

    @property
    def allocated_len(self):
        return self.block_size * len(self.block_table)

    @property
    def allocated_block_num(self):
        return len(self.block_table)

    def get_block_content(self, block_idx):
        assert 0 <= block_idx * self.block_size < self.seq_len
        start_token_idx = block_idx * self.block_size
        return self.tokens[start_token_idx: start_token_idx + self.block_size]

    def append_block(self, block_id, is_computed, is_cached):
        # NOTE: when allocated no cached block, when no cached block has full tokens, should cache this block
        if block_id is not None:
            self.block_table.append(block_id)
            start_slot_id = block_id * self.block_size
            self.slot_mapping.extend(list(range(start_slot_id, start_slot_id + self.block_size)))
        if is_computed:
            self.computed_len += self.block_size
        if is_cached:
            self.cached_len += self.block_size

    def append_token(self, computed_token_num, next_token):
        # NOTE: when chunk prefill, some step not append token, but increase computed len
        self.computed_len += computed_token_num
        if next_token is not None:
            self.tokens.append(next_token)
            self.out_seq_len += 1
            self.seq_len += 1

    def is_finish(self):
        if self.out_seq_len >= self.sample_config.max_new_token_new:
            return True
        return False

class BlockManager:
    def __init__(self, block_num, block_size, enable_prefix_cache):
        self.block_size = block_size
        self.free_blocks = None
        self.block_refs = None
        self.block_to_hashs = dict()
        self.hash_to_blocks = dict()
        self.enable_prefix_cache = enable_prefix_cache

    def reset(self, block_num):
        del self.free_blocks
        del self.block_refs
        del self.block_to_hashs
        del self.hash_to_blocks

        self.free_blocks = deque(range(block_num))
        self.block_refs = [0 for _ in range(block_num)]
        self.block_to_hashs = dict()
        self.hash_to_blocks = dict()

    def get_cached_block(self, block_hash):
        return self.hash_to_blocks.get(block_hash, None)

    def cache_block(self, block_id, block_hash):
        if not self.enable_prefix_cache:
            return

        assert self.block_refs[block_id] > 0

        self.block_to_hashs[block_id] = block_hash
        self.hash_to_blocks[block_hash] = block_id

    def can_alloc(self, block_num):
        return len(self.free_blocks) >= block_num

    def alloc_new_block(self):
        block_id = self.free_blocks.popleft()
        self.block_refs[block_id] += 1

        # NOTE: reset this block prefix cache info
        old_block_hash = self.block_to_hashs.get(block_id, None)
        if old_block_hash is not None:
            self.hash_to_blocks.pop(old_block_hash)
            self.block_to_hashs.pop(block_id)

        return block_id

    def free_block(self, block_id):
        self.block_refs[block_id] -= 1
        if self.block_refs[block_id] == 0:
            self.free_blocks.append(block_id)

class CacheManager:
    def __init__(self, block_num, block_size, enable_prefix_cache):
        self.block_size = block_size
        self.block_manager = BlockManager(block_num, block_size, enable_prefix_cache)

    def set_cache(self, block_num):
        self.block_manager.reset(block_num)

    @staticmethod
    def _get_req_block_hash(req, block_idx):
        if block_idx == 0:
            return None, tuple(req.get_block_content(block_idx))
        else:
            return tuple(req.get_block_content(block_idx - 1)), tuple(req.get_block_content(block_idx))

    def can_alloc(self, req, new_token_num):
        if req.allocated_len >= req.seq_len:
            return True

        next_computed_len = min(req.seq_len, req.computed_len + new_token_num)
        need_new_block_num = ceil_div(next_computed_len - req.allocated_len, req.block_size)
        return self.block_manager.can_alloc(need_new_block_num)

    def cache_computed_block(self, req):
        if req.computed_len - req.cached_len >= req.block_size:
            block_idx = req.cached_len // self.block_size
            self.block_manager.cache_block(req.block_table[block_idx], self._get_req_block_hash(req, block_idx))
            req.append_block(None, False, True)

    def alloc_prefix_cached_blocks(self, req: Request):
        if req.computed_len != 0:
            return

        # NOTE: keep last block to compute next token in prefix cache
        for a in range((req.in_seq_len - 1) // req.block_size):
            block_id = self.block_manager.get_cached_block(self._get_req_block_hash(req, a))
            if block_id is not None:
                req.append_block(block_id, True, True)
            else:
                break

    def alloc_new_blocks(self, req: Request, max_new_token: int):
        if req.allocated_len >= req.seq_len:
            return True

        need_block_num = ceil_div(req.computed_len + max_new_token - req.allocated_len, self.block_size)
        for _ in range(need_block_num):
            block_id = self.block_manager.alloc_new_block()
            req.append_block(block_id, False, False)

    def free_cache(self, req: Request):
        for block_id in req.block_table:
            self.block_manager.free_block(block_id)

@dataclass
class ScheduleInfo:
    k_cache: Any
    v_cache: Any
    now_batch_token_num: int = 0
    now_req_num: int = 0
    req_new_token_nums: Dict[int, int] = field(default_factory=dict)

# TODO from Engine detach Scheduler
class Engine:
    def __init__(self, config: Config, in_queue, out_queue):
        self.config = config
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.run_queue = deque()
        self.wait_queue = deque()

        self.enable_chunked_prefill = config.infer_config.enable_chunked_prefill
        self.max_prefill_len = config.infer_config.max_prefill_len if self.enable_chunked_prefill else float("inf")
        self.max_req_num = config.infer_config.max_req_num
        self.max_batch_token_num = config.infer_config.max_batch_token_num
        self.gpu_memory_utilization = config.infer_config.gpu_memory_utilization

        self.cache_manager = CacheManager(
            block_num=ceil_div(self.max_batch_token_num, config.infer_config.block_size),
            block_size=config.infer_config.block_size,
            enable_prefix_cache=config.infer_config.enable_prefix_cache
        )
        self.cache_storager = CacheStorager(config)
        self.worker = Worker(config)

        self.warm_up()
        logger.info("init scheduler ok")

    def handle_in_loop(self):
        while True:
            try:
                # TODO 可以等待事件方式触发吗？而不是忙等
                req = self.in_queue.get()
                logger.info("req left in_queue")
                # NOTE: wait queue is thread safe
                self.add_wait_request(req)
            except Empty:
                continue
            except Exception as e:
                raise e

    def scheduler_loop(self):
        while True:
            self.step()

    def run(self):
        # TODO this is ok?
        t_handle_in = Thread(target=self.handle_in_loop)
        t_handle_in.start()

        self.scheduler_loop()

    def warm_up(self):
        # TODO should refactor
        req_id = 0
        all_tokens = list(range(self.max_batch_token_num))
        sample_config = SampleConfig()
        pre_step = 0
        for a in range(self.max_req_num):
            step = ((self.max_batch_token_num // self.max_req_num) +
                    int(a < (self.max_batch_token_num % self.max_req_num)))
            tokens = all_tokens[pre_step: pre_step + step]
            pre_step += step

            req = Request(req_id, tokens, self.config.infer_config.block_size, sample_config, None)
            req_id += 1
            self.add_wait_request(req)

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        param_gpu_memory = torch.cuda.memory_allocated()

        # NOTE ...
        dummy_block_num = self.max_batch_token_num
        dummy_cache_gpu_memory = dummy_block_num * self.cache_storager.block_byte_num()
        self.cache_manager.set_cache(dummy_block_num)
        self.cache_storager.set_cache(dummy_block_num)

        before_allocated_gpu_memory = torch.cuda.memory_allocated()
        self.step()
        self.wait_queue.clear()
        torch.cuda.synchronize()
        after_allocated_gpu_memory = torch.cuda.memory_allocated()
        assert before_allocated_gpu_memory == after_allocated_gpu_memory

        activation_gpu_memory = torch.cuda.max_memory_allocated() - after_allocated_gpu_memory
        _, total_gpu_memory = torch.cuda.mem_get_info()
        cache_gpu_memory = (floor(total_gpu_memory * self.gpu_memory_utilization) -
                            activation_gpu_memory - param_gpu_memory)
        assert cache_gpu_memory >= dummy_cache_gpu_memory

        block_num = ceil_div(cache_gpu_memory, self.cache_storager.block_byte_num())
        self.cache_manager.set_cache(block_num)
        self.cache_storager.set_cache(block_num)
        logger.info("param_gpu_memory={:,} activation_gpu_memory={:,} cache_gpu_memory={:,} "
                    "block_num={:,} block_size={}",
                    param_gpu_memory, activation_gpu_memory, cache_gpu_memory,
                    block_num, self.cache_manager.block_size)

    def add_wait_request(self, request):
        self.wait_queue.append(request)
        logger.info(f"add_wait_request req_id={request.id} done")

    def step(self):
        if not self.wait_queue:
            return

        self.run_queue.clear()
        info = ScheduleInfo(
            k_cache=self.cache_storager.k_cache,
            v_cache=self.cache_storager.v_cache
        )

        while (self.wait_queue and info.now_req_num < self.max_req_num
               and info.now_batch_token_num < self.max_batch_token_num):
            req = self.wait_queue[0]
            # NOTE ...
            self.cache_manager.alloc_prefix_cached_blocks(req)
            # NOTE ...
            self.cache_manager.cache_computed_block(req)

            new_token_num = min(req.seq_len - req.computed_len, self.max_batch_token_num - info.now_batch_token_num)
            new_token_num = min(self.max_prefill_len, new_token_num)

            if self.cache_manager.can_alloc(req, new_token_num):
                self.cache_manager.alloc_new_blocks(req, new_token_num)

                self.wait_queue.popleft()
                self.run_queue.append(req)

                info.now_req_num += 1
                info.now_batch_token_num += new_token_num
                info.req_new_token_nums[req.id] = new_token_num
            else:
                # TODO 进行cache释放
                break

        if not self.run_queue: return
        finish_requests, unfinished_requests = self.worker.step(list(self.run_queue), info) # TODO need optimize
        # TODO should refactor
        for req in self.run_queue:
            if req.step_fun is not None:
                req.step_fun(req, self.out_queue)

        # NOTE: first use unfinished req
        self.wait_queue.extendleft(unfinished_requests)

        for req in finish_requests:
            self.cache_manager.free_cache(req)
        return finish_requests

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

        kv_cache_shape = [block_num] + self.block_shape
        self.k_cache = [torch.empty(kv_cache_shape, dtype=self.param_dtype, device="cuda")
                        for _ in range(self.layer_num)]
        self.v_cache = [torch.empty(kv_cache_shape, dtype=self.param_dtype, device="cuda")
                        for _ in range(self.layer_num)]


def get_model(config: Config):
    # TODO below should clear
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    from qwen2_5 import Qwen2

    if config.infer_config.enable_debug:
        logger.info("enable debug mode")
        config.model_config.num_hidden_layers = 2
    model = Qwen2(config.model_config)
    model = model.to("cuda").eval()

    if not config.infer_config.enable_debug:
        load_weight(model, config.infer_config.model_path)

    return model

class Worker:
    def __init__(self, config: Config):
        self.model = get_model(config)
        self.pos_emb_manager = RotaryPositionalEmbedding(
            param_dtype=config.model_config.get_param_dtype(),
            dim=config.model_config.hidden_size // config.model_config.num_attention_heads,
            max_seq_len=config.model_config.max_position_embeddings
        )

    def step(self, request_list: List[Request], info: ScheduleInfo):
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
            k_cache=info.k_cache,
            v_cache=info.v_cache,
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

if __name__ == '__main__':

    sample_config = SampleConfig()
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
    scheduler = Engine(config, None, None)

    tokenizer = AutoTokenizer.from_pretrained(config.infer_config.model_path)

    inputs = [
        [19810, 279, 13458, 88, 9104, 323, 279, 6783, 11980, 4889, 11, 1340, 8570, 5290, 1059, 6930, 22531, 11514, 553,
         279, 39411, 11, 274, 5654, 4017, 17931, 13970, 11, 22930, 2770, 8974, 518, 279, 11174, 33019, 7741, 3241, 11,
         5558, 304, 3381, 11, 6587, 41001, 304, 279, 1879, 315, 71830, 323, 27799, 13],
        [19810, 279, 13458, 88, 9104, 323, 279, 6783, 11980, 4889, 11, 1340, 8570, 5290, 1059, 6930, 22531, 11514, 553,
         279, 39411, 11, 274, 5654, 4017, 17931, 13970, 11, 22930, 2770, 8974, 518, 279, 11174, 33019, 7741, 3241, 11,
         5558, 304, 3381, 11, 6587, 41001, 304, 279, 1879, 315, 71830, 323, 27799, 13],
    ]

    req_factory = RequestFactory(config)
    def step_fun(req: Request, out_queue):
        if req.is_finish():
            print(f"req_id={req.id} output={tokenizer.decode(req.tokens)}")

    for inp in inputs:
        scheduler.add_wait_request(req_factory.create(inp, sample_config, step_fun=step_fun))

    # NOTE: because chunked prefill, has some step not generate new token
    for _ in range(sample_config.max_new_token_new + 5):
        scheduler.step()
