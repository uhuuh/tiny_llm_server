import queue
from collections import Counter, deque, defaultdict
from dataclasses import dataclass

from model.opt2 import OPTConfig, SamplerConfig, OPTForCausalLM, get_torch_dtype
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger

@dataclass
class ServerConfig:
    gpu_block_num: int
    cpu_block_num: int
    block_size: int

@dataclass
class Config:
    model_config: OPTConfig
    sample_config: SamplerConfig
    server_config: ServerConfig

# 向上取整的整数除法
def ceil_div(a, b):
    return (a + b - 1) // b

class Request:
    def __init__(self, id, tokens, sample_config: SamplerConfig):
        self.id = id
        self.tokens = tokens
        self.is_prefill = True
        self.config = sample_config

    def add_toekn(self, token):
        self.is_prefill = False
        self.tokens.append(token)

    def is_finish(self):
        if self.get_token_num() >= self.config.max_new_token_new:
            return True
        return False

    def get_token_num(self):
        return len(self.tokens)

class DeviceTableManager:
    def __init__(self, name, block_num, block_size):
        self.name = name
        self.free_blocks = deque(range(block_num))
        self.block_size = block_size
        self.block_tables = defaultdict(list)
        self.token_num_tables = defaultdict(int)
        self.swap_out_record = []

    def add_tokens(self, table_id, token_num):
        total_token_num = self.token_num_tables[table_id] + token_num
        need_block_num = ceil_div(total_token_num, self.block_size)
        new_block_num = need_block_num - len(self.block_tables[table_id])
        if new_block_num > len(self.free_blocks):
            return False

        self.token_num_tables[table_id] += token_num
        new_blocks = [self.free_blocks.popleft() for _ in range(need_block_num)]
        self.block_tables[table_id].extend(new_blocks)
        return True

    def free_table(self, table_id):
        if table_id not in self.block_tables[table_id]:
            return False
        self.free_blocks.extend(self.block_tables[table_id])
        del self.block_tables[table_id]
        del self.token_num_tables[table_id]
        return True

    def sawp_out_table(self, table_id, other: "DeviceTableManager"):
        if table_id not in self.block_tables:
            return False
        if not other.add_tokens(table_id, self.token_num_tables[table_id]):
            return False
        new_blocks = other.block_tables[table_id]
        old_blocks = self.block_tables[table_id]
        self.free_table(table_id)
        self.swap_out_record.append([old_blocks, new_blocks])
        return True

    def get_and_reset_record(self):
        record = self.swap_out_record
        self.swap_out_record = []
        return record

class CacheManager:
    def __init__(self, cpu_block_num, gpu_block_num, block_size):
        self.cpu_cache = DeviceTableManager("cpu", cpu_block_num, block_size)
        self.gpu_cache = DeviceTableManager("gpu", gpu_block_num, block_size)

    def add_block_table(self, request: Request):
        return self.gpu_cache.add_tokens(request.id, len(request.tokens))

    def pop_block_table(self, request):
        if request.id in self.gpu_cache.block_tables:
            return self.gpu_cache.free_table(request.id)
        if request.id in self.cpu_cache.block_tables:
            return self.cpu_cache.free_table(request.id)
        return False

    def pend_block_block(self, request):
        return self.gpu_cache.sawp_out_table(request.id, self.cpu_cache)

    def restore_block_block(self, request):
        return self.cpu_cache.sawp_out_table(request.id, self.gpu_cache)

    def append_block(self, request):
        return self.gpu_cache.add_tokens(request.id, 1)

    def get_and_reset_record(self):
        gpu2cpu_record = self.gpu_cache.get_and_reset_record()
        cpu2gpu_record = self.cpu_cache.get_and_reset_record()
        return [gpu2cpu_record, cpu2gpu_record]

class Schedule:    
    def __init__(self, cpu_block_num, gpu_block_num, block_size, model_config):
        self.run_queue = deque()
        self.pend_queue = deque()
        self.wait_queue = deque()
        self.finish_queue = deque()

        self.cache_manager = CacheManager(
            cpu_block_num=cpu_block_num,
            gpu_block_num=gpu_block_num,
            block_size=block_size,
        )
        self.worker = Worker(model_config)
    
    def add_wait_request(self, request):
        self.wait_queue.append(request)

    def get_finish_request(self):
        finish_reqs = list(self.finish_queue)
        self.finish_queue.clear()
        return finish_reqs

    def step(self):
        has_free_gpu_block = True

        le_handle, ri_pop = 0, len(self.run_queue)
        while le_handle < ri_pop:
            while le_handle < ri_pop and not self.cache_manager.append_block(self.run_queue[le_handle]):
                if not self.cache_manager.pend_block_block(self.run_queue[ri_pop - 1]):
                    break
                ri_pop -= 1
                self.pend_queue.append(self.run_queue.pop())
                has_free_gpu_block = False

            le_handle += 1

        while has_free_gpu_block and self.pend_queue:
            req = self.pend_queue[0]
            if self.cache_manager.restore_block_block(req):
                self.run_queue.append(self.pend_queue.popleft())
            else:
                break
        
        while has_free_gpu_block and self.wait_queue:
            req = self.wait_queue[0]
            if self.cache_manager.add_block_table(req):
                self.run_queue.append(self.wait_queue.popleft())
            else:
                break
        
        block_update_record = self.cache_manager.get_and_reset_record()

        finish_requests, unfinish_requests = self.worker.step(self.run_queue, block_update_record)

        self.run_queue.clear()
        self.run_queue.extend(unfinish_requests)
        self.finish_queue.extend(finish_requests)

class PhysicalCacheContent:
    def __init__(self, block_num, block_size, model_config: OPTConfig):
        # TODO
        self.k_cache = torch.empty([
            block_num, 
            block_size, 
            model_config.num_hidden_layers, 
            model_config.num_attention_heads, 
            model_config.num_attention_heads / model_config.hidden_size
        ], dtype=get_torch_dtype(model_config.torch_dtype))
        self.v_cache = torch.empty([
            block_num, 
            block_size, 
            model_config.num_hidden_layers, 
            model_config.num_attention_heads, 
            model_config.num_attention_heads / model_config.hidden_size
        ], dtype=get_torch_dtype(model_config.torch_dtype))
    
    def update_cache(self, record):
        # TODO
        return

class Worker:
    def __init__(self, model_config):
        self.model = OPTForCausalLM(model_config)

    def update_cahce(self, record):
        # TODO
        return

    def step(self, request_list, block_update_record):
        finish_requests = []
        unfinish_requests = []

        for request in request_list:
            request.add_toekn(1)

            if request.is_finish():
                finish_requests.append(request)
            else:
                unfinish_requests.append(request)

        return finish_requests, unfinish_requests

if __name__ == '__main__':
    model_config = OPTConfig()
    sample_config = SamplerConfig()
    sample_config.max_new_token_new = 10
    scheduler = Schedule(1000, 1000, 16, model_config)
    req = Request(0, [1, 2, 3, 4, 5], sample_config)
    scheduler.add_wait_request(req)
    for _ in range(20):
        scheduler.step()
        output = scheduler.get_finish_request()
        logger.info(f"output: {output}")

