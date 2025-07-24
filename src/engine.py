from collections import deque

from base import *
from base import SampleConfig
import torch
from loguru import logger
from math import floor
import multiprocessing as mp
import threading as th

from src.base import ScheduleInfo
from src.squence import Sequence
from src.worker import Worker


class BlockManager:
    def __init__(self, block_num, block_size, enable_prefix_cache):
        self.block_size = block_size
        self.free_blocks = None
        self.block_refs = None
        self.block_to_hashs = dict()
        self.hash_to_blocks = dict()
        self.enable_prefix_cache = enable_prefix_cache

    def set(self, block_num):
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
        self.block_manager.set(block_num)

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

    def alloc_prefix_cached_blocks(self, req: Sequence):
        if req.computed_len != 0:
            return

        # NOTE: keep last block to compute next token in prefix cache
        for a in range((req.in_seq_len - 1) // req.block_size):
            block_id = self.block_manager.get_cached_block(self._get_req_block_hash(req, a))
            if block_id is not None:
                req.append_block(block_id, True, True)
            else:
                break

    def alloc_new_blocks(self, req: Sequence, max_new_token: int):
        if req.allocated_len >= req.seq_len:
            return True

        need_block_num = ceil_div(req.computed_len + max_new_token - req.allocated_len, self.block_size)
        for _ in range(need_block_num):
            block_id = self.block_manager.alloc_new_block()
            req.append_block(block_id, False, False)

    def free_cache(self, req: Sequence):
        for block_id in req.block_table:
            self.block_manager.free_block(block_id)


# TODO from Engine detach Scheduler
class Engine:
    def __init__(self, id, config: Config):
        self.id = id
        self.config = config
        self.run_queue = deque()
        self.wait_queue = deque()

        self.enable_chunked_prefill = config.infer_config.enable_chunked_prefill
        self.max_prefill_len = config.infer_config.max_prefill_len if self.enable_chunked_prefill else float("inf")
        self.max_req_num = config.infer_config.max_req_num
        self.max_batch_token_num = config.infer_config.max_batch_token_num
        self.gpu_memory_utilization = config.infer_config.gpu_memory_utilization

        def worker_fun(worker_id, in_queue, out_queue):
            worker = Worker(worker_id, config)
            worker.run(in_queue, out_queue)
        self.workers_procs = []
        self.worker_in_queues = [mp.Queue() for _ in range(self.config.parallel_config.tp_size)]
        self.worker_out_queue = mp.Queue()
        for a in range(self.config.parallel_config.tp_size):
            self.workers_procs.append(mp.Process(target=worker_fun,
                                                 args=(a, self.worker_in_queues[a], self.worker_out_queue)))

        msg_type, block_num = self.worker_out_queue.get()
        assert msg_type == MessageType.worker_start
        self.cache_manager = CacheManager(
            block_num=ceil_div(self.max_batch_token_num, config.infer_config.block_size),
            block_size=config.infer_config.block_size,
            enable_prefix_cache=config.infer_config.enable_prefix_cache
        )
        self.cache_manager.set_cache(block_num)

    def add_wait_request(self, request):
        self.wait_queue.append(request)
        logger.info(f"add_wait_request req_id={request.id} done")

    def send_msg_to_all_workers(self, msg_type, msg_body):
        for a in range(self.config.parallel_config.tp_size):
            self.worker_in_queues[a].put((msg_type, msg_body))

    def run(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.cond = th.Condition()

        def handle_request(context, req):
            # NOTE: wait queue is thread safe
            logger.info("req left in_queue")
            self.add_wait_request(req)
            with context.cond:
                context.cond.notify_all()
        handlers = {
            MessageType.request: [
                (self, handle_request)
            ]
        }
        self.listener = Listener(self.in_queue, handlers)

        self.out_queue.put((MessageType.engine_start, self.id))
        logger.info("init scheduler {} ok", self.id)

        while True:
            with self.cond:
                while not self.wait_queue:
                    self.cond.wait()
            self.step()

    def step(self):
        if not self.wait_queue:
            return

        self.run_queue.clear()
        info = ScheduleInfo()

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
        # TODO need opt
        self.send_msg_to_all_workers(MessageType.worker_step, (list(self.run_queue), info))
        msg_type, (finish_requests, unfinished_requests) = self.worker_out_queue.get()
        assert msg_type == MessageType.worker_step_ret
        # TODO should refactor
        for req in self.run_queue:
            req.step_fun(self)

        # NOTE: first use unfinished req
        self.wait_queue.extendleft(unfinished_requests)

        for req in finish_requests:
            self.cache_manager.free_cache(req)
        return finish_requests




