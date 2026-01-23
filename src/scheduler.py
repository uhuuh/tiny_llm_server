import queue
from collections import deque
from loguru import logger
import multiprocessing as mp
import threading as th

from src.base import *


class CacheManager:
    def __init__(self, block_num):
        self.free_blocks = deque(range(block_num))

    def can_alloc(self, need_block_num):
        return len(self.free_blocks) >= need_block_num

    def alloc(self, need_block_num):
        assert self.can_alloc(need_block_num)
        ret = []
        for _ in range(need_block_num):
            ret.append(self.free_blocks.popleft())
        return ret

    def free(self, block_table: List[int]):
        self.free_blocks.extend(block_table)

class Scheduler:
    def __init__(self, id, config: Config, in_queue: mp.Queue, out_queue: mp.Queue, worker_in_queues: List[mp.Queue], worker_out_queue: mp.Queue):
        self.id = id
        self.config = config
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.worker_in_queues = worker_in_queues
        self.worker_out_queue = worker_out_queue

        self._req_recv_cond = th.Condition()
        self._wait_add_reqs = queue.Queue()
        self.run_queue = deque()
        self.wait_queue = deque()

        self.enable_chunked_prefill = config.infer_config.enable_chunked_prefill
        self.max_prefill_len = config.infer_config.max_prefill_len if self.enable_chunked_prefill else float("inf")
        self.max_req_num = config.infer_config.max_req_num
        self.max_batch_token_num = config.infer_config.max_batch_token_num
        self.gpu_memory_utilization = config.infer_config.gpu_memory_utilization

        tp = self.config.infer_config.tensor_parallel
        min_block_num = float("inf")
        for _ in range(tp):
            msg: WorkerInit  = self.worker_out_queue.get()
            logger.info("recv worker {} init finished message", msg.worker_id)
            min_block_num = min(min_block_num, msg.block_num)
        logger.info("all worker init finished")

        self.cache_manager = CacheManager(block_num=min_block_num)

        logger.info("scheduler {} init finished", self.id)
        self.out_queue.put(SchedulerInit(self.id))

    def recv_req_loop(self):
        while True:
            msg: SchedulerInput = self.in_queue.get()
            logger.info("add_req {}", [r.request_id for r in msg.requests])
            for r in msg.requests:
                self._wait_add_reqs.put(Sequence(
                    id=r.request_id,
                    sample_config=r.sample_config,
                    prompt_tokens=r.prompt_tokens,
                ))

            with self._req_recv_cond:
                self._req_recv_cond.notify_all()

    def step_loop(self):
        while True:
            if not self.wait_queue and self._wait_add_reqs.empty():
                with self._req_recv_cond:
                    while not self._wait_add_reqs:
                        self._req_recv_cond.wait()

            try:
                while True:
                    self.wait_queue.append(self._wait_add_reqs.get_nowait())
            except queue.Empty:
                pass

            msg = self.step()
            if msg:
                self.out_queue.put_nowait(msg)

    def _schedule(self):
        # 按照req到达顺序schedule
        now_req_num = 0
        now_acc_token_num = 0
        now_run_queue = deque()

        # computed_len 应该在forward之前更新, 因为如果释放一个req时需要将这个长度设为0
        def add_req(r: Sequence, new_token_num: int):
            nonlocal now_req_num, now_acc_token_num
            r.block_table.extend(self.cache_manager.alloc(new_token_num))
            r.scheduled_len += new_token_num
            r.compute_len = new_token_num

            now_req_num += 1
            now_acc_token_num += new_token_num
            now_run_queue.append(r)

        def free_req(r: Sequence):
            self.cache_manager.free(r.block_table)
            r.block_table = []
            r.scheduled_len = 0

        i = 0
        while i < len(self.run_queue):
            seq: Sequence = self.run_queue[i]
            if now_req_num + 1 > self.max_req_num:
                break
            if now_acc_token_num >= self.max_batch_token_num:
                break

            can_schedule = False
            schedule_len = min(seq.max_seq_len - seq.scheduled_len, self.max_batch_token_num - now_acc_token_num)

            j = len(self.run_queue) - 1
            while i < j:
                seq2: Sequence = self.run_queue.pop()
                free_req(seq2)
                self.wait_queue.appendleft(seq2)
                j -= 1

                if self.cache_manager.can_alloc(schedule_len):
                    can_schedule = True
                    break

            if can_schedule:
                add_req(seq, now_acc_token_num)
            else:
                free_req(seq)
                self.wait_queue.appendleft(seq)
                break

        while self.wait_queue:
            seq: Sequence = self.wait_queue[0]
            if now_req_num + 1 > self.max_req_num:
                break
            if now_acc_token_num >= self.max_batch_token_num:
                break

            schedule_len = min(seq.max_seq_len - seq.scheduled_len, self.max_batch_token_num - now_acc_token_num)
            if self.cache_manager.can_alloc(schedule_len):
                add_req(seq, now_acc_token_num)
                self.wait_queue.popleft()
            else:
                break

        self.run_queue = now_run_queue

    def _update(self, inp: WorkerInput, out: WorkerOutput):
        assert len(inp.seqs) == len(out.seqs)
        finished_seqs = []
        self.run_queue.clear()

        for i in range(len(inp.seqs)):
            seq: Sequence = inp.seqs[i]
            seq.output_tokens.extend(out.seqs[i].output_tokens)

            # TODO eos token finish
            is_finished = seq.sep_len >= seq.max_seq_len
            if is_finished:
                self.cache_manager.free(seq.block_table)
                finished_seqs.append(seq)
            else:
                self.run_queue.append(seq)
        return finished_seqs

    def step(self):
        if not self.run_queue and not self.wait_queue:
            return

        self._schedule()

        logger.info("schedule_req {}", [req.id for req in self.run_queue])
        msg_inp = WorkerInput(seqs=list(self.run_queue))
        for q in self.worker_in_queues:
            q.put_nowait(msg_inp)

        msg_out: WorkerOutput = self.worker_out_queue.get()

        finished_seqs = self._update(msg_inp, msg_out)

        if not finished_seqs:
            return

        msg_ret = SchedulerOutput(
            scheduler_id=self.id,
            requests=[],
        )
        for req in finished_seqs:
            msg_ret.requests.append(SchedulerOutput.RequestOutputInfo(
                request_id=req.id,
                prompt_tokens=req.prompt_tokens,
                output_tokens=req.output_tokens,
            ))
        logger.info("worker_finished_req {}", [req.id for req in finished_seqs])
        return msg_ret

