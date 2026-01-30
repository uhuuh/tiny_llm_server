import queue
from collections import deque
from loguru import logger
import multiprocessing as mp
import threading as th

from src.base import *


class BlockManager:
    def __init__(self, block_num, block_size):
        self.block_size = block_size
        self.free_blocks = deque(range(block_num))

    def _need_block_num(self, seq: Sequence, new_token_num: int):
        ret = (seq.computed_len + new_token_num + self.block_size - 1) // self.block_size - len(seq.block_table)
        return ret

    def can_alloc(self, seq: Sequence, new_token_num):
        return len(self.free_blocks) >= self._need_block_num(seq, new_token_num)

    def alloc(self, seq: Sequence, new_token_num):
        # TODO block table 真的应该被seq自身所持有吗?
        assert self.can_alloc(seq, new_token_num)
        need_block_num = self._need_block_num(seq, new_token_num)
        seq.block_table.extend([self.free_blocks.popleft() for _ in range(need_block_num)])

    def free(self, seq: Sequence):
        self.free_blocks.extend(seq.block_table)
        seq.block_table.clear()

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
            msg: WorkerInit = self.worker_out_queue.get()
            logger.info("scheduler recv worker {} init", msg.worker_id)
            min_block_num = min(min_block_num, msg.block_num)
        logger.info("scheduler all worker init")

        self.cache_manager = BlockManager(block_num=min_block_num, block_size=self.config.infer_config.block_size)

        logger.info("scheduler {} init", self.id)
        self.out_queue.put(SchedulerInit(self.id))

    def recv_req_loop(self):
        while True:
            msg: SchedulerInput = self.in_queue.get()
            logger.info("schedule input {}", msg)
            for r in msg.requests:
                self._wait_add_reqs.put(Sequence(
                    id=r.request_id,
                    sample_config=r.sample_config,
                    tokens=r.prompt_tokens,
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

    @property
    def total_req_num(self):
        return len(self.run_queue) + len(self.wait_queue)

    def _schedule(self):
        # 按照req到达顺序schedule
        now_req_num = 0
        now_acc_token_num = 0
        now_run_queue = deque()
        logger.info("scheduler before run={} wait={}", self.run_queue, self.wait_queue)

        # computed_len 应该在forward之前更新, 因为如果释放一个req时需要将这个长度设为0
        def _add_req(seq: Sequence, new_token_num: int):
            nonlocal now_req_num, now_acc_token_num
            self.cache_manager.alloc(seq, new_token_num)
            seq.new_len = new_token_num

            now_req_num += 1
            now_acc_token_num += new_token_num
            now_run_queue.append(seq)

        def _free_req(seq: Sequence):
            self.cache_manager.free(seq)
            seq.computed_len = 0
            logger.warning("scheduler recompute {}", seq)

        i = 0
        while i < len(self.run_queue):
            seq: Sequence = self.run_queue[i]
            if now_req_num + 1 > self.max_req_num:
                break
            if now_acc_token_num >= self.max_batch_token_num:
                break

            schedule_len = min(seq.seq_len - seq.computed_len, self.max_batch_token_num - now_acc_token_num)
            can_schedule =  self.cache_manager.can_alloc(seq, schedule_len)

            j = len(self.run_queue) - 1
            while not can_schedule and i < j:
                seq2: Sequence = self.run_queue.pop()
                _free_req(seq2)
                self.wait_queue.appendleft(seq2)
                j -= 1

                can_schedule = self.cache_manager.can_alloc(seq, schedule_len)

            if can_schedule:
                _add_req(seq, schedule_len)
                i += 1
            else:
                _free_req(seq)
                self.wait_queue.appendleft(seq)
                break

        while self.wait_queue:
            seq: Sequence = self.wait_queue[0]
            if now_req_num + 1 > self.max_req_num:
                break
            if now_acc_token_num >= self.max_batch_token_num:
                break

            schedule_len = min(seq.seq_len - seq.computed_len, self.max_batch_token_num - now_acc_token_num)
            if self.cache_manager.can_alloc(seq, schedule_len):
                _add_req(seq, schedule_len)
                self.wait_queue.popleft()
            else:
                break

        self.run_queue = now_run_queue
        logger.info("scheduler after run={} wait={}", self.run_queue, self.wait_queue)

    def _update(self, inp: WorkerInput, out: WorkerOutput):
        assert len(inp.seqs) == len(out.seqs)
        finished_seqs = []
        self.run_queue.clear()

        for i in range(len(inp.seqs)):
            seq: Sequence = inp.seqs[i]
            out_tokens = out.seqs[i].output_tokens
            seq.computed_len += seq.new_len
            seq.new_len = 0
            # TODO 有些情况下不应该sample token, 这个应该在worker那里做
            if seq.computed_len >= seq.seq_len:
                seq.tokens.extend(out_tokens)

            # TODO eos token finish
            is_finished = seq.seq_len >= seq.max_seq_len
            if is_finished:
                self.cache_manager.free(seq)
                finished_seqs.append(seq)
            else:
                self.run_queue.append(seq)
        return finished_seqs

    def step(self):
        if not self.run_queue and not self.wait_queue:
            return

        self._schedule()

        if not self.run_queue:
            return

        msg_inp = WorkerInput(seqs=list(self.run_queue))
        for q in self.worker_in_queues:
            q.put_nowait(msg_inp)

        msg_out: WorkerOutput = self.worker_out_queue.get()

        finished_seqs = self._update(msg_inp, msg_out)
        logger.info("scheduler update finish={} run={}", finished_seqs, self.run_queue)

        if not finished_seqs:
            return

        msg_ret = SchedulerOutput(
            scheduler_id=self.id,
            requests=[],
        )
        for req in finished_seqs:
            msg_ret.requests.append(SchedulerOutput.RequestOutputInfo(
                request_id=req.id,
                prompt_tokens=req.tokens[:req.prompt_len],
                output_tokens=req.tokens[req.prompt_len:],
            ))
        logger.info("scheduler output {}", msg_ret)
        return msg_ret

