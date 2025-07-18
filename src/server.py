from collections import deque
from typing import Dict

from transformers import AutoTokenizer

from base import *
import torch
from loguru import logger

class Request:
    def __init__(self, id, tokens, config: Config, sample_config: SampleConfig):
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
        self.block_size = config.infer_config.block_size

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

class DeviceTableManager:
    def __init__(self, name, block_num, block_size, enable_prefix_cache):
        self.name = name
        self.block_size = block_size
        self.free_blocks = deque(range(block_num))
        self.block_refs = [0 for _ in range(block_num)]
        self.block_to_hashs = dict()
        self.hash_to_blocks = dict()
        self.enable_prefix_cache = enable_prefix_cache

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
    def __init__(self, cpu_block_num, gpu_block_num, block_size, enable_prefix_cache):
        self.block_size = block_size
        self.gpu_cache = DeviceTableManager("gpu", gpu_block_num, block_size, enable_prefix_cache)

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
        return self.gpu_cache.can_alloc(need_new_block_num)

    def cache_computed_block(self, req):
        if req.computed_len - req.cached_len >= req.block_size:
            block_idx = req.cached_len // self.block_size
            self.gpu_cache.cache_block(req.block_table[block_idx], self._get_req_block_hash(req, block_idx))
            req.append_block(None, False, True)

    def alloc_prefix_cached_blocks(self, req: Request):
        if req.computed_len != 0:
            return

        # NOTE: keep last block to compute next token in prefix cache
        for a in range((req.in_seq_len - 1) // req.block_size):
            block_id = self.gpu_cache.get_cached_block(self._get_req_block_hash(req, a))
            if block_id is not None:
                req.append_block(block_id, True, True)
            else:
                break

    def alloc_new_blocks(self, req: Request, max_new_token: int):
        if req.allocated_len >= req.seq_len:
            return True

        need_block_num = ceil_div(req.computed_len + max_new_token - req.allocated_len, self.block_size)
        for _ in range(need_block_num):
            block_id = self.gpu_cache.alloc_new_block()
            req.append_block(block_id, False, False)

    def free_cache(self, req: Request):
        for block_id in req.block_table:
            self.gpu_cache.free_block(block_id)

@dataclass
class ScheduleInfo:
    now_batch_token_num: int = 0
    now_req_num: int = 0
    req_new_token_nums: Dict[int, int] = field(default_factory=dict)

class Schedule:
    def __init__(self, config: Config):
        self.run_queue = deque()
        self.wait_queue = deque()
        self.finish_queue = deque()
        self.enable_chunked_prefill = config.infer_config.enable_chunked_prefill
        self.max_prefill_len = config.infer_config.max_prefill_len
        self.max_req_num = config.infer_config.max_req_num
        self.max_batch_token_num = config.infer_config.max_batch_token_num

        self.cache_manager = CacheManager(
            cpu_block_num=config.infer_config.cpu_block_num,
            gpu_block_num=config.infer_config.gpu_block_num,
            block_size=config.infer_config.block_size,
            enable_prefix_cache=config.infer_config.enable_prefix_cache
        )
        self.worker = Worker(config)

    def add_wait_request(self, request):
        self.wait_queue.append(request)

    def get_finish_request(self):
        finish_reqs = list(self.finish_queue)
        self.finish_queue.clear()
        return finish_reqs

    def step(self):
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
            if self.enable_chunked_prefill:
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

        # NOTE: first use unfinished req
        self.wait_queue.extendleft(unfinished_requests)

        self.finish_queue.extend(finish_requests)
        for req in finish_requests:
            self.cache_manager.free_cache(req)

from vllm import _custom_ops as ops
class DeviceCache:
    def __init__(self, name: DeviceType, config: Config):
        self.name = name
        param_dtype = config.model_config.get_param_dtype()
        layer_num = config.model_config.num_hidden_layers
        head_dim = config.model_config.hidden_size // config.model_config.num_attention_heads
        head_kv_num = config.model_config.num_key_value_heads
        block_size = config.infer_config.block_size
        kv_cache_shape = [config.infer_config.gpu_block_num
                          if self.name == DeviceType.GPU
                          else config.infer_config.cpu_block_num,
                          block_size, head_kv_num, head_dim]
        self.k_cache = [torch.empty(kv_cache_shape, dtype=param_dtype, device="cuda")
                        for _ in range(layer_num)]
        self.v_cache = [torch.empty(kv_cache_shape, dtype=param_dtype, device="cuda")
                        for _ in range(layer_num)]

    def move_out(self, other: "DeviceCache", move_out_record):
        # TODO why vllm has sawp block?
        move_out_record = torch.tensor(move_out_record)
        ops.copy_blocks(self.k_cache, other.k_cache, move_out_record)
        ops.copy_blocks(self.v_cache, other.v_cache, move_out_record)

class DeviceCacheManager:
    def __init__(self, config: Config):
        self.gpu_cache = DeviceCache(name=DeviceType.GPU, config=config)
        self.cpu_cache = DeviceCache(name=DeviceType.CPU, config=config)

    def update_cache(self, gpu2cpu_record, cpu2gpu_record):
        '''
        self.gpu_cache.move_out(self.cpu_cache, gpu2cpu_record)
        self.cpu_cache.move_out(self.gpu_cache, cpu2gpu_record)
        '''
        # TODO update_cache
        return

def get_model(config: Config):
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
        self.device_cache_manager = DeviceCacheManager(config)
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
            k_cache=self.device_cache_manager.gpu_cache.k_cache,
            v_cache=self.device_cache_manager.gpu_cache.v_cache,
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    sample_config = SampleConfig(
        max_new_token_new=50
    )
    infer_config = InferConfig(
        model_path="/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct",
        max_req_num=256,
        max_batch_token_num=4096,
        block_size=16,
        cpu_block_num=100,
        gpu_block_num=100,
        enable_prefix_cache=True,
        enable_debug=False,
        max_prefill_len=32,
        enable_chunked_prefill=True,
    )
    config = Config(
        infer_config=infer_config,
        sample_config=sample_config, # TODO config not should has sample_config
    )
    scheduler = Schedule(config)

    tokenizer = AutoTokenizer.from_pretrained(config.infer_config.model_path)

    #scheduler.add_wait_request(Request(0,  [14990, 11, 1879], config, sample_config))
    #scheduler.add_wait_request(Request(1, [2408, 829, 374], config, sample_config))

    #scheduler.add_wait_request(Request(0, list(range(config.infer_config.block_size * 2)), config, sample_config))
    #scheduler.add_wait_request(Request(1, list(range(config.infer_config.block_size * 2)), config, sample_config))
    scheduler.add_wait_request(Request(0, [19810, 279, 13458, 88, 9104, 323, 279, 6783, 11980, 4889, 11, 1340, 8570, 5290, 1059, 6930, 22531, 11514, 553, 279, 39411, 11, 274, 5654, 4017, 17931, 13970, 11, 22930, 2770, 8974, 518, 279, 11174, 33019, 7741, 3241, 11, 5558, 304, 3381, 11, 6587, 41001, 304, 279, 1879, 315, 71830, 323, 27799, 13], config, sample_config))
    scheduler.add_wait_request(Request(1, [19810, 279, 13458, 88, 9104, 323, 279, 6783, 11980, 4889, 11, 1340, 8570, 5290, 1059, 6930, 22531, 11514, 553, 279, 39411, 11, 274, 5654, 4017, 17931, 13970, 11, 22930, 2770, 8974, 518, 279, 11174, 33019, 7741, 3241, 11, 5558, 304, 3381, 11, 6587, 41001, 304, 279, 1879, 315, 71830, 323, 27799, 13], config, sample_config))

    output = []
    # NOTE: because chunked prefill, has some step not generate new token
    for _ in range(config.sample_config.max_new_token_new + 5):
        scheduler.step()
        output = scheduler.get_finish_request()
        for req in output:
            logger.info("result={}", tokenizer.decode(req.tokens))

