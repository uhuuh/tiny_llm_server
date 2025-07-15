from collections import deque, defaultdict
from collections.abc import Iterable

from sklearn.metrics import nan_euclidean_distances
from transformers import AutoTokenizer

from base import *
import torch
from loguru import logger


# 向上取整的整数除法
def ceil_div(a, b):
    return (a + b - 1) // b

class Request:
    def __init__(self, id, tokens, config: Config, sample_config: SampleConfig):
        self.id = id
        self.tokens = tokens
        self.slot_mapping = []
        self.block_table = []
        self.in_seq_len = len(tokens)
        self.out_seq_len = 0
        self.seq_len = self.in_seq_len + self.out_seq_len
        self.sample_config = sample_config
        self.block_size = config.infer_config.block_size

    def append_token(self, token):
        self.tokens.append(token)
        self.out_seq_len += 1
        self.seq_len += 1

    def is_finish(self):
        if len(self.tokens) >= self.sample_config.max_new_token_new:
            return True
        return False

class DeviceTableManager:
    def __init__(self, name, block_num, block_size):
        self.name = name
        self.block_size = block_size
        self.free_blocks = deque(range(block_num))
        self.block_refs = [0 for _ in range(block_num)]
        self.block_to_hashs = dict()
        self.hash_to_blocks = dict()

    def block_hash_fun(self, block):
        assert len(block) == self.block_size
        return hash(tuple(block))

    def get_cached_block(self, block_content):
        assert len(block_content) == self.block_size
        block_hash = self.block_hash_fun(block_content)
        return self.hash_to_blocks.get(block_hash, None)

    def get_block(self, block_content):
        assert len(self.free_blocks) >= 0
        block_id = self.free_blocks.popleft()
        self.block_refs[block_id] += 1

        block_hash = self.block_to_hashs.get(block_id, None)
        if block_hash is not None:
            self.hash_to_blocks.pop(block_hash)
            self.block_to_hashs.pop(block_id)

        new_block_hash = self.block_hash_fun(block_content)
        self.hash_to_blocks[new_block_hash] = block_id
        self.block_to_hashs[block_id] = new_block_hash
        return block_id

    def free_block(self, block_id):
        self.block_refs[block_id] -= 1
        if self.block_refs[block_id] == 0:
            self.free_blocks.append(block_id)

class CacheManager:
    def __init__(self, cpu_block_num, gpu_block_num, block_size):
        self.block_size = block_size
        self.gpu_cache = DeviceTableManager("gpu", gpu_block_num, block_size)

    def get_block_slots(self, blocks, token_num):
        slots = []
        for a in range(0, token_num, self.block_size):
            start_slot = blocks[a] * self.block_size
            slots.extend(list(range(start_slot, start_slot + self.block_size)))
        return slots

    def alloc_cache(self, req: Request):
        alloc_ok, allocated_prefix_block_num, allocated_no_prefix_block_num = True, 0, 0

        can_alloc_cache = (req.seq_len - len(req.block_table) * self.block_size) > 0
        if not can_alloc_cache:
            return alloc_ok, allocated_prefix_block_num, allocated_no_prefix_block_num

        if not req.block_table:
            prefix_cached_blocks = []
            for a in range(0, req.seq_len - 1, self.block_size):
                block_content = req.tokens[a:a + self.block_size]
                block_id = self.gpu_cache.get_block(block_content)
                if block_id is not None:
                    prefix_cached_blocks.append(block_id)
                else:
                    break

            req.block_table.extend(prefix_cached_blocks)
            allocated_prefix_block_num = len(prefix_cached_blocks)

        alloc_seq_len = len(req.block_table) * self.block_size
        for a in range(alloc_seq_len, req.seq_len + self.block_size - 1, self.block_size):
            block_content = req.tokens[a:a + self.block_size]
            block_id = self.gpu_cache.get_block(block_content)
            if block_id is None:
                return False, -1, -1
            req.block_table.append(block_id)
            allocated_no_prefix_block_num += 1

        return alloc_ok, allocated_prefix_block_num, allocated_no_prefix_block_num

    def free_cache(self, req: Request):
        for block_id in req.block_table:
            self.gpu_cache.free_block(block_id)

class Schedule:
    def __init__(self, config: Config):
        self.run_queue = deque()
        self.wait_queue = deque()
        self.finish_queue = deque()

        self.cache_manager = CacheManager(
            cpu_block_num=config.infer_config.cpu_block_num,
            gpu_block_num=config.infer_config.gpu_block_num,
            block_size=config.infer_config.block_size,
        )
        self.worker = Worker(config)

    def add_wait_request(self, request):
        self.wait_queue.append(request)

    def get_finish_request(self):
        finish_reqs = list(self.finish_queue)
        self.finish_queue.clear()
        return finish_reqs

    def step(self):
        cache_ok = True
        for req in self.run_queue:
            pass

        if cache_ok:
            for req in self.wait_queue:
                pass

        if not self.run_queue: return
        finish_requests, unfinish_requests = self.worker.step(self.run_queue)

        self.run_queue.clear()
        self.run_queue.extend(unfinish_requests)
        self.finish_queue.extend(finish_requests)

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

    #config.model_config.num_hidden_layers = 2
    model = Qwen2(config.model_config)
    model = model.to("cuda").eval()
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

    def step(self, request_list: Iterable[Request]):
        prefill_reqs = []
        prefill_input_ids = []
        prefill_slot_mapping = []
        prefill_position_ids = []
        prefill_cu_seqlens_q = [0]
        prefill_max_seq_len_q = -1
        decode_reqs = []
        decode_input_ids = []
        decode_slot_mapping = []
        decode_position_ids = []
        decode_seq_len = []
        decode_block_table = []
        for req in request_list:
            if req.is_prefill:
                prefill_reqs.append(req)
                prefill_input_ids.extend(req.in_tokens)
                prefill_slot_mapping.extend(req.slot_mapping)
                token_num = req.get_seq_len()
                prefill_position_ids.extend(list(range(token_num)))
                prefill_cu_seqlens_q.append(prefill_cu_seqlens_q[-1] + token_num)
                prefill_max_seq_len_q = max(prefill_max_seq_len_q, token_num)
            else:
                decode_reqs.append(req)
                decode_input_ids.append(req.tokens[-1])
                decode_slot_mapping.append(req.slot_mapping[-1])
                decode_position_ids.append(req.get_seq_len() - 1)
                decode_seq_len.append(req.get_seq_len())
                decode_block_table.append(req.block_table)

        input_ids = torch.tensor(prefill_input_ids + decode_input_ids, dtype=torch.int64)
        position_ids = torch.tensor(prefill_position_ids + decode_position_ids, dtype=torch.int64)
        slot_mapping = torch.tensor(prefill_slot_mapping + decode_slot_mapping, dtype=torch.int64)
        cos, sin = self.pos_emb_manager.get_cos_sin(position_ids)
        prefill_cu_seqlens_q = torch.tensor(prefill_cu_seqlens_q, dtype=torch.int32)
        prefill_max_seq_len_q = torch.tensor(prefill_max_seq_len_q, dtype=torch.int32)
        max_block_num_per_seq = max(map(len, decode_block_table)) if decode_block_table else 0
        decode_block_table = torch.tensor([t + [-1] * (max_block_num_per_seq - len(t)) for t in decode_block_table]
                                   , dtype=torch.int32)
        decode_seq_lens = torch.tensor(decode_seq_len, dtype=torch.int32)
        inp = ModelInput(
            input_ids=input_ids,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            num_prefill_tokens=len(prefill_input_ids),
            k_cache=self.device_cache_manager.gpu_cache.k_cache,
            v_cache=self.device_cache_manager.gpu_cache.v_cache,
            slot_mapping=slot_mapping,
            prefill_cu_seqlens_q=prefill_cu_seqlens_q,
            prefill_max_seqlen_q=prefill_max_seq_len_q,
            decode_block_table=decode_block_table,
            decode_seq_lens=decode_seq_lens,
        )

        output = self.model(inp)

        finish_requests = []
        unfinish_requests = []

        prefill_logits = output[:len(prefill_input_ids)][prefill_cu_seqlens_q[1: ].long() - 1]
        prefill_next_tokens = torch.argmax(prefill_logits, dim=-1).tolist()
        for req, prefill_next_token in zip(prefill_reqs, prefill_next_tokens):
            req.add_toekn(prefill_next_token)
            if req.is_finish():
                finish_requests.append(req)
            else:
                unfinish_requests.append(req)

        decode_logits = output[len(prefill_input_ids): ]
        decode_next_tokens = torch.argmax(decode_logits, dim=-1).tolist()
        for req, next_token in zip(decode_reqs, decode_next_tokens):
            req.add_toekn(next_token)
            if req.is_finish():
                finish_requests.append(req)
            else:
                unfinish_requests.append(req)
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
        block_size=16,
        cpu_block_num=100,
        gpu_block_num=100,
        model_path="/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct",
    )
    config = Config(
        infer_config=infer_config,
        sample_config=sample_config,
    )
    scheduler = Schedule(config)

    tokenizer = AutoTokenizer.from_pretrained(config.infer_config.model_path)

    req = Request(0,  [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465,
       553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847,
       13, 151645, 198, 151644, 872, 198, 14990, 1879, 151645,
       198, 151644, 77091, 198], sample_config)
    # [9707,    0, 1084,  594, 6419]
    scheduler.add_wait_request(req)

    req = Request(1, [151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,     40,   1079, 151645,
            198, 151644,  77091,    198], sample_config)
    # [9707,    0, 2585,  646,  358]
    scheduler.add_wait_request(req)

    output = []
    for _ in range(config.sample_config.max_new_token_new):
        scheduler.step()
        output = scheduler.get_finish_request()
        logger.info(f"output: {[req.tokens for req in output]}")

    logger.info("result={}", [tokenizer.decode(req.tokens) for req in output])


