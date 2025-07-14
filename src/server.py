from collections import deque, defaultdict
from collections.abc import Iterable

from transformers import AutoTokenizer

from base import *
import torch
from loguru import logger


# 向上取整的整数除法
def ceil_div(a, b):
    return (a + b - 1) // b

class Request:
    def __init__(self, id, tokens, sample_config: SampleConfig):
        self.id = id
        self.prefill_tokens = tokens
        self.decode_tokens = []
        self.slot_mapping = []
        self.block_table = []
        self.is_prefill = True
        self.sample_config = sample_config

    def add_toekn(self, token):
        self.is_prefill = False
        self.decode_tokens.append(token)

    def is_finish(self):
        if len(self.decode_tokens) >= self.sample_config.max_new_token_new:
            return True
        return False

    def get_token_num(self):
        return len(self.prefill_tokens) + len(self.decode_tokens)

class DeviceTableManager:
    def __init__(self, name, block_num, block_size):
        self.name = name
        self.free_blocks = deque(range(block_num))
        self.block_size = block_size
        self.block_tables = defaultdict(list)
        self.token_num_tables = defaultdict(int)
        self.slot_mapping = defaultdict(list)
        self.swap_out_record = []

    def add_tokens(self, table_id, token_num):
        total_token_num = self.token_num_tables[table_id] + token_num
        need_block_num = ceil_div(total_token_num, self.block_size)
        new_block_num = need_block_num - len(self.block_tables[table_id])
        if new_block_num > len(self.free_blocks):
            return False

        self.token_num_tables[table_id] += token_num
        new_blocks = [self.free_blocks.popleft() for _ in range(new_block_num)]
        self.block_tables[table_id].extend(new_blocks)
        for block_id in new_blocks:
            start_slot_id = block_id * self.block_size
            self.slot_mapping[table_id].extend(list(range(start_slot_id, start_slot_id+self.block_size)))
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
        return self.gpu_cache.add_tokens(request.id, len(request.prefill_tokens))

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
    def __init__(self, config: Config):
        self.run_queue = deque()
        self.pend_queue = deque()
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
        
        gpu2cpu_record, cpu2gpu_record = self.cache_manager.get_and_reset_record()

        for req in self.run_queue:
            # TODO refactor Request
            req.block_table = self.cache_manager.gpu_cache.block_tables[req.id]
            req_slot_mapping = self.cache_manager.gpu_cache.slot_mapping[req.id]
            req.slot_mapping = req_slot_mapping[:req.get_token_num()] \
                if req.is_prefill else [req_slot_mapping[req.get_token_num() - 1]]

        if not self.run_queue: return
        finish_requests, unfinish_requests = self.worker.step(self.run_queue, gpu2cpu_record, cpu2gpu_record)

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

    def step(self, request_list: Iterable[Request], gpu2cpu_record, cpu2gpu_record):
        self.device_cache_manager.update_cache(gpu2cpu_record, cpu2gpu_record)

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
                prefill_input_ids.extend(req.prefill_tokens)
                prefill_slot_mapping.extend(req.slot_mapping)
                token_num = req.get_token_num()
                prefill_position_ids.extend(list(range(token_num)))
                prefill_cu_seqlens_q.append(prefill_cu_seqlens_q[-1] + token_num)
                prefill_max_seq_len_q = max(prefill_max_seq_len_q, token_num)
            else:
                decode_reqs.append(req)
                decode_input_ids.append(req.decode_tokens[-1])
                decode_slot_mapping.append(req.slot_mapping[-1])
                decode_position_ids.append(req.get_token_num() - 1)
                decode_seq_len.append(req.get_token_num())
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
                    [req.decode_tokens for req in finish_requests], [req.decode_tokens for req in unfinish_requests])

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
        logger.info(f"output: {[req.decode_tokens for req in output]}")

    logger.info("result={}", [tokenizer.decode(req.decode_tokens) for req in output])


