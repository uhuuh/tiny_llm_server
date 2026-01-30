import glob
import os
from collections import deque
from math import floor
import multiprocessing as mp
from typing import List

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file
from transformers.models.esm.openfold_utils.protein import ModelOutput

from src.base import Config, SampleConfig, ceil_div, WorkerInit, WorkerOutput, \
    WorkerInput, Sequence
from itertools import accumulate
from src.attention import AttentionMetadata
from src.utils import set_forward_context, ForwardContext

class Sampler(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, seqs: List[Sequence], logits: torch.Tensor):
        assert logits.dim() == 2
        assert logits.shape[0] == len(seqs)
        assert logits.shape[1] == self.config.model_config.vocab_size

        rets = logits.argmax(-1).tolist()
        return rets

class Worker:
    def __init__(self, id, config: Config, in_queue: mp.Queue, out_queue: mp.Queue):
        self.id = id
        self.config = config
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.get_model()

        # worker 确实应该持有cache_storager，一个是worker是每张卡都有，而engine是每dp域有,
        # 二是所有分配cuda显存应该在一个进程中
        self.cache_storager = CacheStorager(config)

        self.sampler = Sampler(config)

        # TODO self.warm_up()
        logger.info("worker {} init finished", self.id)
        self.out_queue.put_nowait(WorkerInit(worker_id=self.id, block_num=self.cache_storager.block_num))

    def get_model(self):
        # TODO below should clear
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # torch.set_default_device('cuda')
        # torch.set_default_dtype(torch.bfloat16)

        from src.model import Qwen2

        if self.config.infer_config.enable_debug:
            logger.info("enable debug mode")
            self.config.model_config.num_hidden_layers = 2
        model = Qwen2(self.config)
        model = model.to(device="cuda", dtype=self.config.model_config.get_param_dtype()).eval()

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

    def _preprocess(self, inp: WorkerInput):
        # TODO use buffer
        input_ids = []
        position_ids = []
        slot_mapping = []
        block_tables = []
        max_block_num = 0
        seqlen_q = []
        max_seqlen_q = 0
        seqlen_k = []
        max_seqlen_k = 0

        block_size = self.config.infer_config.block_size

        for s in inp.seqs:
            input_ids.extend(s.tokens[s.computed_len: s.computed_len + s.new_len])
            position_ids.extend(list(range(s.computed_len, s.computed_len + s.new_len)))
            full_slot = sum([list(range(b * block_size, b * block_size + block_size)) for b in s.block_table], [])
            slot_mapping.extend(full_slot[s.computed_len: s.computed_len + s.new_len])
            block_tables.append(s.block_table)
            max_block_num = max(max_block_num, len(s.block_table))
            seqlen_q.append(s.new_len)
            max_seqlen_q = max(max_seqlen_q, s.new_len)
            seqlen_k.append(s.computed_len + s.new_len)
            max_seqlen_k = max(max_seqlen_k, s.computed_len + s.new_len)

        input_ids_ = torch.tensor(input_ids, dtype=torch.int64).to(self.device)
        position_ids_ = torch.tensor(position_ids, dtype=torch.int64).to(self.device)
        slot_mapping_ = torch.tensor(slot_mapping, dtype=torch.int64).to(self.device)
        block_tables_ = torch.tensor([b + [-1] * (max_block_num - len(b)) for b in block_tables], dtype=torch.int32).to(self.device)
        cu_seqlens_q_ = torch.tensor(list(accumulate(seqlen_q, initial=0)), dtype=torch.int32).to(self.device)
        max_seqlen_q_ = torch.tensor(max_seqlen_q, dtype=torch.int32).to(self.device)
        seqlen_k_ = torch.tensor(seqlen_k, dtype=torch.int32).to(self.device)
        max_seq_len_k_ = torch.tensor(max_seqlen_k, dtype=torch.int32).to(self.device)

        attn_meta = AttentionMetadata(
            slot_mapping=slot_mapping_,
            block_tables=block_tables_,
            cu_seqlen_q=cu_seqlens_q_,
            max_seqlen_q=max_seqlen_q_,
            seqlen_k=seqlen_k_,
            max_seqlen_k=max_seq_len_k_,
            k_cache=self.cache_storager.k_cache,
            v_cache=self.cache_storager.v_cache,
        )
        return input_ids_, position_ids_, attn_meta

    def step(self, model_input: WorkerInput):
        logger.info("worker input {}", model_input)
        # TODO 这个应该是有问题的, 应该按照请求到达顺序处理, 在worker侧或许有一个重排

        input_ids, position_ids, attn_meta = self._preprocess(model_input)
        logger.info("model forward {}", [input_ids, position_ids, str(attn_meta)])
        with set_forward_context(ForwardContext(attn_meta=attn_meta)):
            hidden_states = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
            )
        logits_indices = (attn_meta.cu_seqlen_q - 1)[1: ]
        logits = self.model.compute_logits(hidden_states[logits_indices])
        sample_tokens = self.sampler(model_input.seqs, logits=logits)
        ret = WorkerOutput(worker_id=self.id, seqs=[WorkerOutput.Info(output_tokens=[s]) for s in sample_tokens])
        logger.info("model output {}", ret)
        return ret

    def step_loop(self):
        while True:
            msg: WorkerInput = self.in_queue.get()

            ret = self.step(msg)

            self.out_queue.put(ret)

class CacheStorager:
    def __init__(self, config: Config,):
        self.param_dtype = config.model_config.get_param_dtype()
        self.layer_num = config.model_config.num_hidden_layers
        head_dim = config.model_config.hidden_size // config.model_config.num_attention_heads
        head_kv_num = config.model_config.num_key_value_heads
        block_size = config.infer_config.block_size
        self.block_shape = [block_size, head_kv_num, head_dim]

        free_mem, _ = torch.cuda.mem_get_info()
        self.block_num = free_mem // self._block_byte_num()
        kv_cache_shape = [self.block_num] + self.block_shape
        self.k_cache = [torch.empty(kv_cache_shape, dtype=self.param_dtype, device="cuda")
                        for _ in range(self.layer_num)]
        self.v_cache = [torch.empty(kv_cache_shape, dtype=self.param_dtype, device="cuda")
                        for _ in range(self.layer_num)]

    def _block_byte_num(self):
        single = torch.empty((), dtype=self.param_dtype).element_size()
        # 2 is k and v
        return 2 * self.layer_num * np.prod(self.block_shape) * single
