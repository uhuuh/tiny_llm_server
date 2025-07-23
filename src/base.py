import enum
import glob
import os
import threading as th
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

import torch
from loguru import logger
from safetensors.torch import load_file
from torch import nn as nn


@dataclass
class Qwen2Config:
    architectures: List[str] = None
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 896
    initializer_range: float = 0.02
    intermediate_size: int = 4864
    max_position_embeddings: int = 32768
    max_window_layers: int = 21
    model_type: str = "qwen2"
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.43.1"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936

    def get_param_dtype(self):
        assert self.torch_dtype == "bfloat16"
        return torch.bfloat16

@dataclass
class SampleConfig:
    temperature: float = 0
    top_k: int = 1
    max_new_token_new: int = 10

@dataclass
class InferConfig:
    gpu_memory_utilization: float
    block_size: int
    model_path: str
    max_req_num: int
    max_batch_token_num: int
    max_prefill_len: int
    data_parallel: int = 1
    tensor_parallel: int = 1
    enable_debug: bool = False
    enable_prefix_cache: bool = False
    enable_chunked_prefill: bool = False

@dataclass
class ParallelConfig:
    dp_size: int = 1
    dp_rank: int = 0
    dp_group: torch.distributed.ProcessGroup = None
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: torch.distributed.ProcessGroup = None


@dataclass
class Config:
    infer_config: InferConfig
    model_config: Qwen2Config = field(default_factory=Qwen2Config)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    def __post_init__(self):
        # TODO from infer_config init model_config
        pass

@dataclass
class ModelInput:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    num_prefill_tokens: int
    k_cache: List[torch.Tensor]
    v_cache: List[torch.Tensor]
    slot_mapping: torch.Tensor
    prefill_cu_seqlens_q: torch.Tensor # [batch_size + 1]
    prefill_max_seqlen_q: torch.Tensor
    decode_block_table: torch.Tensor
    decode_seq_lens: torch.Tensor # [batch_size]
    decode_max_seq_len: torch.Tensor
    decode_max_seq_q_len: torch.Tensor
    decode_cu_seq_q_lens: torch.Tensor
    layer_idx: Optional[int] = None
    hidden_states: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None

    def __str__(self):
        info = asdict(self)

        del info["k_cache"], info["v_cache"]

        display_ele_limit = 16
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v if v.numel() <= display_ele_limit else [v.shape, v.dtype, v.device]

        return str(info)

    def __repr__(self):
        return self.__str__()

class DeviceType(Enum):
    CPU = 0
    GPU = 1

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, param_dtype, dim: int, max_seq_len, base: int = 1000000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.inv_freq = (1.0 / (base ** (torch.arange(0, dim, 2, device="cpu").float() / dim))).cuda()

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1).reshape(-1, dim)
        self.freqs_cos = emb.cos().to(param_dtype)
        self.freqs_sin = emb.sin().to(param_dtype)

    def get_cos_sin(self, position_ids: torch.Tensor):
        # NOTE: support [token_num, head_num, head_dim] layout
        return self.freqs_cos[position_ids, None, :], self.freqs_sin[position_ids, None, :]

    @classmethod
    def apply_rotary_pos_emb(cls, x: ModelInput):
        q = x.q * x.cos + cls.rotate_half(x.q) * x.sin
        k = x.k * x.cos + cls.rotate_half(x.k) * x.sin
        return q, k

    @classmethod
    def rotate_half(cls, x: torch.Tensor):
        last_dim = x.shape[-1]
        x1 = x[..., : last_dim // 2]
        x2 = x[..., last_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def ceil_div(a, b):
    return (a + b - 1) // b


class RequestParam(BaseModel):
    model: str
    prompt: List[int] # TODO should str
    temperature: float
    max_tokens: int
    stream: bool

class MessageType(enum.Enum):
    engine_start = enum.auto()
    request = enum.auto()
    response = enum.auto()

    def __hash__(self):
        return hash(self.value)

class Listener:
    def __init__(self, queue, handlers):
        self.queue = queue
        self.handlers = {k.value: v for k, v in handlers.items()}
        logger.info(f"init listener {self.handlers}")
        self.th = th.Thread(target=self.listen)
        self.th.start()

    def listen(self):
        while True:
            temp = self.queue.get()
            logger.info(">>> debug listen temp={}", temp)
            msg_type, msg_body = temp

            logger.info(">>> debug listen msg_type={} msg_body={} handlers={} queue={}",
                        msg_type, msg_body, self.handlers, self.queue)
            for context, handler in self.handlers[msg_type.value]:
                handler(context, msg_body)


@dataclass
class ScheduleInfo:
    now_batch_token_num: int = 0
    now_req_num: int = 0
    req_new_token_nums: Dict[int, int] = field(default_factory=dict)
