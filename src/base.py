from dataclasses import dataclass, field
from typing import List, Any, Dict
from pydantic import BaseModel

import torch


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
    max_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.7

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


def ceil_div(a, b):
    return (a + b - 1) // b


class RequestParam(BaseModel):
    model: str
    prompt: List[int] # TODO should str
    temperature: float
    max_tokens: int
    stream: bool

@dataclass
class BaseMessage:
    pass

@dataclass
class SchedulerInitEndMessage:
    scheduler_id: str

@dataclass
class SchedulerReqRecvMessage:
    @dataclass
    class RequestInputInfo:
        request_id: int
        prompt_tokens: List[int]
        sample_config: SampleConfig
    requests: List[RequestInputInfo]

@dataclass
class SchedulerReqFinishMessage:
    @dataclass
    class RequestOutputInfo:
        request_id: str
        prompt_tokens: List[int]
        output_tokens: List[int]
        output_text: str

    scheduler_id: str
    requests: List[RequestOutputInfo]

@dataclass
class WorkerInitEndMessage:
    worker_id: str
    block_num: int

@dataclass
class WorkerStepStartMessage:
    args: Any

@dataclass
class WorkerStepEndMessage:
    worker_id: str
    rets: Any

@dataclass
class ScheduleInfo:
    now_batch_token_num: int = 0
    now_req_num: int = 0
    req_new_token_nums: Dict[int, int] = field(default_factory=dict)
