import torch
import torch.nn as nn
from dataclasses import dataclass
from loguru import logger
from scipy.special.tests.test_data import data_local
from transformers import AutoTokenizer
import torchsnooper
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union
from dataclasses import dataclass
from typing import List


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

    def __post_init__(self):
        # 设置 architectures 的默认值为空列表（避免 None 的情况）
        if self.architectures is None:
            self.architectures = ["Qwen2ForCausalLM"]

@dataclass
class Modelinput:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, param_dtype, dim: int, max_seq_len: int = 32768, base: int = 1000000):
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
    def apply_rotary_pos_emb(cls, x: Modelinput):
        q = x.q * x.cos + cls.rotate_half(x.q) * x.sin
        k = x.k * x.cos + cls.rotate_half(x.k) * x.sin
        return q, k

    @classmethod
    def rotate_half(cls, x: torch.Tensor):
        last_dim = x.shape[-1]
        x1 = x[..., : last_dim // 2]
        x2 = x[..., last_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class GroupQueryAttention(nn.Module):
    def __init__(self, head_num, head_kv_num, head_dim):
        super().__init__()
        self.head_num = head_num
        self.head_kv_num = head_kv_num
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.repeat_kv_num = self.head_num // self.head_kv_num

    def forward(self, x: Modelinput):
        q, k, v = x.q, x.k, x.v
        k = k.repeat_interleave(self.repeat_kv_num, dim=1)
        v = v.repeat_interleave(self.repeat_kv_num, dim=1)

        # NOTE: scaled_dot_product_attention only support [bs, head_num, seq_len, head_dim] layout
        assert x.attn_mask is None
        q = q.view(1, -1, self.head_num, self.head_dim).transpose(1, 2)
        k = k.view(1, -1, self.head_num, self.head_dim).transpose(1, 2)
        v = v.view(1, -1, self.head_num, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        return y

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()

        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads

        self.hidden_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.kv_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.kv_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_backend = GroupQueryAttention(self.n_heads, self.n_kv_heads, self.head_dim)

    def forward(self, x: Modelinput):
        h = x.hidden_states
        x.q = self.q_proj(h).view(-1, self.n_heads, self.head_dim)
        x.k = self.k_proj(h).view(-1, self.n_kv_heads, self.head_dim)
        x.v = self.v_proj(h).view(-1, self.n_kv_heads, self.head_dim)

        x.q, x.k = RotaryPositionalEmbedding.apply_rotary_pos_emb(x)

        y = self.attn_backend(x).view(-1, self.hidden_dim)
        y = self.o_proj(y)
        return y


from util import hash_tensor

# @torchsnooper.snoop()
class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * x.to(input_dtype)
        return output

class MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        n_embed, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed, eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed, eps)
        self.mlp = MLP(config)

    def forward(self, x: Modelinput):
        residual = x.hidden_states
        x.hidden_states = self.input_layernorm(x.hidden_states)
        x.hidden_states = residual + self.self_attn(x)

        residual = x.hidden_states
        x.hidden_states = self.post_attention_layernorm(x.hidden_states)
        x.hidden_states = residual + self.mlp(x.hidden_states)
        return x.hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: Modelinput):
        for layer in self.layers:
            x.hidden_states = layer(x)
        x.hidden_states = self.norm(x.hidden_states)
        return x.hidden_states

class Qwen2(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: Modelinput) -> torch.Tensor:
        x.hidden_states = self.model.embed_tokens(x.input_ids)
        x.hidden_states = self.model(x)
        if self.lm_head is None:
            logits = torch.matmul(x.hidden_states, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x.hidden_states)
        return logits

from safetensors.torch import load_file
import os
import glob

def load_weight(model, dir_path: str, device="cuda"):
    merged = {}
    pattern = os.path.join(dir_path, "*.safetensors")
    for path in sorted(glob.glob(pattern)):
        tensors = load_file(path, device=device)
        merged.update(tensors)
    missing_keys, unexpected_keys = model.load_state_dict(merged, strict=False)
    logger.info("load weight missing_keys: {} unexpected_keys: {}", missing_keys, unexpected_keys)

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    pa_model = "/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct"
    config = Qwen2Config()
    # config.num_hidden_layers = 2

    tokenizer = AutoTokenizer.from_pretrained(pa_model)

    model = Qwen2(config)
    model = model.to("cuda").eval()

    load_weight(model, pa_model)

    text = [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465,
       553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847,
       13, 151645, 198, 151644, 872, 198, 14990, 1879, 151645,
       198, 151644, 77091, 198]
    output_text = [9707,    0, 2585,  646,  358]
    max_new_token = 5

    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryPositionalEmbedding(torch.bfloat16, head_dim, config.max_position_embeddings)

    for i in range(max_new_token):
        seq_len = len(text)

        position_ids = torch.arange(seq_len).view(-1).to("cuda")
        cos, sin = rope.get_cos_sin(position_ids)
        input_ids = torch.tensor(text).view(-1).to("cuda")

        model_input = Modelinput(
            input_ids=input_ids,
            position_ids=position_ids,
            cos=cos,
            sin=sin
        )
        with torch.inference_mode():
            output = model(model_input)

        logits = output[-1, :]
        next_token = torch.argmax(logits).item()
        text.append(next_token)
        logger.info("iter {} output_token {}", i, next_token)

    logger.info("token_id_list={} result={}", text, tokenizer.decode(text))


