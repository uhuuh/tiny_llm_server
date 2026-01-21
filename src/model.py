from dataclasses import asdict
from typing import Optional

from loguru import logger
import torch.nn.functional as F

from src.base import *

from torch import distributed as dist, nn as nn

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

class GroupQueryAttention(nn.Module):
    def __init__(self, head_num, head_kv_num, head_dim):
        super().__init__()
        self.head_num = head_num
        self.head_kv_num = head_kv_num
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.repeat_kv_num = self.head_num // self.head_kv_num

    def forward(self, x: ModelInput):
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

from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm._custom_ops import reshape_and_cache_flash
class FlashAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_cache_dtype = "auto"
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    def forward(self, x: ModelInput):
        num_token = x.q.shape[0]
        #o = torch.empty_like(x.q) # TODO
        o = torch.zeros_like(x.q)

        reshape_and_cache_flash(
            key=x.k,
            value=x.v,
            key_cache=x.k_cache[x.layer_idx],
            value_cache=x.v_cache[x.layer_idx],
            slot_mapping=x.slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=self._k_scale,
            v_scale=self._v_scale,
        )

        if x.num_prefill_tokens > 0:
            flash_attn_varlen_func(
                q=x.q[: x.num_prefill_tokens],
                k=x.k[: x.num_prefill_tokens],
                v=x.v[: x.num_prefill_tokens],
                cu_seqlens_q=x.prefill_cu_seqlens_q,
                max_seqlen_q=x.prefill_max_seqlen_q,
                cu_seqlens_k=x.prefill_cu_seqlens_q,
                max_seqlen_k=x.prefill_max_seqlen_q,
                causal=True,
                out=o[: x.num_prefill_tokens],
            )

        if num_token - x.num_prefill_tokens > 0:
            flash_attn_varlen_func(
                q=x.q[x.num_prefill_tokens: ],
                k=x.k_cache[x.layer_idx],
                v=x.v_cache[x.layer_idx],
                cu_seqlens_q=x.decode_cu_seq_q_lens,
                max_seqlen_q=x.decode_max_seq_q_len,
                seqused_k=x.decode_seq_lens,
                max_seqlen_k=x.decode_max_seq_len,
                causal=True,
                block_table=x.decode_block_table,
                out=o[x.num_prefill_tokens: ],
            )

        return o


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tp_size = config.parallel_config.tp_size
        self.tp_group = config.parallel_config.tp_group
        self.n_heads = config.model_config.num_attention_heads
        self.n_kv_heads = config.model_config.num_key_value_heads
        self.local_n_heads = self.n_heads // self.tp_size
        self.local_n_kv_heads = self.n_kv_heads // self.tp_size

        self.hidden_dim = config.model_config.hidden_size
        self.head_dim = config.model_config.hidden_size // config.model_config.num_attention_heads
        self.q_dim = self.local_n_heads * self.head_dim
        self.kv_dim = self.local_n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.q_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.kv_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.kv_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_dim, self.q_dim, bias=False)
        #self.attn_backend = GroupQueryAttention(self.n_heads, self.n_kv_heads, self.head_dim)
        self.attn_backend = FlashAttention()

    def forward(self, x: ModelInput):
        h = x.hidden_states
        x.q = self.q_proj(h).view(-1, self.n_heads, self.head_dim)
        x.k = self.k_proj(h).view(-1, self.n_kv_heads, self.head_dim)
        x.v = self.v_proj(h).view(-1, self.n_kv_heads, self.head_dim)

        x.q, x.k = RotaryPositionalEmbedding.apply_rotary_pos_emb(x)

        y = self.attn_backend(x).view(-1, self.hidden_dim)
        y = self.o_proj(y)
        if self.tp_group is not None:
            logger.info(">>> debug attn reduce before")
            dist.all_reduce(y, group=self.tp_group)
            logger.info(">>> debug attn reduce after")
        return y


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
    def __init__(self, config: Config):
        super().__init__()
        self.tp_size = config.parallel_config.tp_size
        self.tp_group = config.parallel_config.tp_group
        self.local_intermediate_size = config.model_config.intermediate_size // self.tp_size
        self.gate_proj = nn.Linear(config.model_config.hidden_size, self.local_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.model_config.hidden_size, self.local_intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.local_intermediate_size, config.model_config.hidden_size, bias=False)

    def forward(self, x):
        output = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        if self.tp_group is not None:
            dist.all_reduce(output, group=self.tp_group)
        return output


class DecodeLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        n_embed, eps = config.model_config.hidden_size, config.model_config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed, eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed, eps)
        self.mlp = MLP(config)

    def forward(self, x: ModelInput):
        residual = x.hidden_states
        x.hidden_states = self.input_layernorm(x.hidden_states)
        x.hidden_states = residual + self.self_attn(x)

        residual = x.hidden_states
        x.hidden_states = self.post_attention_layernorm(x.hidden_states)
        x.hidden_states = residual + self.mlp(x.hidden_states)
        return x.hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.model_config.vocab_size, config.model_config.hidden_size)
        self.layers = nn.ModuleList(DecodeLayer(config) for _ in range(config.model_config.num_hidden_layers))
        self.norm = RMSNorm(config.model_config.hidden_size, config.model_config.rms_norm_eps)

    def forward(self, x: ModelInput):
        for i, layer in enumerate(self.layers):
            x.layer_idx = i
            x.hidden_states = layer(x)
        x.hidden_states = self.norm(x.hidden_states)
        return x.hidden_states

class Qwen2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)

        self.lm_head = None
        if not config.model_config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.model_config.hidden_size, config.model_config.vocab_size, bias=False)

    def forward(self, x: ModelInput) -> torch.Tensor:
        x.hidden_states = self.model.embed_tokens(x.input_ids)
        x.hidden_states = self.model(x)
        if self.lm_head is None:
            logits = torch.matmul(x.hidden_states, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x.hidden_states)
        return logits


