import torch
import torch.nn.functional as F
from torch import distributed as dist, nn as nn

from src.attention import FlashAttentionBackend
from src.base import Config
from src.utils import SingletonMeta, get_forward_context


class RotaryPositionalEmbedding(nn.Module, metaclass=SingletonMeta):
    def __init__(self, param_dtype, head_dim: int, max_seq_len, base: int = 1000000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.inv_freq = (1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)))

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1).reshape(-1, head_dim)

        # self.freqs_cos: torch.Tensor
        # self.freqs_sin: torch.Tensor
        # register buffer后，使得model.to(device)时可以修改自动修改device
        self.register_buffer("freqs_cos", emb.cos().to(param_dtype), persistent=False)
        self.register_buffer("freqs_sin", emb.sin().to(param_dtype), persistent=False)

    def _get_cos_sin(self, position_ids: torch.Tensor):
        return self.freqs_cos[position_ids, None, :], self.freqs_sin[position_ids, None, :]

    def apply_rotary_pos_emb(self, q, k, position_ids):
        cos, sin = self._get_cos_sin(position_ids)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _rotate_half(self, x: torch.Tensor):
        last_dim = x.shape[-1]
        x1 = x[..., : last_dim // 2]
        x2 = x[..., last_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tp_group = config.parallel_config.tp_group
        self.n_heads = config.model_config.num_attention_heads
        self.n_kv_heads = config.model_config.num_key_value_heads
        self.head_dim = config.model_config.hidden_size // self.n_heads
        self.hidden_dim = config.model_config.hidden_size
        self.q_proj = nn.Linear(self.hidden_dim, (self.n_heads // config.parallel_config.tp_size) * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_dim, (self.n_kv_heads // config.parallel_config.tp_size) * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_dim, (self.n_kv_heads // config.parallel_config.tp_size) * self.head_dim)
        self.o_proj = nn.Linear(self.q_proj.out_features, self.hidden_dim, bias=False)
        self.attn_backend = FlashAttentionBackend()
        self.rotary_emb = RotaryPositionalEmbedding(
            config.model_config.get_param_dtype(),
            self.head_dim,
            config.model_config.max_position_embeddings,
        )

    def forward(self, hidden_states, position_ids):
        q = self.q_proj(hidden_states).view(-1, self.n_heads // (dist.get_world_size(self.tp_group) if self.tp_group else 1), self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.n_kv_heads // (dist.get_world_size(self.tp_group) if self.tp_group else 1), self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.n_kv_heads // (dist.get_world_size(self.tp_group) if self.tp_group else 1), self.head_dim)
        q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, position_ids)
        y = self.attn_backend(q, k, v).view(-1, self.q_proj.out_features)
        y = self.o_proj(y)
        if self.tp_group is not None:
            dist.all_reduce(y, group=self.tp_group)
        return y

class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps)).to(dtype)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tp_group = config.parallel_config.tp_group
        local_inter = config.model_config.intermediate_size // config.parallel_config.tp_size
        self.gate_proj = nn.Linear(config.model_config.hidden_size, local_inter, bias=False)
        self.up_proj = nn.Linear(config.model_config.hidden_size, local_inter, bias=False)
        self.down_proj = nn.Linear(local_inter, config.model_config.hidden_size, bias=False)

    def forward(self, x):
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        if self.tp_group is not None:
            dist.all_reduce(out, group=self.tp_group)
        return out

class DecodeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.model_config.hidden_size, config.model_config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.model_config.hidden_size, config.model_config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states, position_ids):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_ids)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.model_config.vocab_size, config.model_config.hidden_size)
        self.layers = nn.ModuleList(DecodeLayer(config) for _ in range(config.model_config.num_hidden_layers))
        self.norm = RMSNorm(config.model_config.hidden_size, config.model_config.rms_norm_eps)

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids)
        ctx = get_forward_context()
        for i, layer in enumerate(self.layers):
            ctx.layer_idx = i
            hidden_states = layer(hidden_states, position_ids)
        return self.norm(hidden_states)

class Qwen2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.tie_word_embeddings = config.model_config.tie_word_embeddings
        if self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.model_config.hidden_size, config.model_config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, position_ids)

    def compute_logits(self, hidden_states):
        if self.tie_word_embeddings:
            return F.linear(hidden_states, self.lm_head.weight)
        else:
            return self.lm_head(hidden_states)
