import torch.nn as nn
from loguru import logger
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

from example.base import ModelInput, Qwen2Config, load_weight, RotaryPositionalEmbedding
from model.util import register_hooks


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

from vllm.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
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
            # NOTE: only support [bs, seq, head_num, head_dim] layout
            flash_attn_with_kvcache(
                q=x.q[x.num_prefill_tokens: ].unsqueeze(1),
                k_cache=x.k_cache[x.layer_idx],
                v_cache=x.v_cache[x.layer_idx],
                block_table=x.decode_block_table,
                cache_seqlens=x.decode_seq_lens,
                causal=True,
                out=o[x.num_prefill_tokens: ].unsqueeze(1),
            )

        return o


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
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecodeLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        n_embed, eps = config.hidden_size, config.rms_norm_eps
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
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecodeLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: ModelInput):
        for i, layer in enumerate(self.layers):
            x.layer_idx = i
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

    def forward(self, x: ModelInput) -> torch.Tensor:
        x.hidden_states = self.model.embed_tokens(x.input_ids)
        x.hidden_states = self.model(x)
        if self.lm_head is None:
            logits = torch.matmul(x.hidden_states, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x.hidden_states)
        return logits


import os

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

    # register_hooks(model)

    load_weight(model, pa_model)

    text = [151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,     40,   1079, 151645,
            198, 151644,  77091,    198]
    output_text = [9707,    0, 2585,  646,  358]
    max_new_token = 50

    block_num = 10
    block_size = 128
    head_kv_num = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryPositionalEmbedding(torch.bfloat16, head_dim, config.max_position_embeddings)
    k_cache = [torch.zeros([block_num, block_size, head_kv_num, head_dim]) for _ in range(config.num_hidden_layers)]
    v_cache = [torch.zeros([block_num, block_size, head_kv_num, head_dim]) for _ in range(config.num_hidden_layers)]
    prompt_len = len(text)
    assert prompt_len + max_new_token <= block_size

    for i in range(max_new_token):
        seq_len = len(text)

        if i == 0:
            position_ids = torch.arange(seq_len).view(-1).to("cuda")
            cos, sin = rope.get_cos_sin(position_ids)
            input_ids = torch.tensor(text).view(-1).to("cuda")

            model_input = ModelInput(
                input_ids=input_ids,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                num_prefill_tokens=seq_len,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=torch.arange(0, seq_len),
                prefill_cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32),
                prefill_max_seqlen_q=torch.tensor(seq_len, dtype=torch.int32),
                decode_block_table = torch.empty([]),
                decode_seq_lens=torch.empty([]),
            )
        else:
            position_ids = torch.tensor([seq_len - 1]).view(-1).to("cuda")
            cos, sin = rope.get_cos_sin(position_ids)
            input_ids = torch.tensor([text[-1]]).view(-1).to("cuda")

            model_input = ModelInput(
                input_ids=input_ids,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                num_prefill_tokens=0,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=position_ids,
                prefill_cu_seqlens_q=torch.empty([]),
                prefill_max_seqlen_q=torch.empty([]),
                decode_block_table=torch.tensor([[0]], dtype=torch.int32),
                decode_seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            )

        with torch.inference_mode():
            output = model(model_input)

        logits = output[-1, :]
        next_token = torch.argmax(logits).item()
        text.append(next_token)
        logger.info("iter {} output_token {}", i, next_token)

    logger.info("token_id_list={} result={}", text, tokenizer.decode(text))


