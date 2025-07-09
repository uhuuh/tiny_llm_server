import torch
import torch.nn as nn
from dataclasses import dataclass
from loguru import logger

from transformers import AutoTokenizer
import torchsnooper

from vllm._custom_ops import reshape_and_cache_flash
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)

# 一个向上取整的整数除法
def div_up(x, y):
    return (x + y - 1) // y

@dataclass
class SamplerConfig:
    temperature: float = 0
    # num_beams: int = 1
    top_k: int = 1
    top_p: float = 1
    max_new_token_new = 10

class Sampler(nn.Module):
    def __init__(self, config: SamplerConfig=None):
        super(Sampler, self).__init__()
        if config is None:
            config = SamplerConfig()
        self.config = config
    
    def forward(self, logits):
        #logger.info(f"hidden_states shape={logits.shape}")
        logits = logits[:, -1, :].unsqueeze(1)
        #logger.info(f"logits logtis={logits}")

        if self.config.temperature == 0:
            next_token = torch.argmax(logits, dim=-1)
            return next_token

        # TODO test
        logits = logits / self.config.temperature

        top_val, topk_idx = torch.topk(logits, self.config.top_k, dim=-1)
        new_logits = torch.zeros_like(logits)
        new_logits[topk_idx] = top_val
        new_logits = new_logits / new_logits.sum(dim=-1, keepdim=True)

        sort_val, sort_idx = new_logits.sort(dim=-1)
        acc = torch.cumsum(sort_val, dim=-1) / torch.sum(new_logits, dim=-1)
        acc[acc < self.config.top_p] = 0
        new_logits = acc / acc.sum(dim=-1, keepdim=True)

        new_logits = torch.softmax(new_logits)
        next_token = torch.multinomial(1, new_logits).sample()
        return next_token

@dataclass
class OPTConfig:
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    vocab_size: int = 50272
    max_position_embeddings: int = 2048
    word_embed_proj_dim: int = 512
    hidden_size: int = 1024
    ffn_dim: int = 4096
    dtype: str = "bfloat16"
    max_model_len: int = 2024
    max_batch_size: int = 1

def get_torch_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"dtype {dtype} not supported")

class GPTAttention(nn.Module):
    def __init__(self, config: OPTConfig):
        super(GPTAttention, self).__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scale = self.head_dim ** -0.5

    # @torchsnooper.snoop()
    def forward(self, hidden_state, position_ids, mask, kv_cache, layer_id, slot_mapping, block_table, context_lens):
        
        #print(f">>>>>>>>> hidden_state={hidden_state.shape}")
        start_pos_id = position_ids[:, 0]
        batch_size, query_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state).reshape(batch_size, query_len, self.head_num, self.head_dim)
        k = self.k_proj(hidden_state).reshape(batch_size, query_len, self.head_num, self.head_dim)
        v = self.v_proj(hidden_state).reshape(batch_size, query_len, self.head_num, self.head_dim)

        q_seq_start_locs = [0]
        max_q_seq_len = 0
        for i in range(context_lens.shape[0]):
            a = context_lens[i]
            q_seq_start_locs.append(q_seq_start_locs[i] + a)
            max_q_seq_len = max(max_q_seq_len, int(a))
        q_seq_start_locs = torch.tensor(q_seq_start_locs, dtype=torch.int32, device=hidden_state.device)
        max_context_len = len(context_lens)

        reshape_and_cache_flash(k, v, kv_cache[0][layer_id], kv_cache[1][layer_id],
                                slot_mapping=slot_mapping.reshape(-1),
                                kv_cache_dtype="auto",
                                k_scale=torch.tensor(1.0, dtype=torch.float32),
                                v_scale=torch.tensor(1.0, dtype=torch.float32))

        output = flash_attn_varlen_func(q.reshape(-1, self.head_num, self.head_dim), kv_cache[0][layer_id], kv_cache[1][layer_id],
                                        cu_seqlens_q=q_seq_start_locs ,
                                        max_seqlen_q=max_q_seq_len,
                                        seqused_k=context_lens,
                                        max_seqlen_k=max_context_len,
                                        block_table=block_table)

        output = output.reshape(batch_size, query_len, self.embedding_dim)
        return output
        # hidden_state = flash_attn_func(q, k, v, causal=True).reshape(batch_size, query_len, self.embedding_dim)

class GPTLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super(GPTLayer, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GPTAttention(config)

        self.fc1 = nn.Linear(self.config.hidden_size, self.config.ffn_dim)
        self.fc2 = nn.Linear(self.config.ffn_dim, self.config.hidden_size)
        self.act = nn.ReLU()
        self.self_attn_layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(self.config.hidden_size)

    def forward(self, hidden_states, position_ids, mask, kv_cache, layer_id, slot_mapping, block_table, context_lens):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_ids, mask, kv_cache,
                                       layer_id, slot_mapping, block_table, context_lens)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim) # TODO 为什么要加2

    def forward(self, positions: torch.LongTensor):
        return super().forward(positions + self.offset)

class GPTModel(nn.Module):
    def __init__(self, config: OPTConfig):
        super(GPTModel, self).__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.word_embed_proj_dim)
        self.embed_positions = OPTLearnedPositionalEmbedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.project_in = nn.Linear(self.config.word_embed_proj_dim, self.config.hidden_size, bias=False)
        self.project_out = nn.Linear(self.config.hidden_size, self.config.word_embed_proj_dim, bias=False)

        self.layers = nn.ModuleList([
            GPTLayer(config) for _ in range(self.config.num_hidden_layers)
        ])

    def forward(self, input_ids, position_ids, mask, kv_cache, slot_mapping, block_table, context_lens):
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        input_embeds = self.project_in(input_embeds)
        hidden_states = input_embeds + position_embeds

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, position_ids, mask, kv_cache, i, slot_mapping, block_table, context_lens)

        # [batch_size, seq_len, hidden_dim] @ [vocab_size, hidden_dim].T
        hidden_states = self.project_out(hidden_states)
        logits = hidden_states @ self.embed_tokens.weight.T
        return logits

class OPTForCausalLM(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config

        defalut_dtype = torch.get_default_dtype()
        torch.set_default_dtype(get_torch_dtype(config.dtype))
        self.decoder = GPTModel(config)
        torch.set_default_dtype(defalut_dtype)

        self.sampler = Sampler()

    def forward(self, *args, **kwargs):
        logtis = self.decoder(*args, **kwargs)
        # TODO 采样如何控制，一部分逻辑根据logits生成新token， 一部分逻辑在该请求是否处理完
        next_tokens = self.sampler(logtis)
        return next_tokens


if __name__ == "__main__":
    config = OPTConfig(
        num_hidden_layers=24,
        num_attention_heads=16,
        vocab_size=50272,
        # max_position_embeddings=2048, # TODO why is 2050
        max_position_embeddings=2048,
        word_embed_proj_dim=512,
        hidden_size=1024,
        ffn_dim=4096,
        dtype="bfloat16"
    )

    model = OPTForCausalLM(config)
    model = model.to("cuda")
    model.eval()

    pa = "/mnt/c/Users/uh/code/ckpt/opt-350m/"
    pa_weight = pa + "pytorch_model.bin"
    state_dict = torch.load(pa_weight)

    tokenizer = AutoTokenizer.from_pretrained(pa)

    model.load_state_dict(state_dict)

    input_token_ids = torch.tensor([2, 2264,   32,   52,  519,   13, 3630,  116])
    #  text=[2,  2264,    32,    52,   519,    13,  3630,   116, 50118]
    input_token_ids = input_token_ids.reshape(1, -1)

    # 已经实现了一个接口 `register_hooks` 来一次性为模型所有子模块注册钩子
    # 以下代码保持不变，它就是调用一个接口实现注册的方式
    # print_model_parameter_weights_hash(model)
    # register_hooks(model)
    # print("model_state_dict", model.state_dict(), flush=True)
    block_num = 1024
    block_size = 16
    max_batch_size = 1
    max_new_token = 10
    prompt_len = config.max_model_len

    kv_cache = [torch.empty(
        config.num_hidden_layers,
        block_num,
        block_size,
        config.num_attention_heads, # 实际上这里应该是kv头数量
        config.hidden_size // config.num_attention_heads,
        dtype=get_torch_dtype(config.dtype), device="cuda") for _ in range(2)]
    slot_mapping = torch.zeros((max_batch_size, prompt_len), dtype=torch.long, device="cuda")
    block_table = torch.zeros((max_batch_size, div_up(prompt_len, block_size)),
                              dtype=torch.int32, device="cuda")
    context_lens = torch.zeros((max_batch_size, ), dtype=torch.int32, device="cuda")

    input_ids = torch.empty((max_batch_size, prompt_len), dtype=torch.int32, device="cuda")
    position_ids = torch.arange(0, prompt_len).unsqueeze(0).to("cuda")
    mask = torch.full((prompt_len, prompt_len), -float('inf')).to("cuda")
    mask = torch.triu(mask, diagonal=1)

    cur = input_token_ids.shape[1]
    input_ids[:, :cur] = input_token_ids
    context_lens[:] = cur

    # 似乎slot_mapping和block_table长度必须与输入相匹配，要不然访问存储报错
    slot_mapping[:] = torch.arange(prompt_len, dtype=slot_mapping.dtype, device="cuda").reshape(1, -1)
    block_table[:] = torch.arange(div_up(prompt_len, block_size), dtype=block_table.dtype, device="cuda").reshape(1, -1)

    print(slot_mapping)
    print(block_table)

    for iter in range(max_new_token):
        now_input_ids = input_ids[:, :cur]
        now_position_ids = position_ids[:, :cur]
        now_mask = mask[:cur, :cur]

        slot_mapping = slot_mapping[:, :cur]
        block_table = block_table[:, :div_up(cur, block_size)]

        output = model(now_input_ids, now_position_ids, now_mask, kv_cache, slot_mapping, block_table, context_lens)

        input_ids[:, cur] = output
        cur += 1

        if output.item() == tokenizer.eos_token_id:
            break

    result = input_ids[0].cpu().numpy()
    logger.info(f"result={result}, result_text={tokenizer.decode(result)}")




