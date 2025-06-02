import torch
import torch.nn as nn
from dataclasses import dataclass
from loguru import logger
from util import register_hooks, print_model_parameter_weights_hash, hash_tensor
from transformers import AutoTokenizer
import torchsnooper


@dataclass
class SamplerConfig:
    temperature: float = 0
    # num_beams: int = 1
    top_k: int = 1
    top_p: float = 1

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
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    max_position_embeddings: int
    word_embed_proj_dim: int
    hidden_size: int
    ffn_dim: int
    dtype: str
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
    def forward(self, hidden_state, position_ids, mask, kv_cache, layer_id):
        
        #print(f">>>>>>>>> hidden_state={hidden_state.shape}")
        start_pos_id = position_ids[:, 0]
        batch_size, query_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state).reshape(batch_size, query_len, self.head_num, self.head_dim)
        k = self.k_proj(hidden_state).reshape(batch_size, query_len, self.head_num, self.head_dim)
        v = self.v_proj(hidden_state).reshape(batch_size, query_len, self.head_num, self.head_dim)

        kv_cache[0][:, start_pos_id: start_pos_id + query_len, layer_id] = k
        kv_cache[1][:, start_pos_id: start_pos_id + query_len, layer_id] = v
        k = kv_cache[0][:, : start_pos_id + query_len, layer_id]
        v = kv_cache[1][:, : start_pos_id + query_len, layer_id]

        q = q.permute(0, 2, 1, 3) * self.scale # [batch_size, head_num, query_len, head_dim]
        k = k.permute(0, 2, 3, 1) # [batch_size, head_num, head_dim, key_len]
        v = v.permute(0, 2, 1, 3) # [batch_size, head_num, key_len, head_dim]

        qk = torch.matmul(q, k)
        #logger.info(f"attn_weights hash={hash_tensor(qk)} shape={qk.shape}")

        qk += mask
        #logger.info(f"mask hash={hash_tensor(qk)} shape={qk.shape}")
        # logger.info(f"mask={mask}")

        qk = qk.softmax(dim=-1) # 这里应该强制使用fp32
        #logger.info(f"softmax hash={hash_tensor(qk)} shape={qk.shape}")

        qkv = qk @ v
        #logger.info(f"final_matmul hash={hash_tensor(qkv)} shape={qkv.shape}")

        qkv = qkv.permute(0, 2, 1, 3).reshape(batch_size, query_len, self.embedding_dim) # [batch_size, query_len, embedding_dim]
        hidden_state = self.out_proj(qkv)
        #logger.info(f"out_proj hash={hash_tensor(hidden_state)} shape={hidden_state.shape}")

        return hidden_state

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

    def forward(self, hidden_states, position_ids, mask, kv_cache, layer_id):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_ids, mask, kv_cache, layer_id)
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

    def forward(self, input_ids, position_ids, mask, kv_cache):
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        input_embeds = self.project_in(input_embeds)
        hidden_states = input_embeds + position_embeds

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, position_ids, mask, kv_cache, i)

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
        dtype="float32"
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

    kv_cache = [torch.empty(
        config.max_batch_size,
        config.max_model_len, 
        config.num_hidden_layers, 
        config.num_attention_heads, 
        config.hidden_size // config.num_attention_heads,
        dtype=get_torch_dtype(config.dtype), device="cuda") for _ in range(2)]
    input_ids = torch.empty((1, config.max_model_len), dtype=torch.long, device="cuda")
    position_ids = torch.arange(0, config.max_model_len).unsqueeze(0).to("cuda")
    mask = torch.full((config.max_model_len, config.max_model_len), -float('inf')).to("cuda")
    mask = torch.triu(mask, diagonal=1)

    cur = input_token_ids.shape[1]
    input_ids[:, :cur] = input_token_ids

    for iter in range(100):
        now_input_ids = input_ids[:, :cur]
        now_position_ids = position_ids[:, :cur]
        now_mask = mask[:cur, :cur]
        output = model(now_input_ids, now_position_ids, now_mask, kv_cache)

        input_ids[:, cur] = output
        cur += 1

        if output.item() == tokenizer.eos_token_id:
            break

    result = input_ids[0].cpu().numpy()
    logger.info(f"result={result}, result_text={tokenizer.decode(result)}")
