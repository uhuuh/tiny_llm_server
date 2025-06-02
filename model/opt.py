import torch
import torch.nn as nn
from dataclasses import dataclass
from loguru import logger
from util import register_hooks, print_model_parameter_weights_hash, hash_tensor
from transformers import AutoTokenizer


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

class ops:
    flash_atte = None
    paged_atte = None
    reshape_and_cache = None

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

    def forward(self, hidden_state, mask):
        #print(f">>>>>>>>> hidden_state={hidden_state.shape}")
        batch_size, seq_len, _ = hidden_state.shape
        q = self.q_proj(hidden_state).reshape(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3) * self.scale
        k = self.k_proj(hidden_state).reshape(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_state).reshape(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        qk = torch.matmul(q, k.permute(0, 1, 3, 2))
        #logger.info(f"attn_weights hash={hash_tensor(qk)} shape={qk.shape}")

        qk += mask
        #logger.info(f"mask hash={hash_tensor(qk)} shape={qk.shape}")
        # logger.info(f"mask={mask}")

        qk = qk.softmax(dim=-1) # 这里应该强制使用fp32
        #logger.info(f"softmax hash={hash_tensor(qk)} shape={qk.shape}")

        qkv = qk @ v
        #logger.info(f"final_matmul hash={hash_tensor(qkv)} shape={qkv.shape}")

        qkv = qkv.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embedding_dim) # [batch_size, seq_len, embedding_dim]
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

    def forward(self, hidden_states, mask):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, mask)
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

    def forward(self, input_ids, psoition_ids):
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(psoition_ids)
        input_embeds = self.project_in(input_embeds) # TODO 是这样使用的吗
        hidden_states = input_embeds + position_embeds

        _, seq_len, _ = hidden_states.shape
        mask = torch.full((seq_len, seq_len), -float('inf')).to(input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, mask)

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

    def forward(self, input_ids, position_ids):
        logtis = self.decoder(input_ids, position_ids)
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

    print(model)
    # for k, v in model.state_dict().items():
    #     print(k, v.shape, v.dtype)

    pa = "/mnt/c/Users/uh/code/ckpt/opt-350m/"
    pa_weight = pa + "pytorch_model.bin"
    state_dict = torch.load(pa_weight)

    tokenizer = AutoTokenizer.from_pretrained(pa)
    # print(tokenizer.encode("hello world"))
    # print(tokenizer.eos_token_id)
    # exit()

    model.load_state_dict(state_dict)

    input_ids = torch.tensor([2, 2264,   32,   52,  519,   13, 3630,  116])
    #  text=[2,  2264,    32,    52,   519,    13,  3630,   116, 50118]
    input_ids = input_ids.reshape(1, -1)

    # 已经实现了一个接口 `register_hooks` 来一次性为模型所有子模块注册钩子
    # 以下代码保持不变，它就是调用一个接口实现注册的方式
    # print_model_parameter_weights_hash(model)
    # register_hooks(model)
    # print("model_state_dict", model.state_dict(), flush=True)

    for iter in range(500):

        # print(f">>>>>> input_ids={input_ids.shape} position_ids={position_ids.shape}")
        position_ids = torch.arange(0, input_ids.numel()).unsqueeze(0)

        input_ids = input_ids.to("cuda")
        position_ids = position_ids.to("cuda")
        output = model(input_ids, position_ids)

        # logger.info(f">>> iter={iter} input={input_ids} input_shape={input_ids.shape} output={output} output={output.shape}")

        input_ids = torch.cat([input_ids, output], dim=-1)

        if output.item() == tokenizer.eos_token_id:
            break

        # print(output.shape)  # Should be (1, 10, config.vocab_size)

        # print(list(model.state_dict().items())[0], flush=True)

    # logger.info(f"input_ids={input_ids}, input_ids.shape={input_ids.shape}")
    result = input_ids[0].cpu().numpy()
    logger.info(f"result={result}, result_text={tokenizer.decode(result)}")
