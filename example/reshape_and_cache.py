import triton
import triton.language as tl
import torch

@triton.jit
def reshape_and_cache_kernel(
    key_ptr, # [token_num, head_num, head_size]
    value_ptr, # [token_num, head_num, head_size]
    key_cache_ptr, # [block_num, head_num, head_size, block_size]
    value_cache_ptr, # [block_num, head_num, head_size, block_size]
    slot_mapping, # [token_num]
    block_num,
    block_size,
    token_num,
    head_num,
    head_size,
):
    pid = tl.program_id(0)
    mask = tl.full((head_size, ), dtype=torch.bool, value=True)
    # TODO 如果不指定形状， load加载什么shape的tensor

    key_offset = key_ptr + pid * token_num * head_num * head_size
    value_offset = value_ptr + pid * token_num * head_num * head_size
    key_data = tl.load(key_ptr + key_offset, mask=mask)
    value_data = tl.load(value_ptr + value_offset, mask=mask)
    key_cache_offset = pid * head_num * head_size * block_size + slot_mapping[pid] % block_num * head_size
    value_cache_offset = pid * head_num * head_size * block_size + slot_mapping[pid] % block_num * head_size
    tl.store(key_cache_ptr + key_cache_offset, key_data, mask=mask)
    tl.store(value_cache_ptr + value_cache_offset, value_data, mask=mask)

def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: list[int]
):
    token_num, head_num, head_size = key.shape
    block_num, _, _, block_size = key_cache.shape
    return reshape_and_cache_kernel[token_num](
        key, value, key_cache, value_cache, slot_mapping, block_num, block_size, token_num, head_num, head_size)







    


