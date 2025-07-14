import os
os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
import torch

@triton.jit
def paged_attention_kernel(
    out_ptr, # [seq_len, head_num, head_size]
    q_ptr, # [seq_len, head_num, head_size]
    qk_ptr, # [max_content_len, head_num]
    key_cache, # [block_num, head_num, head_size, block_size]
    value_cache, # [block_num, head_num, head_size, block_size]
    block_table, # [seq_len, max_block_num_in_seq]
    context_len, # [seq_len]
    block_size: tl.constexpr,
    head_num: tl.constexpr,
    head_size: tl.constexpr,
    max_content_len: tl.constexpr
):
    # tl.constexpr 不一定真的是编译常量，因为jit是运行时编译
    # TODO 常量值改变会重新编译吗？带来编译开销吗
    # TODO triton与torch.compile配合是什么意思
    # TODO 这个kernel还有很大的缺陷，一些输入要求是2的指数
    # TODO 如何使用ncu来优化triton kernel
    # TODO 可以在内部申请一个tensor，然后写入该tensor吗。用where方法似乎可行，但是只有这一种方法可以吗
    # TODO 接下来的一种改进是提取所有shape变量

    seq_id = tl.program_id(0)
    token_num_in_seq = tl.load(context_len + seq_id)

    q_offset = seq_id * head_num * head_size + tl.arange(0, head_num * head_size)
    q = tl.load(q_ptr + q_offset).reshape((head_num, head_size))
    # 动态申请的shape也必须是常量和2的指数
    for token_id in range(0, token_num_in_seq, block_size):
        block_table_offset = seq_id + token_id // block_size
        block_id = tl.load(block_table + block_table_offset)
        block_offset = block_id + tl.arange(0, head_num * head_size * block_size)
        k_tile = tl.load(key_cache + block_offset).reshape((head_num, head_size, block_size))

        # 新增一个维度为1的维度，然后在这个新增维度上广播，来实现repeat类似的功能
        lhs = tl.broadcast_to(q.reshape(head_num, 1, head_size), (head_num, block_size, head_size)).reshape(block_size * head_num, 1, head_size)
        rhs = k_tile.permute(2, 1, 0).reshape(block_size * head_num, head_size, 1)
        # tl.device_print("-------> lhs", lhs.shape)
        # tl.device_print("------> rhs", rhs.shape)
        qk_tile = tl.dot(lhs, rhs).reshape((block_size, head_num))
        # tl.device_print(f"------> qk_tile {qk_tile.shape}")

        qk_tile_offset = token_id * block_size * head_num + tl.arange(0, block_size * head_num).reshape((block_size, head_num))
        tl.store(qk_ptr + qk_tile_offset, qk_tile)
    
    qk_offset = tl.arange(0, max_content_len * head_num).reshape((max_content_len, head_num))
    qk_val = tl.load(qk_ptr + qk_offset)
    # 这种静态图方式的写法，应该适合用静态语言来写
    # 内部tensor使用索引方式复制 ---> AssertionError()
    # tensor.max使用。只有少部分方法采用后缀方式调用。尽量所有方式都使用前缀方式调用 ---> RuntimeError: Cannot call @triton.jit'd outside of the scope of a kernel
    # tl.store传入内部tensor ---> AttributeError("'pointer_type' object has no attribute 'primitive_bitwidth'")
    # dot不支持广播 ---> AssertionError: First input shape (['constexpr[16]', 'constexpr[16]', 'constexpr[1]']) and second input shape ['constexpr[16]', 'constexpr[64]', 'constexpr[16]'] are not compatible for matmul
    # 广播的维度必须为1 ---> triton.runtime.errors.InterpreterError: ValueError("Cannot broadcast, the expanded size of the tensor (256) must match the existing size (16) at non-singleton dimension 0: ['16', '1', '64'], [256, 1, 64]")
    # device_print里面每一项必须是str或者tensor ---> triton.runtime.errors.InterpreterError: TypeError("cannot convert ['constexpr[16]', 'constexpr[16]'] of type <class 'triton.language.core.tuple'> to tensor")
    qk_val = qk_val * tl.rsqrt(head_size)
    qk_max = tl.max(qk_val, axis=0)
    qk_sum = tl.sum(qk_val, axis=0)
    qk_val = (qk_val - qk_max) / qk_sum
    
    qkv = tl.zeros((head_num, head_size), dtype=tl.float32)
    for token_id in range(0, token_num_in_seq, block_size):
        block_table_offset = seq_id + token_id // block_size
        block_id = tl.load(block_table + block_table_offset)
        block_offset = block_id + tl.arange(0, head_num * head_size * block_size).reshape((head_num, head_size, block_size))
        v_tile = tl.load(value_cache + block_offset)
    
        qk_tile_offset = token_id * block_size * head_num + tl.arange(0, block_size * head_num).reshape((block_size, head_num))
        qk_tile = tl.load(qk_ptr + qk_tile_offset)

        # 好像只有load和store才支持广播
        qkv_tile = qk_tile[:, :, None].broadcast_to(block_size, head_num, head_size) * v_tile.permute((2, 0, 1))
        qkv += tl.sum(qkv_tile, axis=0)

    out_offset = seq_id * head_num * head_size + tl.arange(0, head_num * head_size).reshape((head_num, head_size))
    tl.store(out_ptr + out_offset, qkv)

def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: list[list],
    context_len_table: list[int],
    max_content_len: int,
):
    seq_len, head_num, head_size = query.shape
    _, _, _, block_size = key_cache.shape
    output = torch.empty_like(query)
    qk = torch.empty((max_content_len, head_num), dtype=query.dtype)
    # 可以输入list或者dict吗 ---> 不行
    paged_attention_kernel[(seq_len, )](
        output, query, qk, key_cache, value_cache, block_table, context_len_table, block_size, head_num, head_size, max_content_len)
    return output
    
    
if __name__ == "__main__":
    # Test the paged_attention function
    seq_len = 128
    head_num = 16
    head_size = 64
    block_size = 16
    block_num = 100
    max_content_len = 16

    query = torch.randn(seq_len, head_num, head_size)
    key_cache = torch.randn(block_num, head_num, head_size, block_size)
    value_cache = torch.randn(block_num, head_num, head_size, block_size)
    block_table = torch.randint(0, block_num, (seq_len, (max_content_len + block_size - 1) // block_size))
    context_len_table = torch.randint(1, max_content_len, (seq_len,))
    print("----------->", query.dtype)
    
    output = paged_attention(query, key_cache, value_cache, block_table, context_len_table, max_content_len)
    print(output.shape) # Should be (128, 12, 64)

