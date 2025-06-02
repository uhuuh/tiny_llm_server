import os
os.environ['TRITON_INTERPRET'] = '1'

import triton
import triton.language as tl
import torch


'''
Q [seq_len, hidden_dim]
K [seq_len, hidden_dim]
V [seq_len, hidden_dim]
O [seq_len, hidden_dim]  zeros
M [seq_len] -inf
L [seq_len] zeros
step

for y in range(0, seq_len, step)
    k = K[y:y+step, :]
    v = V[y:y+step, :]
    for x in range(0, seq_len, step)
        q = Q[x:x+step, :]
        o = O[x:x+step, :]

        m = M[x:x+step, :]
        l = L[x:x+step, :]
        score = q @ k^T # [step, step]
        next_m = maxium(m, qk.max(axis=1))
        p = (score - next_m).exp()
        m_scale = (m - next_m).exp()
        next_l = l * m_scale  + p.sum(axis=1)

        # 在实际的flash atte中，下面是如何实现的呢？一种是broadcast， 另外一种是向量转化为对角矩阵
        o = o * l^-1 * next_l * m_scale + p / next_l @ v
        m = next_m
        l = next_l
# TODO 还要考虑mask

'''

@triton.jit
def flash_attention_kernel(
    out_ptr,
    q_ptr, 
    k_ptr,
    v_ptr,
    m_ptr,
    l_ptr,
    seq_len,
    head_num: tl.constexpr,
    head_size: tl.constexpr,
    step: tl.constexpr,
):
    # 如果在y索引上切分为不同block的话，同一行之间如何同步呢？
    # 如果按照y维度上切分为不同block的话，那么v1版本所提倡的qk在外循环的益处还存在吗
    # TODO 还需要处理head_num
    x = tl.program_id(0)
    tile_shape = tl.arange(0, head_num * head_size).reshape(head_num, head_size)
    # broadcast_vec = lambda vec: tl.broadcast_to(tl.reshape(vec, (step, 1)), (step, step))
    
    q = tl.load(q_ptr + x * head_num * head_size + tile_shape)
    o = tl.load(out_ptr + x * head_num * head_size + tile_shape)
    l = tl.load(l_ptr + x * step + tile_shape)
    m = tl.load(m_ptr + x * step + tile_shape)
    for y in range(0, seq_len, step):
        k = tl.load(k_ptr + y * head_num * head_size + tile_shape)
        v = tl.load(v_ptr + y * head_num * head_size + tile_shape)

        qk = tl.dot(q, k.T) * tl.rsqrt(head_size)
        m_next = tl.max(qk, axis=1)
        score_upper = (qk - m_next).exp()
        m_scale = (m - m_next).exscore_upper()
        l_next = l * m_scale + score_upper.sum(axis=1)

        o_next = o * broadcast_vec(l ** -1) * broadcast_vec(l_next) * broadcast_vec(m_scale) + tl.dot(score_upper / l_next ,v)

        o = o_next
        m = m_next
        l = l_next

    tl.store(out_ptr + x * head_num * head_size + tile_shape, o)
    tl.store(m_ptr + x * head_num * head_size + tile_shape, m)
    tl.store(l_ptr + x * head_num * head_size + tile_shape, l)


    
    
def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
):
    seq_len, head_num, head_size = Q.shape
    step = 1
    output = torch.empty_like(Q)
    m = torch.full((seq_len,), dtype=Q.dtype, fill_value=float('-inf'))
    l = torch.zeros((seq_len,), dtype=Q.dtype)
    flash_attention_kernel[(seq_len, )](
        output, 
        Q, 
        K, 
        V, 
        m, 
        l,
        seq_len, 
        head_num, 
        head_size, 
        step
    )
    return output
  
if __name__ == "__main__":
    # Test the flash_attention function
    seq_len = 128
    head_num = 16
    head_size = 64

    Q = torch.randn(seq_len, head_num, head_size).cuda()
    K = torch.randn(seq_len, head_num, head_size).cuda()
    V = torch.randn(seq_len, head_num, head_size).cuda()

    output = flash_attention(Q, K, V)
    print(output.shape) # Should be (128, 12, 64)
    
