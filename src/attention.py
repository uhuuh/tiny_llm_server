from typing import List

import torch
from torch import nn as nn
from vllm._custom_ops import reshape_and_cache_flash
from vllm.vllm_flash_attn import flash_attn_varlen_func

from src.base import WorkerInput
from src.utils import get_forward_context
from dataclasses import dataclass

@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlen_q: torch.Tensor
    max_seqlen_q: torch.Tensor
    seqlen_k: torch.Tensor
    max_seqlen_k: torch.Tensor
    k_cache: List[torch.Tensor]
    v_cache: List[torch.Tensor]

    def __str__(self) -> str:
        res = [f"{self.__class__.__name__}:"]
        for field_name, value in self.__dict__.items():
            # 处理 KV Cache (List[torch.Tensor])
            if field_name in ("k_cache", "v_cache") and isinstance(value, list):
                shapes = [list(t.shape) for t in value]
                res.append(f"  {field_name}: List of {len(value)} tensors, shapes={shapes}")
            elif isinstance(value, torch.Tensor):
                res.append(f"  {field_name}: shape={value.shape}, data={value}")
            else:
                res.append(f"  {field_name}: {value}")

        return "\n".join(res)

class FlashAttentionBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_cache_dtype = "auto"
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    def forward(self, q, k, v):
        ctx = get_forward_context()
        attn_meta: AttentionMetadata = ctx.attn_meta

        reshape_and_cache_flash(
            key=k,
            value=v,
            key_cache=attn_meta.k_cache[ctx.layer_idx],
            value_cache=attn_meta.v_cache[ctx.layer_idx],
            slot_mapping=attn_meta.slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=self._k_scale,
            v_scale=self._v_scale,
        )

        o = torch.zeros_like(q)
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=attn_meta.cu_seqlen_q,
            max_seqlen_q=attn_meta.max_seqlen_q,
            seqused_k=attn_meta.seqlen_k,
            max_seqlen_k=attn_meta.max_seqlen_k,
            causal=True,
            out=o,
        )
        return o
