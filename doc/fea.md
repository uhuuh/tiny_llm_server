

## add model

- 相同的算子，处理cpu和gpu输入的时候，精度不一样
- stack和cat算子，精度不一样

### attention input layout
- TH
- **TND**
- BSH
- BSND
- BNSD

### tensor splice
```python
# fail
prefill_logits = output[:len(prefill_input_ids)][prefill_cu_seqlens_q[1: ].long() - 1: ]
# ok
prefill_logits = output[:len(prefill_input_ids)][prefill_cu_seqlens_q[1: ].long() - 1] 
```

## auto prefix cache

- prefill阶段时，复用相同请求的前缀cache，减少qkv proj计算量
- prefill所有块中，必须留最后一个块用于计算下一个token
- req完成块不需要立刻释放，以便进行cache的复用
- TODO: 部分分配请求，分配的时候块不完整，无法及时缓存

## chunked prefill

- 与auto prefix cache结合的时候，如果按照seq_len - computed_len来分配缓存block，下一个相同的req会使用实际没有计算的缓存块
- **当处理同一个批次的相同两个请求时，一个请求有复用另外一个请求没有计算但分配的块时，这两个请求的输出精度不一样**

## profile run

- cpu块数量是如何确定的

