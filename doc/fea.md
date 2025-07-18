

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
- 如果申请块的时候token不满，需要后面添加进cache
- 如果申请块token还没有参与计算，但是这一步就参与计算，那么要等参与计算之后再添加进行cache。**vllm是这样做的，感觉这些块添加进cache也行，如果添加，reshape and cache可能会同时写入相同的位置**

## chunked prefill

- 与auto prefix cache结合的时候，如果按照seq_len - computed_len来分配缓存block，下一个相同的req会使用实际没有计算的缓存块
- **当处理同一个批次的相同两个请求时，一个请求有复用另外一个请求没有计算但分配的块时，这两个请求的输出精度不一样**

## scheduler

- 需要为cache manager添加一个can alloc接口，因为调度约束的存在，需要提前判断在满足约束的情况下是否可以分配
- v1版本的调度器中似乎没有cache swap功能了。如果cache不够，将末尾的req占用cache释放，当该req后面重新调度时候，在重新计算token对应的cache
- cache管理对象有两种接口，一种进行资源判断，一种进行资源分配，分配前务必进行判断，因此分配接口中不应该进行多余的检查

## profile run


