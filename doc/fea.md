

## auto prefix cache

- prefill阶段时，复用相同请求的前缀cache，减少qkv proj计算量
- prefill所有块中，必须留最后一个块用于计算下一个token
- req完成块不需要立刻释放，以便进行cache的复用
- TODO: 部分分配请求，分配的时候块不完整，无法及时缓存

