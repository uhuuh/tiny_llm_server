

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

cache 分配
- 在请求prefill和decode阶段都需要事先为请求分配cache块来满足其attn计算, 其中prefill必须分配新块, decode有可能分配新块, 因为有些时候decode复用之前的cache就足够了
- schedule维护一个空闲块表, req维护一个占用块表, 每次调度时按照req在queue中的顺序和req自身状态分配cache, 具体有以下情况, 其中run queue指哪些占用了cache的req, 而wait queue指哪些没有占用cache的req, scheduler按照req到达顺序优先处理, 当然中间的req可能提前就生成完token返回上层
 
- run queue和wait queue都存在请求, 优先处理run queue, 又分为以下多种情况
-- 剩余的块不能满足run queue中任意req的需要, 这种情况强制释放run queue中最后一个req占用的cache(使用v1中的重计算策略, 将该req释放cache, 挪至wait queue头部, 而不是v0中的offload cpu策略), 要不然造成一种死锁状态, 所有req都无法继续处理
-- 剩余的块仅仅可以满足run queue前面部分req的需要
-- 剩余的块可以满足run queue全部的req的需要
-- 剩余的块可以满足run queue全部req的需要, 另外可以满足一些wait queue中req的需要, wait queue中req如果分配了cache将其加入到run queue中
- 仅wait queue中存在请求, 按照顺序为req分配cache 块直到块无剩余

- 实际上进行cache分配的时候, 还需要考虑max req num和max batch token num 这两个因素. 因此可能后面的一些req可以分配cache导致因为这两个限制而没有分配, 退回到wait queue中

## profile run

- 把cache storage从worker提升到了scheduler中，放在在warm up中细粒度的控制cache的申请和释放

## http

- 协程是用户态线程，需要在用户态实现上下文切换和事件循环和监听机制来实现协程
- 协程应该有线程的一些机制，同步机制(future, event, queue)，创建和等待新协程(task)，调用子协程(await async fun)
- torch分配的tensor，必须在同一个进程同一个线程才可以使用，同一个进程的不同的线程不行

- enum对象如何跨进程通信，作为dict的key会失效
- 实例方法作为回调使用时，默认捕获self为参数




