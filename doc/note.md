
## roadmap

- [x] qwen 2h, 4h
- [x] cache scheduler 1h, 6.5h
- [ ] http server
- [ ] auto prefix cache 8h, 6h
- [ ] cache swap
- [ ] distributed infer
- [ ] torch compile and cuda graph
- [ ] chunk prefill
- [ ] predictive decoding
- [ ] struct output

## acc compare
- cpu and cuda 
- stack and cat

## attention input layout
- TH
- **TND**
- BSH
- BSND
- BNSD

## tensor splice
```python
# fail
prefill_logits = output[:len(prefill_input_ids)][prefill_cu_seqlens_q[1: ].long() - 1: ]
# ok
prefill_logits = output[:len(prefill_input_ids)][prefill_cu_seqlens_q[1: ].long() - 1] 
```
