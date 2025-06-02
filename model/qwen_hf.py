from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct").cuda().eval()

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_length=30)
res = tokenizer.batch_decode(generated_ids)[0]
print(res)





