from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.generation import GenerationConfig
from util import register_hooks, print_model_parameter_weights_hash

set_seed(32)

model_path = "/mnt/c/Users/uh/code/ckpt/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()

model_inputs = tokenizer(["What are we having for dinner?"], return_tensors="pt").to("cuda")
print(">>> model_inputs", model_inputs)

print(model)
# print_model_parameter_weights_hash(model)
# register_hooks(model)
print(list(model.state_dict().items())[0], flush=True)

# 配置生成参数，控制只生成一个token，采样温度为0
generation_config = GenerationConfig(
    max_new_tokens=500, # 控制只生成一个token
    temperature=0  # 采样温度设置为0
)

generated_ids = model.generate(**model_inputs, generation_config=generation_config)
print(">>> generated_ids", generated_ids)

res = tokenizer.batch_decode(generated_ids)
print(res)





