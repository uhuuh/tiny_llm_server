import hashlib
import torch
import hashlib
import torch

def hash_tensor(tensor):
    """
    计算tensor的hash值
    """
    # 将tensor转换为numpy数组并转换为字节流
    tensor_bytes = tensor.cpu().detach().numpy().tobytes()
    # 使用SHA-256算法计算哈希值
    sha256 = hashlib.sha256()
    sha256.update(tensor_bytes)
    return sha256.hexdigest()[:8]

def hook(module_name, module, input, output):
    """
    模块计算后钩子函数，打印模块完整名字、输入tensor的hash和输出tensor的hash
    """
    input_hash = [hash_tensor(x) if isinstance(x, torch.Tensor) else None for x in input]
    output_hash = hash_tensor(output) if isinstance(output, torch.Tensor) else None
    print(f"模块完整名: {module_name}, 输入tensor的hash: {input_hash}, 输出tensor的hash: {output_hash}")

def register_hooks(model):
    """
    注册模块计算后钩子到模型的所有子模块，并显示完整层级名
    """
    # 直接使用生成器表达式获取所有子模块，避免定义额外函数
    submodules = list(model.named_modules())
    for module_name, module in submodules:
        module.register_forward_hook(lambda m, i, o, name=module_name: hook(name, m, i, o))

def print_model_parameter_weights_hash(model):
    """
    打印模型所有参数的完整层名、权重的哈希值
    """
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            param_hash = hash_tensor(param)
            print(f"完整层名: {full_param_name}, 参数权重的hash: {param_hash}")
