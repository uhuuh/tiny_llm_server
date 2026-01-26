import hashlib
import multiprocessing
import threading

import hashlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Any

import torch

from src.base import InferConfig, Config
from contextlib import contextmanager


def hash_tensor(tensor):
    """
    计算tensor的hash值
    """
    # 将tensor转换为numpy数组并转换为字节流
    tensor_bytes = tensor.cpu().detach().float().numpy().tobytes()
    # 使用SHA-256算法计算哈希值
    sha256 = hashlib.sha256()
    sha256.update(tensor_bytes)
    h = sha256.hexdigest()[:8]
    return [h, tensor.shape, tensor.dtype, tensor.device]

def hook(module_name, module, input, output):
    """
    模块计算后钩子函数，打印模块完整名字、输入tensor的hash和输出tensor的hash
    """
    input_hash = [hash_tensor(x) if isinstance(x, torch.Tensor) else None for x in input]
    output_hash = hash_tensor(output) if isinstance(output, torch.Tensor) else None
    print(f"model={module_name} input={input_hash} output={output_hash}")

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


class ProcessExecutor:
    def __init__(self, cls, cls_args=None, cls_kwargs=None, io_methods=None):
        """
        :param cls: 要在子进程中执行的类
        :param cls_args: 类构造参数 (tuple / list)
        :param io_methods: 类中需要作为 IO 线程执行的方法名列表
        """
        self.cls = cls
        self.cls_args = cls_args or ()
        self.cls_kwargs = cls_kwargs or {}
        self.io_methods = io_methods or []
        self.process = None

    def _process_main(self):
        """子进程入口"""
        # 构造类实例
        obj = self.cls(*self.cls_args, **self.cls_kwargs)

        threads = []

        # 启动 IO 线程
        for method_name in self.io_methods:
            method = getattr(obj, method_name)
            t = threading.Thread(target=method, daemon=True)
            t.start()
            threads.append(t)

        # 等待所有 IO 线程结束
        for t in threads:
            t.join()

    def start(self):
        """启动子进程"""
        self.process = multiprocessing.Process(
            target=self._process_main
        )
        self.process.start()

    def wait(self):
        """阻塞等待子进程退出"""
        if self.process:
            self.process.join()

def get_config():
    infer_config = InferConfig(
        model_path="/mnt/c/Users/uh/code/ckpt/Qwen2.5-0.5B-Instruct",
        max_req_num=256,
        max_batch_token_num=4096,
        gpu_memory_utilization=0.5,
        block_size=16,
        enable_prefix_cache=True,
        enable_debug=True,
        max_prefill_len=32,
        enable_chunked_prefill=True,
        data_parallel=1,
    )
    config = Config(
        infer_config=infer_config,
    )
    return config


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class ForwardContext:
    attn_meta: Any
    layer_idx: int = 0

_FORWARD_CONTEXT = None

def get_forward_context():
    ctx = _FORWARD_CONTEXT
    if ctx is None:
        raise RuntimeError("Forward context is not set. Use set_forward_context() before forward pass.")
    return ctx

@contextmanager
def set_forward_context(ctx: ForwardContext):
    global _FORWARD_CONTEXT
    prev = _FORWARD_CONTEXT
    _FORWARD_CONTEXT = ctx
    yield
    _FORWARD_CONTEXT = prev
