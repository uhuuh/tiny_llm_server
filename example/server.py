import queue
from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class ServerConfig:
    model_name: str
    block_size: int

@dataclass
class SampleConfig:
    temperature: float
    top_p: float
    top_k: int
    n: int
    use_beam: bool
    stop_id_list: list[int]
    max_new_tokens: int

@dataclass
class Seqence:
    id: int
    prompt: str
    sample_config: SampleConfig

class IdCounter:
    def __init__(self):
        self.id = 0
    def next(self):
        ret = self.id
        self.id += 1
        return ret

class Server:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.seq_id_counter = IdCounter()
        self.wait_queue = deque()

    def add_request(self, prompt: str, sample_config: SampleConfig):
        # TODO convert str to token_id ???
        self.wait_queue.append(Seqence(self.seq_id_counter.next(), prompt, sample_config))

    def get_wait_queue(self):
        return self.wait_queue

class Scheduler:
    def __init__(self, config: ServerConfig):
        self.config = config


