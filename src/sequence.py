from typing import List

from sympy.polys.matrices.dense import ddm_ilu

from src.base import SampleConfig, SchedulerReqRecvMessage


class RequestFactory:
    def __init__(self, config):
        self.id = 0
        self.config = config

    def create(self, tokens, sample_config, step_fun):
        req = Sequence(self.id, tokens, self.config.infer_config.block_size, sample_config, step_fun)
        self.id += 1
        return req


class Sequence:
    def __init__(self, id, tokens, block_size, sample_config: SampleConfig, step_fun=None):
        self.id = id
        self.engine_id = None
        # TODO refactor
        self.tokens = tokens
        self.output_tokens = []
        self.prompt_tokens = [t for t in tokens]
        self.slot_mapping = []
        self.block_table = []

        self.in_seq_len = len(tokens)
        self.out_seq_len = 0
        self.seq_len = self.in_seq_len + self.out_seq_len
        self.computed_len = 0
        self.cached_len = 0

        self.sample_config = sample_config
        self.block_size = block_size
        self._step_fun = step_fun

    @classmethod
    def from_message(cls, msg: SchedulerReqRecvMessage.RequestInputInfo, block_size):
        # TODO block_size
        return cls(
            id=msg.request_id,
            tokens=msg.prompt_tokens,
            sample_config=msg.sample_config,
            block_size=block_size,
        )

    @property
    def allocated_len(self):
        return self.block_size * len(self.block_table)

    @property
    def allocated_block_num(self):
        return len(self.block_table)

    def get_block_content(self, block_idx):
        assert 0 <= block_idx * self.block_size < self.seq_len
        start_token_idx = block_idx * self.block_size
        return self.tokens[start_token_idx: start_token_idx + self.block_size]

    def append_block(self, block_id, is_computed, is_cached):
        # NOTE: when allocated no cached block, when no cached block has full tokens, should cache this block
        if block_id is not None:
            self.block_table.append(block_id)
            start_slot_id = block_id * self.block_size
            self.slot_mapping.extend(list(range(start_slot_id, start_slot_id + self.block_size)))
        if is_computed:
            self.computed_len += self.block_size
        if is_cached:
            self.cached_len += self.block_size

    def append_token(self, computed_token_num, next_token):
        # NOTE: when chunk prefill, some step not append token, but increase computed len
        self.computed_len += computed_token_num
        if next_token is not None:
            self.tokens.append(next_token)
            self.output_tokens.append(next_token)
            self.out_seq_len += 1
            self.seq_len += 1

    def step_fun(self, engine):
        if self._step_fun is not None:
            self._step_fun(self, engine)

    def is_finish(self):
        if self.out_seq_len >= self.sample_config.max_tokens:
            return True
        return False
