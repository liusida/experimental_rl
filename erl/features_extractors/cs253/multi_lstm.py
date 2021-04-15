from contextlib import contextmanager
from enum import Enum
import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class ModuleStatus(Enum):
    ROLLOUT = 0
    TRAINING = 1
    TESTING = 2

class MultiLSTMExtractor(BaseFeaturesExtractor):
    """
    Parallel Multiple LSTM Features Extractor
    input -+--------+--> output
           |        |
           + LSTM_1 +
           + LSTM_2 +
           + LSTM_3 +
           + LSTM_4 +

    RNN needs the back-propagation through time, and it breaks the i.i.d. assumption, so many things need to be done.
    It is not supported in sb3.

    """

    def __init__(self, observation_space: gym.Space, m=2, num_envs=2, include_input=False):
        """
        m: number of parallel mlps, need to be power of 2.
        The final result will always be of size 64 plus the original input size `n_input`.
        """
        self.current_status = ModuleStatus.ROLLOUT # possible values 0: rollout, 1: training, 2: testing. keep 0 by default.
        self.num_envs = num_envs
        self.include_input = include_input
        
        self.final_layer_size = 64  # without n_input
        # check power of 2: https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        assert (m & (m-1) == 0) and m != 0, "m is not power of 2"
        assert m <= self.final_layer_size, "m is too large"

        self.num_parallel_module = m
        self.size_per_module = int(self.final_layer_size / self.num_parallel_module)  # this is why we need m to be power of 2

        n_input = gym.spaces.utils.flatdim(observation_space) if self.include_input else 0
        super().__init__(observation_space, n_input+self.final_layer_size)

        self.flatten = nn.Flatten()

        self.ensembled_modules = nn.ModuleList()
        # hx is for long term memory
        # cx is for short term memory

        self.hx_rollout = th.zeros(self.num_envs, self.num_parallel_module, self.size_per_module)
        self.cx_rollout = th.zeros(self.num_envs, self.num_parallel_module, self.size_per_module)

        self.hx_test = th.zeros(1, self.num_parallel_module, self.size_per_module)
        self.cx_test = th.zeros(1, self.num_parallel_module, self.size_per_module)

        # need to be arrays so we don't partially modify the tensors
        self.hx_manual = [None] * self.num_parallel_module
        self.cx_manual = [None] * self.num_parallel_module

        for i in range(self.num_parallel_module):
            self.ensembled_modules.append(
                nn.LSTMCell(input_size=n_input, hidden_size=self.size_per_module),
            )

    @contextmanager
    def start_training(self, short_hidden_states: th.Tensor, long_hidden_states: th.Tensor) -> None:
        """
        Sida: 
            manually set hidden state using states saved in rollout buffer before forward pass during training,
            set self.is_training to True for forward()
        """
        try:
            for i in range(self.num_parallel_module):
                self.hx_manual[i] = long_hidden_states[:,i].detach().clone()
                self.cx_manual[i] = short_hidden_states[:,i].detach().clone()
            self.current_status = ModuleStatus.TRAINING
            yield
        finally:
            self.current_status = ModuleStatus.ROLLOUT

    @contextmanager
    def start_testing(self):
        """
        Sida:
            set status to testing
        """
        try:
            self.current_status = ModuleStatus.TESTING
            yield
        finally:
            self.current_status = ModuleStatus.ROLLOUT

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)

        # branch
        xs = [x] if self.include_input else []
        for i, modules in enumerate(self.ensembled_modules):
            # TODO:
            # current plan: 1 env indicates testing,
            # 2 envs indicate collecting rollout,
            # 64 envs indicate training.
            if self.current_status==ModuleStatus.ROLLOUT:
                self.hx_rollout[:,i], self.cx_rollout[:,i] = modules(x, (self.hx_rollout[:,i], self.cx_rollout[:,i]))
                xs.append(self.hx_rollout[:,i])
            elif self.current_status==ModuleStatus.TRAINING:
                self.hx_manual[i], self.cx_manual[i] = modules(x, (self.hx_manual[i], self.cx_manual[i]))
                xs.append(self.hx_manual[i])
            elif self.current_status==ModuleStatus.TESTING:
                self.hx_test[:,i], self.cx_test[:,i] = modules(x, (self.hx_test[:,i], self.cx_test[:,i]))
                xs.append(self.hx_test[:,i])
            else:
                raise NotImplementedError
        # concatenate
        x = th.cat(xs, dim=1)

        return x

    def _apply(self, fn):
        """ Override the methods like .to(device), .float(), .cuda(), .cpu(), etc.
        """
        super()._apply(fn)
        self.hx_rollout = fn(self.hx_rollout)
        self.cx_rollout = fn(self.cx_rollout)
        self.hx_test = fn(self.hx_test)
        self.cx_test = fn(self.cx_test)
        if self.hx_manual[0] is not None:
            self.hx_manual = [fn(x) for x in self.hx_manual]
            self.cx_manual = [fn(x) for x in self.cx_manual]

        return self
