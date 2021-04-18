from typing import Any, Dict, List, Optional, Tuple, Type, Union
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

class MultiExtractor(BaseFeaturesExtractor):
    """
    Parallel Multiple Features Extractors
    input -+--------+--> output
           |        |
           + LSTM_1 +
           + LSTM_2 +
           |        |
           + Mlp_1  +
           + Mlp_2  +

    RNN is not supported in sb3 yet.
    """

    def __init__(self, observation_space: gym.Space, num_envs=2, flatten=1, num_rnns=2, num_mlps=2):
        """
        modules: a string from command line arguments, so that we can specify what modules we want to use in command line.
                 f: flatten input (0 no, 1 yes)
                 r: number of rnn modules (integer)
                 m: number of mlp modules (integer)
                 the sum of parallel rnns and mlps need to be power of 2.
        The final result will always be of size 64 plus the original input size `n_input`.
        """
        self.current_status = ModuleStatus.ROLLOUT # possible values 0: rollout, 1: training, 2: testing. keep 0 by default.
        self.num_envs = num_envs
        self.include_input = flatten      # f: flatten input (0 no, 1 yes)
        self.num_parallel_mlps = num_mlps  # r: number of rnn modules (integer)
        self.num_parallel_rnns = num_rnns  # m: number of mlp modules (integer)
        self.num_parallel_sum = num_mlps+num_rnns
        
        # check power of 2: https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        assert (self.num_parallel_sum & (self.num_parallel_sum-1) == 0) or self.num_parallel_sum == 0, "num_parallel_sum is not power of 2"

        self.final_layer_size = 0  # without n_input
        self.size_per_module = 0
        if self.num_parallel_sum:
            self.final_layer_size = 64  # without n_input
            self.size_per_module = int(self.final_layer_size / self.num_parallel_sum)  # this is why we need m to be power of 2
        assert self.num_parallel_sum <= self.final_layer_size, "num_parallel_sum is too large"

        n_input = gym.spaces.utils.flatdim(observation_space)
        if self.include_input:
            super().__init__(observation_space, n_input+self.final_layer_size)
        else:
            super().__init__(observation_space, self.final_layer_size)

        self.flatten = nn.Flatten()

        self.ensembled_rnns = nn.ModuleList()
        self.ensembled_mlps = nn.ModuleList()
        # hx is for long term memory
        # cx is for short term memory
        if self.num_parallel_rnns:
            self.hx_rollout = th.zeros(self.num_envs, self.num_parallel_rnns, self.size_per_module)
            self.cx_rollout = th.zeros(self.num_envs, self.num_parallel_rnns, self.size_per_module)

            self.hx_test = th.zeros(1, self.num_parallel_rnns, self.size_per_module)
            self.cx_test = th.zeros(1, self.num_parallel_rnns, self.size_per_module)

            # need to be arrays so we don't partially modify the tensors
            self.hx_manual = [None] * self.num_parallel_rnns
            self.cx_manual = [None] * self.num_parallel_rnns

            for i in range(self.num_parallel_rnns):
                self.ensembled_rnns.append(
                    nn.LSTMCell(input_size=n_input, hidden_size=self.size_per_module),
                )

        for i in range(self.num_parallel_mlps):
            self.ensembled_mlps.append(
                nn.Linear(n_input, self.size_per_module),
            )

    @contextmanager
    def start_training(self, short_hidden_states: th.Tensor, long_hidden_states: th.Tensor) -> None:
        """
        Sida: 
            manually set hidden state using states saved in rollout buffer before forward pass during training,
            set self.is_training to True for forward()
        """
        try:
            for i in range(self.num_parallel_rnns):
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

    def forward(self, observations: th.Tensor, new_start: Optional[th.Tensor] = None) -> th.Tensor:
        x = self.flatten(observations)

        # branch
        if self.include_input:
            xs = [x]
        else:
            xs = []

        # all rnns
        for i, modules in enumerate(self.ensembled_rnns):
            if self.current_status==ModuleStatus.ROLLOUT:
                if new_start is not None:
                    self.hx_rollout[:,i] = (~new_start).unsqueeze(1) * self.hx_rollout[:,i]
                    self.cx_rollout[:,i] = (~new_start).unsqueeze(1) * self.cx_rollout[:,i]
                self.hx_rollout[:,i], self.cx_rollout[:,i] = modules(x, (self.hx_rollout[:,i], self.cx_rollout[:,i]))
                xs.append(self.hx_rollout[:,i])
            elif self.current_status==ModuleStatus.TRAINING:
                if new_start is not None:
                    self.hx_manual[i] = (~new_start).unsqueeze(1) * self.hx_manual[i]
                    self.cx_manual[i] = (~new_start).unsqueeze(1) * self.cx_manual[i]
                self.hx_manual[i], self.cx_manual[i] = modules(x, (self.hx_manual[i], self.cx_manual[i]))
                xs.append(self.hx_manual[i])
            elif self.current_status==ModuleStatus.TESTING:
                if new_start is not None:
                    self.hx_test[:,i] = (~new_start).unsqueeze(1) * self.hx_test[:,i]
                    self.cx_test[:,i] = (~new_start).unsqueeze(1) * self.cx_test[:,i]
                self.hx_test[:,i], self.cx_test[:,i] = modules(x, (self.hx_test[:,i], self.cx_test[:,i]))
                xs.append(self.hx_test[:,i])
            else:
                raise NotImplementedError
        # all mlps
        for i, modules in enumerate(self.ensembled_mlps):
            x = modules(x)
            xs.append(x)
            
        # concatenate
        x = th.cat(xs, dim=1)

        return x

    def _apply(self, fn):
        """ Override the methods like .to(device), .float(), .cuda(), .cpu(), etc.
        """
        super()._apply(fn)
        if self.num_parallel_rnns:
            self.hx_rollout = fn(self.hx_rollout)
            self.cx_rollout = fn(self.cx_rollout)
            self.hx_test = fn(self.hx_test)
            self.cx_test = fn(self.cx_test)
            if self.hx_manual[0] is not None:
                self.hx_manual = [fn(x) for x in self.hx_manual]
                self.cx_manual = [fn(x) for x in self.cx_manual]

        return self
