import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


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

    def __init__(self, observation_space: gym.Space, m=4):
        """
        m: number of parallel mlps, need to be power of 2.
        The final result will always be of size 64 plus the original input size `n_input`.
        """
        self.final_layer_size = 64  # without n_input
        # check power of 2: https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        assert (m & (m-1) == 0) and m != 0, "m is not power of 2"
        assert m <= self.final_layer_size, "m is too large"

        self.num_parallel_module = m
        self.size_per_module = int(self.final_layer_size / self.num_parallel_module)  # this is why we need m to be power of 2

        n_input = gym.spaces.utils.flatdim(observation_space)
        super().__init__(observation_space, n_input+self.final_layer_size)

        self.flatten = nn.Flatten()

        self.ensembled_modules = nn.ModuleList()
        # hx is for long term memory
        # cx is for short term memory

        # TODO: 2 is the number of environments
        self.hx_rollout = th.randn(2, self.num_parallel_module, self.size_per_module)
        self.cx_rollout = th.randn(2, self.num_parallel_module, self.size_per_module)

        self.hx_test = th.randn(1, self.num_parallel_module, self.size_per_module)
        self.cx_test = th.randn(1, self.num_parallel_module, self.size_per_module)

        # TODO: 64 is the training batch size
        self.hx_manual = th.randn(64, self.num_parallel_module, self.size_per_module)
        self.cx_manual = th.randn(64, self.num_parallel_module, self.size_per_module)

        for i in range(self.num_parallel_module):
            self.ensembled_modules.append(
                nn.LSTMCell(input_size=n_input, hidden_size=self.size_per_module),
            )

    def manually_set_hidden_state(self, short_hidden_states: th.Tensor, long_hidden_states: th.Tensor) -> None:
        """
        Sida: manually set hidden state using states saved in rollout buffer before forward pass during training
        """
        self.hx_manual = long_hidden_states
        self.cx_manual = short_hidden_states

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)

        # branch
        xs = [x]
        for i, modules in enumerate(self.ensembled_modules):
            # TODO:
            # current plan: 1 env indicates testing,
            # 2 envs indicate collecting rollout,
            # 64 envs indicate training.
            if x.shape[0] == 1:
                self.hx_test[:,i], self.cx_test[:,i] = modules(x, (self.hx_test[:,i], self.cx_test[:,i]))
                xs.append(self.hx_test[:,i])
            elif x.shape[0] == 2:
                self.hx_rollout[:,i], self.cx_rollout[:,i] = modules(x, (self.hx_rollout[:,i], self.cx_rollout[:,i]))
                xs.append(self.hx_rollout[:,i])
            elif x.shape[0] == 64:
                self.hx_manual[:,i], self.cx_manual[:,i] = modules(x, (self.hx_manual[:,i], self.cx_manual[:,i]))
                xs.append(self.hx_manual[:,i])

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
        self.hx_manual = fn(self.hx_manual)
        self.cx_manual = fn(self.cx_manual)

        return self
