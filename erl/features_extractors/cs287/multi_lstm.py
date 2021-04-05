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
    """
    def __init__(self, observation_space: gym.Space, m=4):
        """
        m: number of parallel mlps, need to be power of 2.
        The final result will always be of size 64 plus the original input size `n_input`.
        """
        self.final_layer_size = 64 # without n_input
        # check power of 2: https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        assert (m & (m-1) == 0) and m != 0, "m is not power of 2"
        assert m<=self.final_layer_size, "m is too large"

        self.num_parallel_module = m
        self.size_per_module = int(self.final_layer_size / self.num_parallel_module) # this is why we need m to be power of 2

        n_input = gym.spaces.utils.flatdim(observation_space)
        super().__init__(observation_space, n_input+self.final_layer_size)
        
        self.flatten = nn.Flatten()

        self.ensembled_modules = nn.ModuleList()
        self.hx, self.cx = [], []

        for i in range(self.num_parallel_module):
            self.ensembled_modules.append(
                nn.LSTMCell(input_size=n_input, hidden_size=self.size_per_module),
            )
            self.hx.append(th.randn(4, self.size_per_module))
            self.cx.append(th.randn(4, self.size_per_module))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)

        # branch
        xs = [x]
        for i, modules in enumerate(self.ensembled_modules):
            self.hx[i], self.cx[i] = modules(x, (self.hx[i], self.cx[i]))
            xs.append(self.hx[i])
        
        # concatenate
        x = th.cat(xs, dim=1)

        return x

    def _apply(self, fn):
        """ Override the methods like .to(device), .float(), .cuda(), .cpu(), etc.
        """
        super()._apply(fn)
        self.hx = [fn(x) for x in self.hx]
        self.cx = [fn(x) for x in self.cx]
        return self