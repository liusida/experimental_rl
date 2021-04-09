import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class MultiMlpExtractor(BaseFeaturesExtractor):
    """
    Parallel Multiple Multi-Layer Perceptrons Features Extractor
    input -+-------+--> output
           |       |
           + Mlp_1 +
           + Mlp_2 +
           + Mlp_3 +
           + Mlp_4 +
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

        self.num_parallel_mlps = m
        self.size_per_mlps = int(self.final_layer_size / self.num_parallel_mlps) # this is why we need m to be power of 2

        n_input = gym.spaces.utils.flatdim(observation_space)
        super().__init__(observation_space, n_input+self.final_layer_size)
        
        self.flatten = nn.Flatten()

        self.mlps = nn.ModuleList()
        for i in range(self.num_parallel_mlps):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(n_input, 64),
                    nn.Linear(64, 64),
                    nn.Linear(64, self.size_per_mlps)
                )
            )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)

        # branch
        xs = [x]
        for modules in self.mlps:
            xs.append(modules(x))
        
        # concatenate
        x = th.cat(xs, dim=1)

        return x
