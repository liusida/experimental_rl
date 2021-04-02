import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class MultiMlpExtractor(BaseFeaturesExtractor):
    """
    treatment group: one mlp module, with hidden layers [64,64,64]
    """
    def __init__(self, observation_space: gym.Space, m=4):
        self.num_parallel_mlps = m
        super().__init__(observation_space, 64*self.num_parallel_mlps)
        
        n_input = gym.spaces.utils.flatdim(observation_space)
        self.flatten = nn.Flatten()

        self.mlps = nn.ModuleList()
        for i in range(self.num_parallel_mlps):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(n_input, 64),
                    nn.Linear(64, 64),
                    nn.Linear(64, 64)
                )
            )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)

        # branch
        xs = []
        for modules in self.mlps:
            xs.append(modules(x))
        
        # concatenate
        x = th.cat(xs, dim=1)
        return x
