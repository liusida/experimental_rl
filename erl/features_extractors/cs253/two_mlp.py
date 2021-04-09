import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class TwoMlpExtractor(BaseFeaturesExtractor):
    """
    treatment group: one mlp module, with hidden layers [64,64,64]
    """
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, 64)
        
        n_input = gym.spaces.utils.flatdim(observation_space)
        self.flatten = nn.Flatten()
        self.mlp1_fc1 = nn.Linear(n_input, 64)
        self.mlp1_fc2 = nn.Linear(64, 64)
        self.mlp1_fc3 = nn.Linear(64, 32)

        self.mlp2_fc1 = nn.Linear(n_input, 64)
        self.mlp2_fc2 = nn.Linear(64, 64)
        self.mlp2_fc3 = nn.Linear(64, 32)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)
        # mlp 1
        x1 = F.relu(self.mlp1_fc1(x))
        x1 = F.relu(self.mlp1_fc2(x1))
        x1 = F.relu(self.mlp1_fc3(x1))
        # mlp 2
        x2 = F.relu(self.mlp2_fc1(x))
        x2 = F.relu(self.mlp2_fc2(x2))
        x2 = F.relu(self.mlp2_fc3(x2))
        # concatenate
        x = th.cat([x1,x2], dim=1)
        return x
