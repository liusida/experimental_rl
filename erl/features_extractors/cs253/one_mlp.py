import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class OneMlpExtractor(BaseFeaturesExtractor):
    """
    treatment group: one mlp module, with hidden layers [64,64,64]
    """
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, 64)
        
        n_input = gym.spaces.utils.flatdim(observation_space)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
