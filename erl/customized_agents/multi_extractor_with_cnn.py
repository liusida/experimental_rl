from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from .multi_extractor import MultiExtractor


class MultiExtractorWithCNN(MultiExtractor):
    def __init__(self, observation_space: gym.Space, num_envs=2, flatten=1, num_rnns=2, num_mlps=2, rnn_layer_size=16):
        dim = gym.spaces.utils.flatdim(observation_space)
        dim -= 3*8*8 # minus image
        dim += 16 # plus image features
        new_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))
        super().__init__(
            observation_space=new_observation_space, # change the dim
            num_envs=num_envs,
            flatten=flatten,
            num_rnns=num_rnns,
            num_mlps=num_mlps,
            rnn_layer_size=rnn_layer_size
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor, new_start: Optional[th.Tensor]) -> th.Tensor:
        # obs = 10 + joints + 3x8x8
        print(observations.shape)
        x0 = observations[:, :-3*8*8]
        x = observations[:, -3*8*8:].reshape([-1,3,8,8])
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        observations = th.cat([x0, self.flatten(x)], dim=1)
        print(x.shape)
        print("")
        return super().forward(observations, new_start=new_start)
