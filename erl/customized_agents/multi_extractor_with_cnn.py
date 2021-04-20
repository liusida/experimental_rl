from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
import torch as th
from torch import nn
from torch.nn import functional as F
from .multi_extractor import MultiExtractor


class MultiExtractorWithCNN(MultiExtractor):
    def __init__(self, observation_space: gym.Space, num_envs=2, flatten=1, num_rnns=2, num_mlps=2, rnn_layer_size=16):
        super().__init__(
            observation_space=observation_space,
            num_envs=num_envs,
            flatten=flatten,
            num_rnns=num_rnns,
            num_mlps=num_mlps,
            rnn_layer_size=rnn_layer_size
        )
        self.cnn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)

    def forward(self, observations: th.Tensor, new_start: Optional[th.Tensor]) -> th.Tensor:
        # obs = 10 + joints + 8x8x3
        image = observations[:, -8*8*3-1:].reshape([-1,8,8,3])
        import matplotlib.pyplot as plt
        plt.imshow(image[0, :,:,0])
        plt.show()
        print(observations.shape)
        print("")
        return super().forward(observations, new_start=new_start)
