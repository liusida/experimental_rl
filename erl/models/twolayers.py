from typing import *
import torch
from torch import nn
from torch.nn import functional as F

class TwoLayerNet(nn.Module):
    """ The simplest network module.
    Contains only one linear layer.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """ 
        input_dim: the dimension of input layer, could be a list or a number, e.g. [1,28,28] is equivalent to 1*28*28.
        output_dim: the dimension of output layer
        """
        super().__init__()
        if isinstance(input_dim, List):
            # e.g. [1,28,28] is equivalent to 1*28*28.
            _tmp = 1
            for i in input_dim:
                _tmp *= i
            input_dim = _tmp
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x