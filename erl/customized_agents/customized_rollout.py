import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import RolloutBuffer


class CustomizedRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        """
        reset short and long states, and call super
        """
        # TODO: dynamically set hidden_dim
        self.hidden_dim = 64

        self.short_hidden_states = np.zeros((self.buffer_size, self.n_envs, self.hidden_dim), dtype=np.float32)
        self.long_hidden_states = np.zeros((self.buffer_size, self.n_envs, self.hidden_dim), dtype=np.float32)
        super().reset()

    def add(
        self,
        short_hidden_state: np.ndarray,
        long_hidden_state: np.ndarray,
        obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: th.Tensor, log_prob: th.Tensor
    ) -> None:
        """
        Sida: add short and long states, and call super.
        """
        self.short_hidden_states[self.pos] = np.array(short_hidden_state).copy()
        self.long_hidden_states[self.pos] = np.array(long_hidden_state).copy()

        super().add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            """
            Sida: add short and long
            """
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns", "short_hidden_states", "long_hidden_states"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        """
        Sida: add short and long
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.short_hidden_states[batch_inds],
            self.long_hidden_states[batch_inds],
        )
        return CustomizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))

class CustomizedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    short_hidden_states: th.Tensor
    long_hidden_states: th.Tensor
