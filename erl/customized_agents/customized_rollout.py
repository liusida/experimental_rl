from typing import Dict, Generator, Optional, Union, NamedTuple
from collections import defaultdict

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import RolloutBuffer


class CustomizedRolloutBuffer(RolloutBuffer):
    def __init__(self, rnn_num_parallel_module: int, rnn_size_per_module: int, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: Union[th.device, str], gae_lambda: float, gamma: float, n_envs: int):
        self.rnn_num_parallel_module = rnn_num_parallel_module
        self.rnn_size_per_module = rnn_size_per_module

        super().__init__(buffer_size, observation_space, action_space, device=device, gae_lambda=gae_lambda, gamma=gamma, n_envs=n_envs)
    def reset(self) -> None:
        """
        reset short and long states, and call super
        """
        self.short_hidden_states = np.zeros((self.buffer_size, self.n_envs, self.rnn_num_parallel_module, self.rnn_size_per_module), dtype=np.float32)
        self.long_hidden_states = np.zeros((self.buffer_size, self.n_envs, self.rnn_num_parallel_module, self.rnn_size_per_module), dtype=np.float32)
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
        if short_hidden_state is not None:
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
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns", "short_hidden_states", "long_hidden_states", "dones"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor]) # shape need to be [batch_size, num_env, ...something else...]
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
            self.dones[batch_inds],
        )
        return CustomizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get_sequence(self, batch_size, rnn_seq_length, rnn_move_window_step) -> Generator[RolloutBufferSamples, None, None]:
        """ modified from get() 
        batch_size: is equivalent to rnn_sequence_length.
        """
        assert self.full, ""
        indices = np.arange(self.buffer_size)

        stack_rollout_data = batch_size // self.n_envs # this is the actual batch size
        stack_rollout_data_i = 0
        stack_rollout_data_buf = []

        start_idx = 0
        while start_idx <= self.buffer_size - rnn_seq_length: # don't form sequence shorter than batch_size at the end
            stack_rollout_data_i += 1
            batch_inds = indices[start_idx : start_idx + rnn_seq_length]
            data = (
                self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds],
                self.log_probs[batch_inds],
                self.advantages[batch_inds],
                self.returns[batch_inds],
                self.short_hidden_states[batch_inds],
                self.long_hidden_states[batch_inds],
                self.dones[batch_inds],
            )
            stack_rollout_data_buf.append(tuple(map(self.to_torch, data)))
            if stack_rollout_data_i % stack_rollout_data == 0:
                _data = zip(*stack_rollout_data_buf)
                _data = [th.cat(x, dim=1) for x in _data]
                stack_rollout_data_buf = []
                yield CustomizedRolloutBufferSamples(*_data)
            start_idx += rnn_move_window_step

    def _get_samples_seq(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        """
        Sida: add short and long
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
            self.short_hidden_states[batch_inds],
            self.long_hidden_states[batch_inds],
            self.dones[batch_inds],
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
    dones: th.Tensor