# from ppo.py
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

# from on_policy_algorithm.py
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

# this file
from stable_baselines3 import PPO

from erl.customized_agents.customized_rollout import CustomizedRolloutBuffer


class CustomizedPPO(PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """
        Sida: switch to CustomizedRolloutBuffer
        """
        super().__init__(policy, env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
                         gae_lambda=gae_lambda, clip_range=clip_range, clip_range_vf=clip_range_vf, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                         use_sde=use_sde, sde_sample_freq=sde_sample_freq, target_kl=target_kl, tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                         policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)
        self.rollout_buffer = CustomizedRolloutBuffer(
            rnn_num_parallel_module=self.policy.features_extractor.num_parallel_rnns,
            rnn_size_per_module = self.policy.features_extractor.size_per_module,
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Sida:
        Two parts: one for mlps or flatten, one for rnns.
        They will be trained differently.
        """
        # train rnn
        self.train_recurrent()
        # train normal feed forward
        super().train()
    
    def train_recurrent(self):
        """
        get non-randomized obs sequence, and pass through rnn, and backpropagate through time.
        """
        if self.policy.features_extractor.num_parallel_rnns==0: # no rnn
            return
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_sequence(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                """
                Sida: Change the input to evaluate_actions()
                """
                with self.policy.features_extractor.start_training(rollout_data.short_hidden_states, rollout_data.long_hidden_states):
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        logger.record("train_rnn/entropy_loss", np.mean(entropy_losses))
        logger.record("train_rnn/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train_rnn/value_loss", np.mean(value_losses))
        logger.record("train_rnn/approx_kl", np.mean(approx_kl_divs))
        logger.record("train_rnn/clip_fraction", np.mean(clip_fractions))
        logger.record("train_rnn/loss", loss.item())
        logger.record("train_rnn/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train_rnn/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train_rnn/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train_rnn/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train_rnn/clip_range_vf", clip_range_vf)
        logger.dump(step=self.num_timesteps)


    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: CustomizedRolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        """
        Sida
        """
        short_hidden_states, long_hidden_states = None, None

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                """
                Sida: get memory before passing forward, assuming there's only one rnn module for now.
                """
                if self.policy.features_extractor.num_parallel_rnns:
                    short_hidden_states = self.policy.features_extractor.cx_rollout.cpu().numpy()
                    long_hidden_states = self.policy.features_extractor.hx_rollout.cpu().numpy()

                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            
            """
            Sida: add memory to rollout buffer
            """
            rollout_buffer.add(
                short_hidden_states, long_hidden_states,
                self._last_obs, actions, rewards, self._last_dones, values, log_probs)

            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True