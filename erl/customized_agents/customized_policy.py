from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch as th
from stable_baselines3.common.preprocessing import get_action_dim, maybe_transpose, preprocess_obs

from stable_baselines3.common.policies import ActorCriticPolicy


class CustomizedPolicy(ActorCriticPolicy):
    def evaluate_actions_rnn(self, obs: th.Tensor, actions: th.Tensor, dones: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        evaluate actions recurrently.
        """
        for i in range(obs.shape[0]):
            if i==0:
                new_start = None
            else:
                new_start = dones[i-1]
            latent_pi, latent_vf, latent_sde = self._get_latent(obs[i], new_start)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def extract_features(self, obs: th.Tensor, new_start: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs, new_start)

    def _get_latent(self, obs: th.Tensor, new_start: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs, new_start)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde
