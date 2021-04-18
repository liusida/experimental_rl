from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomizedPolicy(ActorCriticPolicy):
    def evaluate_actions_rnn(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        evaluate actions recurrently.
        """
        seq = []
        for i in range(obs.shape[0]):
            seq.append(obs[i])
        
        for obs_step in seq:
            latent_pi, latent_vf, latent_sde = self._get_latent(obs_step.unsqueeze(0))
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
