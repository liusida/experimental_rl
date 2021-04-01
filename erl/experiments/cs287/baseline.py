from collections import defaultdict
import time
# import cv2
import gym

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
import erl.envs  # need this to register the bullet envs

import wandb


class WandbCallback(EventCallback):
    """ Watch the models, so the architecture can be uploaded to WandB """

    def __init__(self):
        super().__init__()
        self.log_interval = 1000

        self.last_time_length = defaultdict(lambda: 0)

        self.average_episodic_distance_G = 0
        self.average_episodic_distance_N = 0
        self.average_episodic_length_G = 0
        self.average_episodic_length_N = 0

    def _on_training_start(self) -> None:
        wandb.watch([self.model.policy], log="all", log_freq=100)
        return True

    def episodic_log(self):
        for env_i in range(self.training_env.num_envs):
            if self.locals['dones'][env_i]:
                distance_x = self.training_env.envs[env_i].robot.body_xyz[0]
                self.average_episodic_distance_N += 1
                self.average_episodic_distance_G += (distance_x - self.average_episodic_distance_G) / self.average_episodic_distance_N

                self.average_episodic_length_N += 1
                self.average_episodic_length_G += (self.last_time_length[env_i] - self.average_episodic_length_G) / self.average_episodic_length_N

            self.last_time_length[env_i] = self.training_env.envs[env_i].episodic_steps
        if self.n_calls % self.log_interval != 0:
            # Skip
            return
        wandb.log({
            f'episodes/distance': self.average_episodic_distance_G,
            f'episodes/time_length': self.average_episodic_length_G,
            'step': self.num_timesteps,
        })

    def detailed_log(self):
        if self.n_calls % self.log_interval != 0:
            # Skip
            return

        for env_i in range(self.training_env.num_envs):
            relative_height = self.training_env.buf_obs[None][env_i][0]
            velocity = self.training_env.buf_obs[None][env_i][3]
            distance_x = self.training_env.envs[env_i].robot.body_xyz[0]
            wandb.log({
                f'raw_relative_height/env_{env_i}': relative_height,
                f'raw_velocity/env_{env_i}': velocity,
                f'raw_distance/env_{env_i}': distance_x,
                'step': self.num_timesteps,
            })

        wandb.log({
            'network/values': self.locals['values'].detach().mean().cpu().numpy(),
            'step': self.num_timesteps,
        })

    def _on_step(self):
        self.episodic_log()
        self.detailed_log()
        return True


def make_env(env_id, rank, seed, render, render_index=0):
    def _init():
        # only render one environment
        _render = render and rank in [render_index]

        env = gym.make(env_id, render=_render)
        return env
    return _init


class BaselineExp:
    """ One experiment is a treatment group or a control group.
    It should contain: (1) environments, (2) policies, (3) training, (4) testing.
    The results should be able to compare with other experiments.
    """

    def __init__(self,
                 args,
                 env_id="Walker2DwithVisionEnv-v0",
                 algorithm=PPO,
                 policy="MlpPolicy",
                 features_extractor_class=FlattenExtractor,
                 ) -> None:
        """ Init with parameters to control the training process """
        self.args = args
        self.env_id = env_id
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Make Environments
        print("Making train environments...")
        venv = DummyVecEnv([make_env(env_id=env_id, rank=i, seed=args.seed, render=args.render) for i in range(args.num_venvs)])
        self.model = algorithm(policy, venv, tensorboard_log="tb", policy_kwargs={"features_extractor_class": features_extractor_class})
        self.model.experiment = self  # pass the experiment handle into the model, and then into the TrainVAECallback

    def train(self) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")
        callback = [
            WandbCallback(),
        ]
        self.model.learn(self.args.total_timesteps, callback=callback)
