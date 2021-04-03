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
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
import erl.envs  # need this to register the bullet envs
from erl.tools.wandb_logger import WandbCallback

import wandb


def make_env(env_id, rank, seed, render, render_index=0):
    def _init():
        # only render one environment
        _render = render and rank in [render_index]

        env = gym.make(env_id, render=_render)

        assert rank < 100, "seed * 100 + rank is assuming rank <100"
        env.seed(seed*100 + rank)

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
                 features_extractor_kwargs={},
                 ) -> None:
        """ Init with parameters to control the training process """
        self.args = args
        self.env_id = env_id
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Make Environments
        print("Making train environments...")
        venv = DummyVecEnv([make_env(env_id=env_id, rank=i, seed=args.seed, render=args.render) for i in range(args.num_envs)])
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs
        }
        architecture = [dict(pi=[64, 64], vf=[64, 64])]
        self.model = algorithm(policy, venv, tensorboard_log="tb", policy_kwargs=policy_kwargs)
        self.model.experiment = self  # pass the experiment handle into the model, and then into the TrainVAECallback
        
        self.eval_env = make_env(env_id=env_id, rank=99, seed=args.seed, render=False)()

    def train(self) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")

        callback = [
            WandbCallback(self.args),
            EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=2000,
                n_eval_episodes=3,
                verbose=0,
            )
        ]
        self.model.learn(self.args.total_timesteps, callback=callback)
