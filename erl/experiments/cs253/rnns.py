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
from stable_baselines3.common.torch_layers import FlattenExtractor
import erl.envs  # need this to register the bullet envs
from erl.tools.wandb_logger import WandbCallback
from erl.tools.gym_helper import make_env

import wandb

from erl.customized_agents.customized_ppo import CustomizedPPO
from erl.customized_agents.customized_callback import CustomizedEvalCallback
class MultiRNNExp:
    """ 
    A whole experiment.
    It should contain: (1) environments, (2) policies, (3) training, (4) testing.
    The results should be able to compare with other experiments.

    The Multi-RNN experiment.
    """

    def __init__(self,
                 args,
                 env_id="HopperBulletEnv-v0",
                 policy="MlpPolicy",
                 features_extractor_class=FlattenExtractor,
                 features_extractor_kwargs={},
                 ) -> None:
        print("Starting MultiRNNExp")
        """ Init with parameters to control the training process """
        self.args = args
        self.env_id = env_id
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Make Environments
        print("Making train environments...")
        venv = DummyVecEnv([make_env(env_id=env_id, rank=i, seed=args.seed, render=args.render) for i in range(args.num_envs)])
        features_extractor_kwargs["num_envs"] = args.num_envs
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
            # Note: net_arch must be specified, because sb3 won't set the default network architecture if we change the features_extractor.
            # pi: Actor (policy-function); vf: Critic (value-function)
            "net_arch" : [dict(pi=[64, 64], vf=[64, 64])],
        }
        
        self.model = CustomizedPPO(policy, venv, n_steps=args.rollout_n_steps, tensorboard_log="tb", policy_kwargs=policy_kwargs)
        self.model.experiment = self  # pass the experiment handle into the model, and then into the TrainVAECallback
        
        self.eval_env = make_env(env_id=env_id, rank=99, seed=args.seed, render=False)()

    def train(self) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")

        callback = [
            WandbCallback(self.args),
            CustomizedEvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=self.args.eval_freq,
                n_eval_episodes=3,
                verbose=0,
            )
        ]
        with torch.autograd.set_detect_anomaly(True):
            self.model.learn(self.args.total_timesteps, callback=callback)

