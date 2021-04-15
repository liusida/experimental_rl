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
from erl.tools.gym_helper import make_env
from erl.tools.adjust_camera_callback import AdjustCameraCallback

import wandb


class BaselineExp:
    """ 
    default flatten version, no customization.
    """

    def __init__(self,
                 args,
                 env_id="HopperBulletEnv-v0",
                 ) -> None:
        """ Init with parameters to control the training process """
        self.args = args
        self.env_id = env_id
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Make Environments
        print("Making train environments...")
        venv = DummyVecEnv([make_env(env_id=env_id, rank=i, seed=args.seed, render=args.render) for i in range(args.num_envs)])
       
        self.model = PPO("MlpPolicy", venv, tensorboard_log="tb", device=self.device)
        self.model.experiment = self  # pass the experiment handle into the model, and then into the TrainVAECallback
        
        self.eval_env = make_env(env_id=env_id, rank=99, seed=args.seed, render=False)()

    def train(self) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")

        callback = [
            AdjustCameraCallback(),
            WandbCallback(self.args),
            EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=self.args.eval_freq,
                n_eval_episodes=3,
                verbose=0,
            )
        ]
        self.model.learn(self.args.total_timesteps, callback=callback)
