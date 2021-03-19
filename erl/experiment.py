from .tests.erl.test_experiment import *

import gym
from stable_baselines3 import PPO
import pybullet_envs


class Experiment:
    def __init__(self,
                 env_id="HumanoidFlagrunHarderBulletEnv-v0",
                 render=True,
                 ) -> None:
        """ Init with parameters to control the training process """
        self.env_id = env_id
        self.render = render

    def train(self) -> None:
        """ Start training """
        env = gym.make(self.env_id)
        if self.render:
            env.render(mode="human")
        model = PPO('MlpPolicy', env)
        model.learn(100)
        env.close()

        print("train")

    def test(self) -> None:
        """ Start testing """
        
        print("test")