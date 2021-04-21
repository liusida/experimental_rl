import numpy as np
from pybullet_envs.gym_locomotion_envs import HopperBulletEnv

class HopperMaskEnv(HopperBulletEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = np.zeros_like(obs)
        return obs, reward, done, info