import numpy as np
from pybullet_envs.gym_locomotion_envs import HopperBulletEnv

class HopperMaskEnv(HopperBulletEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        # obs = np.zeros_like(obs)
        obs[11] = 0
        obs[13] = 0
        return obs, reward, done, info