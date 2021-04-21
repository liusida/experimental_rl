import numpy as np
from pybullet_envs.gym_locomotion_envs import HopperBulletEnv

class HopperMaskEnv(HopperBulletEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        # obs = np.zeros_like(obs)
        obs[10] = 0
        obs[12] = 0
        obs[14] = 0
        return obs, reward, done, info