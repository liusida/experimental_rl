import gym
from gym.envs.registration import registry
import pybullet_envs

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)

register(id='Walker2DwithVisionEnv-v0',
         entry_point='erl.envs.walker2d_with_vision:Walker2DWithVisionEnv',
         max_episode_steps=1000)

register(id='Walker2DOnlyVisionEnv-v0',
         entry_point='erl.envs.walker2d_only_vision:Walker2DOnlyVisionEnv',
         max_episode_steps=1000)
