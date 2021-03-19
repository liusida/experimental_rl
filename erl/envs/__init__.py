import gym
from gym.envs.registration import registry, make, spec
import pybullet_envs

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)

register(id='DefaultEnv-v0',
         entry_point='erl.envs.default_envs:DefaultEnv',
         max_episode_steps=1000)
