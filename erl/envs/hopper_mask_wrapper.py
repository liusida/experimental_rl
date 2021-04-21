import gym

class HopperMaskWrapper(gym.ObservationWrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info