import gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
import erl.envs # need this to register the bullet envs

class RLExperiment:
    """ One experiment is a treatment group or a control group.
    It should contain: (1) environments, (2) policies, (3) training, (4) testing.
    The results should be able to compare with other experiments.
    """
    def __init__(self,
                 env_id="DefaultEnv-v0",
                 algorithm=PPO,
                 policy="MlpPolicy",
                 features_extractor_class=FlattenExtractor,
                 render=True,
                 ) -> None:
        """ Init with parameters to control the training process """
        self.env_id = env_id
        self.render = render

        env = gym.make(self.env_id)
        if self.render:
            env.render(mode="human")
        self.model = algorithm(policy, env, tensorboard_log="tb", policy_kwargs={"features_extractor_class":features_extractor_class}, n_steps=100)

    def train(self, total_timesteps=1e4) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")
        self.model.learn(total_timesteps)

