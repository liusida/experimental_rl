from stable_baselines3 import PPO
from .baseline import BaselineExp

class MultiMlpsExp(BaselineExp):
    def __init__(self,
                 args,
                 env_id="Walker2DwithVisionEnv-v0",
                 algorithm=PPO,
                 policy="MlpPolicy",
                 features_extractor_class=None,
                 features_extractor_kwargs={},
                 ) -> None:
        super().__init__(
            args=args,
            env_id=env_id,
            algorithm=algorithm,
            policy=policy,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )