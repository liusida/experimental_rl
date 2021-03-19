from .experiment import Experiment
from .features_extractors.default_extractor import DefaultFeaturesExtractor

def run():
    t = Experiment()
    t.train()

def run1(render=False):
    exp1 = Experiment(features_extractor_class=DefaultFeaturesExtractor, env_id="DefaultEnv-v0", render=render)
    exp1.train(total_timesteps=1e8)