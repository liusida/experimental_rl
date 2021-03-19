from .experiment import Experiment
from .features_extractors.default_extractor import DefaultFeaturesExtractor

def run():
    t = Experiment()
    t.train()

def run1():
    exp1 = Experiment(features_extractor_class=DefaultFeaturesExtractor, env_id="DefaultEnv-v0")
    exp1.train()

def run2():
    exp1 = Experiment(features_extractor_class=DefaultFeaturesExtractor, env_id="HumanoidFlagrunBulletEnv-v0")
    exp1.train()