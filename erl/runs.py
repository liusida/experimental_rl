from .experiment import Experiment
from .features_extractors import DefaultFeaturesExtractor, VAEFeaturesExtractor

def run():
    t = Experiment()
    t.train()

def run1(args):
    exp1 = Experiment(features_extractor_class=VAEFeaturesExtractor, env_id="DefaultEnv-v0", render=args.render)
    exp1.train(total_timesteps=args.total_timesteps)
