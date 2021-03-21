from erl.rl_experiment import RLExperiment
from erl.classification_experiment import ClassificationExperiment
from erl.features_extractors import DefaultFeaturesExtractor, VAEFeaturesExtractor
from erl.models.simple import SimpleNet

def run():
    t = RLExperiment()
    t.train()

def run_rl(args):
    exp1 = RLExperiment(features_extractor_class=VAEFeaturesExtractor, env_id="DefaultEnv-v0", render=args.render)
    exp1.train(total_timesteps=args.total_timesteps)

def run_mnist(args):
    exp2 = ClassificationExperiment(network_class=SimpleNet)
    exp2.load_mnist()
    exp2.train(num_epochs=3)
