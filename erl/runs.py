from erl.experiments.vae.vae_with_rl_camera_img import VAECameraExperiment
from erl.experiments.rl.rl_experiment import RLExperiment
from erl.experiments.classifications.basic_mnist import BasicMNISTExperiment
from erl.experiments.vae.basic_vae import BasicVAEExperiment
from erl.experiments.rl.rl_but_vae import RLButVAEExperiment

from erl.features_extractors import DefaultFeaturesExtractor, VAEFeaturesExtractor
from erl.models.simple import SimpleNet
from erl.models.twolayers import TwoLayerNet
from erl.models.vae import VanillaVAE

def run_current_exp(args):
    run_cs287_baseline(args)

def run(args):
    t = RLExperiment()
    t.train()

def run_rl(args):
    exp = RLExperiment(features_extractor_class=VAEFeaturesExtractor, env_id="DefaultEnv-v0", render=args.render)
    exp.train(total_timesteps=args.total_timesteps)

def run_mnist_one_layer(args):
    exp = BasicMNISTExperiment(network_class=SimpleNet, experiment_name="one-layer")
    exp.train(num_epochs=1)

def run_mnist_two_layers(args):
    exp = BasicMNISTExperiment(network_class=TwoLayerNet, experiment_name="two-layer-hidden-16384", network_args={"hidden_dim": 16384})
    exp.train(num_epochs=10)

def run_vae(args):
    exp = BasicVAEExperiment(network_class=VanillaVAE, experiment_name="vae", save_model_path="trained_models/vae.pth")
    # exp = BasicVAEExperiment(network_class=VanillaVAE, experiment_name="vae", pretrained_model_path="trained_models/vae.pth", save_model_path="trained_models/vae.pth")
    exp.train(num_epochs=100)

def run_vae_camera(args):
    exp = VAECameraExperiment(network_class=VanillaVAE, pretrained_model_path="trained_models/vae.pth", save_model_path="trained_models/vae.pth")
    exp.train(num_epochs=100)

def run_rl_with_vae(args):
    exp = RLButVAEExperiment(render=False, features_extractor_class=VAEFeaturesExtractor, vae_class=VanillaVAE)
    exp.train()
    pass

def run_cs287_baseline(args):
    from erl.experiments.cs287.baseline import BaselineExp
    args.seed = 0
    # args.render = False
    # args.num_venvs = 16
    args.total_timesteps = 1e6
    exp = BaselineExp(env_id="HopperBulletEnv-v0", args=args)
    exp.train()