import os
import gym
import torch as th
from torch import nn
from torchvision.utils import save_image

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from erl.models.vae import VanillaVAE

class VAEFeaturesExtractor(BaseFeaturesExtractor):
    """
    VAE Features Extractor
    Copied from FlattenExtractor.
    This class could used as a template for future extractors.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        # Hard coded warning. see default_envs.py
        self.basic_state_dim = 46
        self.camera_img_dim = (256,256,3)
        self.vae_latent_dim = 16

        self.camera_img_idx = 0

        self.output_dim = 64

        super().__init__(observation_space, features_dim=self.output_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.basic_state_dim+self.vae_latent_dim*2, self.output_dim)
        self.vae = VanillaVAE(in_channels=3, latent_dim=self.vae_latent_dim, default_img_size=256)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        TODO: The plan is to wire this with VAE and RNN, but How?
        """
        # need to cut the observations in a specific way, so that we can recover the camera image.
        basic_state = observations[:,:46]
        camera_img = observations[:,46:].view(-1,3,256,256)
        if self.camera_img_idx==0:
            os.makedirs("data/camera/default/", exist_ok=True)
        save_image(camera_img[0], f"data/camera/default/{self.camera_img_idx:06d}.png")
        self.camera_img_idx += 1
        if self.camera_img_idx>1e6:
            exit(0)

        recons, mu, log_var = self.vae(camera_img)
        x = th.cat([basic_state, mu, log_var], axis=1)
        x = self.fc1(x)

        return x