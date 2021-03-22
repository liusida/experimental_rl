# From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

from typing import *
from abc import abstractmethod
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 128,
                 default_img_size=64,
                 **kwargs) -> None:
        """ Setup all modules """
        super(VanillaVAE, self).__init__()
        self.in_channels = in_channels
        self.size_after_cnn = 2  # image width and height at the bottleneck, accompanied by many channels.

        # automatcally calculate the hidden channels:
        # 32 -> 64 -> 128 -> 256 ...
        hidden_dims = [32]
        for i in range(int(np.log2(default_img_size))-2):
            hidden_dims.append(hidden_dims[-1]*2)

        self.hidden_dims = hidden_dims.copy()
        self.latent_dim = latent_dim

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # mean and variance
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.size_after_cnn*self.size_after_cnn, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.size_after_cnn*self.size_after_cnn, latent_dim)

        # after sampling, turn latent z into image size (in the last convolutional encoder layer)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.size_after_cnn * self.size_after_cnn)

        # Build Decoder
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.size_after_cnn, self.size_after_cnn)
        assert self.batch_size == result.shape[0], "view() disrupts the dimension, please check the shape."
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        self.batch_size = input.shape[0]
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recons, x, mu, log_var, beta_weight=1.0) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :return:
        """

        kld_weight = beta_weight / recons.shape[0]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        # return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
        return loss

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
