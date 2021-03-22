import cv2
import gym

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from stable_baselines3 import PPO
# from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
import erl.envs # need this to register the bullet envs

import wandb

class TrainVAECallback(BaseCallback):
    """ Train VAE in this callback:
        1. Get an image
        2. pass that image to VAE and train.
    """
    def __init__(self, device, verbose: int=False):
        super().__init__(verbose=verbose)
        self.camera_img_batch_size = 64
        self.camera_img_batch = []
        self.resize = transforms.Resize(64) # default image size, hard coded
        self.device = device
        self.test_img = None


    def _on_step(self):
        observations = self.locals["new_obs"]
        # this is hard coded. refer to default_envs.py
        camera_img = observations[:,46:].reshape(-1,3,256,256)
        camera_img = camera_img[0] # image from the first environment
        camera_img = self.resize(torch.as_tensor(camera_img))
        wandb.log({"camera/img": wandb.Image(camera_img), "num_timesteps": self.num_timesteps})

        if self.test_img is None:
            self.test_img = camera_img
            self.test_img = torch.unsqueeze(self.test_img, 0)
            self.test_img = self.test_img.to(self.device)

        if len(self.camera_img_batch) >= self.camera_img_batch_size:
            self.train_vae()
            self.camera_img_batch = []
        else:
            self.camera_img_batch.append(camera_img)

    def train_vae(self):
        data = torch.stack(self.camera_img_batch, dim=0)
        data = data.to(self.device)
        self.model.experiment.optimizer.zero_grad()
        recon, mu, sigma = self.model.experiment.vae_model(data)
        # beta_weight stands for "How much Variational flavor do you want?"
        loss = self.model.experiment.vae_model.loss_function(recon, data, mu, sigma, beta_weight=0.2) # loss function defined with the model 
        loss.backward()
        self.model.experiment.optimizer.step()
        wandb.log({"vae/train_loss": loss.item(), "num_timesteps": self.num_timesteps})

        recon, _, _ = self.model.experiment.vae_model(self.test_img)
        _test_img = torch.cat([self.test_img, recon], dim=3)
        wandb.log({"camera/test_img_reconstruct": wandb.Image(_test_img[0]), "num_timesteps": self.num_timesteps})


class RLButVAEExperiment:
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
                 vae_class=None,
                 vae_kargs={},
                 ) -> None:
        """ Init with parameters to control the training process """
        self.env_id = env_id
        self.render = render
        self.vae_class = vae_class

        # Set up VAE model
        self.default_img_size = 64
        self.in_channels = 3
        self.train_loader = None
        self.train_loader = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.vae_model = vae_class(in_channels=self.in_channels, default_img_size=self.default_img_size, **vae_kargs).to(self.device)
        self.optimizer = optim.Adam(self.vae_model.parameters())


        env = gym.make(self.env_id)
        if self.render:
            env.render(mode="human")
        self.model = algorithm(policy, env, tensorboard_log="tb", policy_kwargs={"features_extractor_class":features_extractor_class})
        self.model.experiment = self # pass the experiment handle into the model, and then into the TrainVAECallback

    def train(self, total_timesteps=1e4) -> None:
        """ Start training """
        print(f"train using {self.model.device.type}")
        self.model.learn(total_timesteps, callback=TrainVAECallback(device=self.device))

