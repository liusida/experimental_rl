import os
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from stable_baselines3.common import logger
from stable_baselines3.common.utils import get_latest_run_id
from torchvision.transforms.transforms import Resize


class VAECameraExperiment:
    def __init__(self, network_class, experiment_name = "camera_vae", network_args={}, pretrained_model_path=None, save_model_path=None) -> None:
        self.experiment_name = experiment_name

        self.default_img_size = 64
        self.in_channels = 3
        self.train_loader = None
        self.train_loader = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = network_class(in_channels=self.in_channels, default_img_size=self.default_img_size, **network_args).to(self.device)
        if pretrained_model_path is not None: # load pretrained model
            if os.path.exists(pretrained_model_path):
                self.model.load_state_dict(torch.load(pretrained_model_path))
            else:
                print("Warning: pretrained model not found. Training from scratch.")
        self.save_model_path = save_model_path

        self.optimizer = optim.Adam(self.model.parameters())

        self.setup_log()
        self.load_rl_camera()

    def setup_log(self):
        """ Setup Tensorboard loger sub-system
        Tensorboard logger was implemented in stable-baselines3.
        Use the command `tensorboard --logdir tb` to view the log.
        """
        latest_run_id = get_latest_run_id('tb', self.experiment_name)
        save_path = os.path.join('tb', f"{self.experiment_name}_{latest_run_id + 1}")
        logger.configure(save_path, ['tensorboard'])
        print(f"Write to {save_path}")

    def load_rl_camera(self):
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        data_path = f'{os.getcwd()}/data/camera/'
        transform = transforms.Compose([
            transforms.Resize(self.default_img_size),
            transforms.ToTensor(), 
            transforms.Normalize(self.normalize_mean, self.normalize_std)
            ])
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=1,
            shuffle=True
        )


    def train(self, num_epochs=1):
        current_step = 0 # How many data points have been visited.
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, sigma = self.model(data)
                # beta_weight stands for "How much Variational flavor do you want?"
                loss = self.model.loss_function(recon, data, mu, sigma, beta_weight=0.2) # loss function defined with the model 
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    current_step = batch_idx*len(data) + epoch*len(self.train_loader.dataset)
                    print(f"[{epoch}]({batch_idx* len(data)}/{len(self.train_loader.dataset)}) loss: {loss.item():.6f}")
                    logger.record("vae/train_loss", loss.item())
                    self.test_reconstructions()
                    logger.dump(step=current_step)
                # break # early stop during debugging
            if self.save_model_path is not None: # save model
                torch.save(self.model.state_dict(), self.save_model_path)

    def denormalize(self, t):
        denormlized_img = t.mul_(self.normalize_std).add_(self.normalize_mean)
        return torch.clip(denormlized_img, 0.0, 1.0)

    def test_reconstructions(self):
        """ test reconstruction """
        # self.model.eval()
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                recons, mu, sigma = self.model(data)
                logger.record(f"latent/mu", mu)
                logger.record(f"latent/sigma", sigma)
                recons_1, mu, sigma = self.model(data)

                for i in range(3):
                    _source = data[i].view(self.in_channels, self.default_img_size,self.default_img_size)
                    _recon = recons[i].view(self.in_channels, self.default_img_size,self.default_img_size)
                    _recon_1 = recons_1[i].view(self.in_channels, self.default_img_size,self.default_img_size)
                    
                    _source = self.denormalize(_source)
                    _recon = self.denormalize(_recon)
                    _recon_1 = self.denormalize(_recon_1)

                    _img = torch.cat([_source, _recon, _recon_1], dim=2)
                    # _img = torch.clip(_img, 0.0, 1.0)
                    # logger.record(f"source/({i})_{target[i]}", logger.Image(_source, "HW"))
                    # logger.record(f"recon/({i})_{target[i]}", logger.Image(_recon, "HW"))

                    logger.record(f"compare/({i})_{target[i]}", logger.Image(_img, "CHW"))
                    logger.record(f"source_dist/({i})_{target[i]}", _source)
                    logger.record(f"reconstructions_dist/({i})_{target[i]}", _recon)
                break