import os
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from stable_baselines3.common import logger
from stable_baselines3.common.utils import get_latest_run_id
from torchvision.transforms.transforms import Resize
import wandb

class BasicVAEExperiment:
    def __init__(self, network_class, experiment_name = "mnist_vae", network_args={}, pretrained_model_path=None, save_model_path=None) -> None:
        self.experiment_name = experiment_name

        self.default_img_size = 32
        self.train_loader = None
        self.test_loader = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = network_class(default_img_size=self.default_img_size, **network_args).to(self.device)
        if pretrained_model_path is not None: # load pretrained model
            if os.path.exists(pretrained_model_path):
                self.model.load_state_dict(torch.load(pretrained_model_path))
            else:
                print("Warning: pretrained model not found. Training from scratch.")
        self.save_model_path = save_model_path

        self.optimizer = optim.Adam(self.model.parameters())

        self.setup_log()
        self.load_mnist()

    def setup_log(self):
        """ Setup Tensorboard loger sub-system
        Tensorboard logger was implemented in stable-baselines3.
        Use the command `tensorboard --logdir tb` to view the log.
        """
        latest_run_id = get_latest_run_id('tb', self.experiment_name)
        save_path = os.path.join('tb', f"{self.experiment_name}_{latest_run_id + 1}")
        logger.configure(save_path, ['tensorboard', 'wandb'])
        print(f"Write to {save_path}")
        # after init
        wandb.config.hyper_test = 0
        wandb.watch([self.model], log='all')

    def load_mnist(self):
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        train_kwargs = {'batch_size': 256}
        test_kwargs = {'batch_size': 10}
        if self.use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
            test_kwargs.update({'shuffle': False})
        transform = transforms.Compose([
            transforms.Resize(self.default_img_size),
            transforms.ToTensor(),
            transforms.Normalize((self.normalize_mean,), (self.normalize_std,)),
        ])
        # download is not available according to https://github.com/pytorch/vision/issues/1938
        # use this repo instead: https://github.com/knamdar/data
        dataset1 = datasets.MNIST(f'{os.getcwd()}/data', train=True, download=False, transform=transform)
        dataset2 = datasets.MNIST(f'{os.getcwd()}/data', train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    def train(self, num_epochs=1):
        current_step = 0 # How many data points have been visited.
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, sigma = self.model(data)
                loss = self.model.loss_function(recon, data, mu, sigma) # loss function defined with the model 
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
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                recons, mu, sigma = self.model(data)
                logger.record(f"latent/mu", mu.detach().cpu())
                logger.record(f"latent/sigma", sigma.detach().cpu())
                recons_1, mu, sigma = self.model(data)

                for i in range(3):
                    _source = data[i].view(self.default_img_size,self.default_img_size)
                    _recon = recons[i].view(self.default_img_size,self.default_img_size)
                    _recon_1 = recons_1[i].view(self.default_img_size,self.default_img_size)
                    
                    _source = self.denormalize(_source)
                    _recon = self.denormalize(_recon)
                    _recon_1 = self.denormalize(_recon_1)

                    _img = torch.cat([_source, _recon, _recon_1], dim=1)
                    # _img = torch.clip(_img, 0.0, 1.0)
                    # logger.record(f"source/({i})_{target[i]}", logger.Image(_source, "HW"))
                    # logger.record(f"recon/({i})_{target[i]}", logger.Image(_recon, "HW"))

                    logger.record(f"compare/({i})_{target[i]}", wandb.Image(_img.detach().cpu())) # will only log to wandb

                    logger.record(f"source_dist/({i})_{target[i]}", _source.detach().cpu())
                    logger.record(f"reconstructions_dist/({i})_{target[i]}", _recon.detach().cpu())
                break