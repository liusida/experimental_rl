import os
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from stable_baselines3.common import logger
from stable_baselines3.common.utils import get_latest_run_id

from erl.models.simple import SimpleNet


class BasicMNISTExperiment:
    def __init__(self, network_class, experiment_name = "mnist", network_args={}) -> None:
        self.experiment_name = experiment_name

        self.train_loader = None
        self.test_loader = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = network_class(input_dim=28*28, output_dim=10, **network_args).to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.01)
        self.setup_log()
        self.load_mnist()

    def setup_log(self):
        """ Setup Tensorboard loger sub-system
        Tensorboard logger was implemented in stable-baselines3.
        Use the command `tensorboard --logdir tb` to view the log.
        """
        latest_run_id = get_latest_run_id('tb', self.experiment_name)
        save_path = os.path.join('tb', f"{self.experiment_name}_{latest_run_id + 1}")
        logger.configure(save_path, ['tensorboard'])
        print(f"Write to {save_path}")

    def load_mnist(self):
        train_kwargs = {'batch_size': 64}
        test_kwargs = {'batch_size': 1000}
        if self.use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
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
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target) # CrossEntropyLoss() combines LogSoftmax and NLLLoss in one single class.
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    current_step = batch_idx*len(data) + epoch*len(self.train_loader.dataset)
                    print(f"[{epoch}]({batch_idx* len(data)}/{len(self.train_loader.dataset)}) loss: {loss.item():.6f}")
                    logger.record("mnist/train_loss", loss.item())
                    self.test()
                    logger.dump(step=current_step)
        self.record_incorrect_predictions()
        logger.dump(step=current_step)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_items = pred.eq(target.view_as(pred))
                correct += correct_items.sum().item()

        test_accuracy = 100.*correct/len(self.test_loader.dataset)
        print(f"Test loss: {test_loss:.6f}. Accuracy: {test_accuracy} %.")
        logger.record("mnist/test_loss", test_loss)
        logger.record("mnist/test_accuracy", test_accuracy)

    def record_incorrect_predictions(self):
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_items = pred.eq(target.view_as(pred))
                incorrect_idx = correct_items.logical_not()
                incorrect_data = data[incorrect_idx]
                incorrect_target = target.view_as(pred)[incorrect_idx]
                incorrect_pred = pred[incorrect_idx]
                for i in range(incorrect_data.shape[0]):
                    logger.record(f"incorrect_samples/({i})_{incorrect_target[i]}_pred_{incorrect_pred[i]}", logger.Image(incorrect_data[i], "HW"))
                break # only sample part of them