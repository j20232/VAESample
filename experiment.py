import math

import torch
from torch import optim
from torch.utils.data import DataLoader

import torchvision.utils as vutils
from torchvision import transforms
from torchvision.datasets import CelebA

import pytorch_lightning as pl

from models import BaseVAE
from models.types_ import *


class VAEExperiment(pl.LightningModule):

    def __init__(self, n, vae_model: BaseVAE, params: dict):
        # n: pytorch_lightning 0.8.5 contains bugs about arguments, this argument is to fix the bug
        super(VAEExperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.current_device = params["device"] if torch.cuda.is_available else "cpu"
        RFB = "retain_first_backpass"
        self.hold_graph = False if RFB not in self.params.keys() else self.params[RFB]

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self._step(batch, batch_idx, optimizer_idx, is_train=True)  # loss
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        return self._step(batch, batch_idx, optimizer_idx, is_train=False)  # loss

    def _step(self, batch, batch_idx, optimizer_idx=0, is_train=True):
        imgs, labels = batch
        self.current_device = imgs.device
        results = self.forward(imgs, labels=labels)
        num_imgs = self.num_train_imgs if is_train else self.num_val_imgs
        return self.model.loss_fn(*results, M_N=self.params["batch_size"] / num_imgs, optimizer_idx=optimizer_idx, batch_idx=batch_idx)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.sample_images()
        return {"val_loss": avg_loss}

    def train_dataloader(self):
        return self._get_dataloader(is_train=True)

    def val_dataloader(self):
        self.sample_dataloader = self._get_dataloader(is_train=False)
        return self.sample_dataloader

    def _get_dataloader(self, is_train=True):
        transform = self.data_transforms()
        if self.params["dataset"] == "celeba":
            split = "train" if is_train else "test"
            dataset = CelebA(root=self.params["data_path"], split=split, transform=transform, download=self.params["download"])
        else:
            raise ValueError("")
        if is_train:
            self.num_train_imgs = len(dataset)
        else:
            self.num_val_imgs = len(dataset)
        return DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=is_train, drop_last=True)

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))
        if self.params["dataset"] == "celeba":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.params["img_size"]),
                transforms.ToTensor(),
                SetRange
            ])
        else:
            return NotImplementedError
        return transform

    def sample_images(self):
        # Sample a reconstructed image
        test_img, test_label = next(iter(self.sample_dataloader))  # N = 1
        test_img = test_img.to(self.current_device)
        test_label = test_label.to(self.current_device)
        recons = self.model.generate(test_img, labels=test_label)
        nrow = 12
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True, nrow=nrow)
        del test_img, recons

    def configure_optimizers(self):

        # Define optimizers
        optims = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        # Check if more than 1 optimizer is required (Used for adversarial training)
        if "LR_2" in self.params and self.params["LR_2"] is not None:
            optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                    lr=self.params['LR_2'])
            optims.append(optimizer2)

        # Check whether to use a scheduler
        scheds = []
        if "scheduler_gamma" in self.params and self.params["scheduler_gamma"] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma=self.params['scheduler_gamma'])
            scheds.append(scheduler)

            # Check if another scheduler is required for the second optimizer
            if "scheduler_gamma_2" in self.params and self.params["scheduler_gamma_2"] is not None:
                scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                              gamma=self.params['scheduler_gamma_2'])
                scheds.append(scheduler2)
            return optims, scheds
        return optims
