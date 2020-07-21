import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int, latent_dim: int, hidden_dims: List = None, out_channels=3,
                 ksize=3, stride=2, padding=1, **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self._build_encoder(in_channels, hidden_dims, ksize, stride, padding)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self._build_decoder(latent_dim, hidden_dims, out_channels, ksize, stride, padding)

    def _build_encoder(self, in_channels: int, hidden_dims: int,
                       ksize=3, stride=2, padding=1):
        encoder_modules = []
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=ksize, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_modules)

    def _build_decoder(self, latent_dim: int, hidden_dims: List[int], out_channels: int,
                       ksize=3, stride=2, padding=1, output_padding=1):
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        decoder_modules = []
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=ksize, stride=stride, padding=padding, output_padding=output_padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*decoder_modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=ksize, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels, kernel_size=ksize, stride=stride, padding=padding),
            nn.Tanh()
        )
