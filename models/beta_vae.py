import torch
from torch import nn
from torch.nn import functional as F

from models import BaseVAE
from .types_ import *


class BetaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int, latent_dim: int, hidden_dims: List = None, out_channels=3,
                 beta: int = 4, gamma: float = 10.0, max_capacity: int = 25, capacity_max_iter: int = 1e5,
                 loss_type: str = "B", ksize=3, stride=2, padding=1, **kwargs) -> None:
        super(BetaVAE, self).__init__()
        self.num_iter = 0
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.max_capacity = torch.tensor(float(max_capacity))
        self.capacity_max_iter = capacity_max_iter
        self.have_label = False
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.last_hdim = hidden_dims[-1]  # last dimension of hidden layers
        self._build_encoder(in_channels, hidden_dims, ksize, stride, padding)
        self.fc_mu = nn.Linear(self.last_hdim * 4, latent_dim)
        self.fc_var = nn.Linear(self.last_hdim * 4, latent_dim)
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
        self.decoder_input = nn.Linear(latent_dim, self.last_hdim * 4)
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
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels, kernel_size=ksize, padding=padding),
            nn.Tanh()
        )

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def encode(self, x: Tensor) -> List[Tensor]:
        """Encodes the input tensor by the encoder and returns the latent vectors

        Args:
            x (Tensor): observation [N, C, H, W]

        Returns:
            List[Tensor]: List of latent vectors (mu, log_var) where log_var = log(sigma ** 2)
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample the latent vector the distribution N(0, 1) converted from N(mu, var) with Reparameterization Trick

        Args:
            mu (Tensor): mean of the latent Gaussian distribution [N, latent_dim]
            log_var (Tensor): [N, latent_dim]

        Returns:
            Tensor: [N, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decodes the given latent vector and mapt it onto the image space

        Args:
            z (Tensor): latent vector [N, latent_dim]

        Returns:
            Tensor: [N, C, H, W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def loss_fn(self, *args, **kwargs) -> dict:
        """Compute the loss.
        KL(N(\mu, \sigma), N(0, 1)) =log(\frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}).
        The theoretical details are written in [Kingma and Welling. ICLR 2014]

        Args:
            args (List): [reconstructed_x, x, mu, log_var]
            kwargs (dict): weight dictionary

        Returns:
            dict: loss dictionary. {"name": value}
        """
        self.num_iter += 1
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, x)
        kld_weight = kwargs["M_N"]
        # KL divergence where both distribution follow Gaussian distributions
        # log_var = log(sigma ** 2)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.max_capacity = self.max_capacity.to(x.device)
            C = torch.clamp(self.max_capacity / self.capacity_max_iter * self.num_iter, 0, self.max_capacity)
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise NotImplementedError("Undefined loss")

        return {"loss": loss, "Reconstruction_loss": recons_loss, "KLD": -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """Samples from the latent space and return the corresponding image space map

        Args:
            num_samples (int): number of samples
            current_device (int): device

        Returns:
            Tensor: samples
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def sample_with_value(self, array, current_device: int, **kwargs) -> Tensor:
        z = array.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """Returns the reconstructed image given an input image x

        Args:
            x (Tensor): [B, C, H, W] where B = 1

        Returns:
            Tensor: [B, C, H, W]
        """
        return self.forward(x)[0]
