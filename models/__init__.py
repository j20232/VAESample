from .base import *
from .vanilla_vae import *
from .cvae import *
from .beta_vae import *


vae_models = {
    "VAE": VanillaVAE,
    "VanillaVAE": VanillaVAE,
    "CVAE": ConditonalVAE,
    "ConditonalVAE": ConditonalVAE,
    "BetaVAE": BetaVAE,
}
