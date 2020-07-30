from .base import *
from .vanilla_vae import *
from .cvae import *


vae_models = {
    "VAE": VanillaVAE,
    "VanillaVAE": VanillaVAE,
    "CVAE": ConditonalVAE,
    "ConditonalVAE": ConditonalVAE
}
