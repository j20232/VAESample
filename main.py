import yaml
import argparse
import os
import random
import torch
import numpy as np

from models import *
# from expriment import VAEExperiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


def seed_everything(seed=1116):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_everything()
    model = vae_models[config["model_params"]["name"]](**config["model_params"])
