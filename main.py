import yaml
import argparse
import os
import random
import torch
import numpy as np

# from models import *
# from expriment import VAEExperiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger


def seed_everything(seed=1116):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_everything()
