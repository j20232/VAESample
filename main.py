import yaml
import argparse
import os
import random
import torch
import numpy as np

from models import *
from experiment import VAEExperiment
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
            raise Exception(exc)
    seed_everything()

    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )
    model = vae_models[config["model_params"]["name"]](**config["model_params"])
    experiment = VAEExperiment(model,
                               config['exp_params'])

    runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                     min_nb_epochs=1,
                     logger=tt_logger,
                     log_save_interval=100,
                     train_percent_check=1.,
                     val_percent_check=1.,
                     num_sanity_val_steps=5,
                     early_stop_callback=False,
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
