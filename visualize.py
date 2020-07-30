import yaml
import argparse
from pathlib import Path

from experiment import VAEExperiment
from models import *
from utils import seed_everything
from visualizer import ImageVisualizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('config', type=str, help='path to the config file')
    parser.add_argument('version', type=int, help='Version')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise Exception(exc)
    seed_everything()
    model_name = config["model_params"]["name"]
    model = vae_models[model_name](**config["model_params"])

    ROOT_PATH = Path(".").resolve()
    check_point_dir = ROOT_PATH / "logs" / model_name / f"version_{args.version}" / "checkpoints"
    check_point_file = list(check_point_dir.glob("*"))[0]
    experiment = VAEExperiment.load_from_checkpoint(checkpoint_path=str(check_point_file),
                                                    vae_model=model, params=config["exp_params"])
    visualizer = ImageVisualizer(experiment, latent_num=experiment.model.latent_dim, device=config["exp_params"]["device"])
    visualizer.run()
