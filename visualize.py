import yaml
import argparse
import tkinter as tk
import numpy as np
from PIL import ImageTk
from pathlib import Path
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
from torchvision import transforms
from experiment import VAEExperiment
from models import *
from utils import seed_everything


class LatentAdjustor(tk.Frame):
    def __init__(self, parent_canvas, latent_num, width=None, height=None,
                 img=None, image_canvas=None, image_panel=None, experiment=None, device="cpu"):
        super().__init__(parent_canvas, width=width, height=height)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.width = width
        self.height = height
        self.latent_num = latent_num

        # Create a canvas and a scroll bar
        self.scroll_bar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.scroll_bar.pack(fill=tk.Y, side=tk.RIGHT, expand=False)
        self.canvas = tk.Canvas(self, bg="white", yscrollcommand=self.scroll_bar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_bar.config(command=self.canvas.yview)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        # Create a frame
        self.interior = tk.Frame(self.canvas, bg="gray90", borderwidth=10)
        self.interior_id = self.canvas.create_window(0, 0, window=self.interior, anchor=tk.NW)
        self.interior.bind('<Configure>', self._configure_interior)
        self.canvas.bind('<Configure>', self._configure_canvas)
        self.canvas.bind("<MouseWheel>", self._mouse_y_scroll)
        self.interior.bind("<MouseWheel>", self._mouse_y_scroll)

        self.vals = []
        self.scales = []
        for i in range(latent_num):
            self.vals.append(tk.DoubleVar(master=self.canvas, value=1.0))
            self.scales.append(tk.Scale(self.interior, orient='h', showvalue=True, variable=self.vals[i],
                                        from_=0.0, to=255.0, length=int(self.width / 2 * 0.8),
                                        command=self._scaled(i)))
            self.scales[i].pack(anchor=tk.NW, fill=tk.X, padx=(30, 30), pady=(0, 10))
        self.img = img
        self.image_canvas = image_canvas
        self.image_panel = image_panel
        self.experiment = experiment

    def _configure_interior(self, event=None):
        size = (self.interior.winfo_reqwidth(), self.interior.winfo_reqheight())
        self.canvas.config(scrollregion="0 0 %s %s" % size)
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            self.canvas.config(width=self.interior.winfo_reqwidth())

    def _configure_canvas(self, event=None):
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())

    def _mouse_y_scroll(self, event):
        if event.delta > 0:
            self.canvas.yview_scroll(-1, 'units')
        elif event.delta < 0:
            self.canvas.yview_scroll(1, 'units')

    def _scaled(self, i):
        def x(v):
            values = np.array([(self.vals[i].get() - 128) / 255 for i in range(len(self.vals))], dtype=np.float32).reshape(1, -1)
            z = torch.tensor(values)
            tensor = self.experiment.model.sample_with_value(z, self.device)
            tensor = (tensor + 1.0) / 2.0
            img = transforms.ToPILImage()(tensor[0].detach().cpu())

            img = img.resize((int(self.width), int(self.height)))
            self.img = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=self.img)
            self.image_panel.image = self.img
        return x


class LatentVisualizer():
    def __init__(self, experiment, latent_num, title="LatentVisualizer", width=800, height=400, device="cpu"):
        self.device = device
        self.title = title
        self.width = width
        self.height = height
        self.latent_num = latent_num
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(f"{self.width}x{self.height}")
        self.experiment = experiment
        self._set_image()

        self.frame_canvas = tk.Canvas(self.root, bg="white")
        self.frame_canvas.place(x=int(self.width / 2), y=0, width=int(self.width / 2), height=self.height)
        self.frame = LatentAdjustor(parent_canvas=self.frame_canvas, latent_num=self.latent_num, width=self.width / 2, height=self.height,
                                    img=self.img, image_canvas=self.image_canvas, image_panel=self.image_panel, experiment=self.experiment, device=self.device)
        self.frame.place(x=0, y=0, relwidth=1.0, height=height)

    def _set_image(self):
        self.image_canvas = tk.Canvas(self.root, bg="white")
        self.image_canvas.place(x=0, y=0, width=int(self.width / 2), height=self.height)

        # tmp
        z = torch.tensor(np.zeros((1, self.experiment.model.latent_dim), dtype=np.float32)).to(self.device)
        tensor = self.experiment.model.sample_with_value(z, self.device)
        tensor = (tensor + 1.0) / 2.0
        img = transforms.ToPILImage()(tensor[0].detach().cpu())

        img = img.resize((int(self.width / 2), self.height))
        self.img = ImageTk.PhotoImage(img)
        self.image_panel = tk.Label(self.image_canvas, image=self.img)
        self.image_panel.pack(side="bottom", fill="both", expand="yes")

    def run(self):
        self.root.mainloop()


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
    visualizer = LatentVisualizer(experiment, latent_num=experiment.model.latent_dim, device=config["exp_params"]["device"])
    visualizer.run()
