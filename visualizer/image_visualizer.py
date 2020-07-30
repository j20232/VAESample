import numpy as np
import tkinter as tk
import torch
from PIL import ImageTk
from torchvision import transforms

from .latent_slider import LatentSlider


class ImageVisualizer():
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
        self.frame = LatentSlider(parent_canvas=self.frame_canvas, latent_num=self.latent_num, width=self.width / 2, height=self.height,
                                  img=self.img, image_panel=self.image_panel, experiment=self.experiment, impact_order=self.sorted_order, device=self.device)
        self.frame.place(x=0, y=0, relwidth=1.0, height=height)

    def _set_image(self):
        self.image_canvas = tk.Canvas(self.root, bg="white")
        self.image_canvas.place(x=0, y=0, width=int(self.width / 2), height=self.height)

        # tmp
        impact_order = {}
        for i in range(self.experiment.model.latent_dim):
            z = torch.tensor(np.zeros((1, self.experiment.model.latent_dim), dtype=np.float32)).to(self.device)
            z[0, i] = -3.0
            minus_tensor = self.experiment.model.sample_with_value(z, self.device)
            z = torch.tensor(np.zeros((1, self.experiment.model.latent_dim), dtype=np.float32)).to(self.device)
            z[0, i] = 3.0
            plus_tensor = self.experiment.model.sample_with_value(z, self.device)
            impact_order[i] = torch.norm(plus_tensor - minus_tensor).detach().cpu().numpy()
        self.sorted_order = sorted(impact_order.items(), key=lambda x: x[1], reverse=True)

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
