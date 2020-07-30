import numpy as np
import tkinter as tk
import torch
from PIL import ImageTk
from torchvision import transforms


class LatentSlider(tk.Frame):
    def __init__(self, parent_canvas, latent_num, width=None, height=None,
                 img=None, image_panel=None, experiment=None, device="cpu", impact_order=None):
        super().__init__(parent_canvas, width=width, height=height)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.width = width
        self.height = height
        self.latent_num = latent_num
        self.impact_order = impact_order

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
        self.labels = []
        self.texts = []
        for i in range(latent_num):
            self.texts.append(tk.StringVar(master=self.canvas))
            self.labels.append(tk.Label(self.interior, textvariable=self.texts[i]))
            text = str(self.impact_order[i][0]) + "(" + format(self.impact_order[i][1], ".3f") + "): " + str(0.0)
            self.texts[i].set(text)
            self.labels[i].pack(anchor=tk.NW, fill=tk.X, padx=(30, 30), pady=(0, 10))

            self.vals.append(tk.DoubleVar(master=self.canvas, value=1.0))
            self.scales.append(tk.Scale(self.interior, orient='h', variable=self.vals[i], showvalue=False,
                                        from_=-128.0, to=128.0, length=int(self.width / 2 * 0.8),
                                        command=self._scaled(i)))
            self.vals[i].set(0.0)
            self.scales[i].pack(anchor=tk.NW, fill=tk.X, padx=(30, 30), pady=(0, 10))
        self.img = img
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
            values = np.array([6 * self.vals[i].get() / 255 for i in range(len(self.vals))], dtype=np.float32).reshape(1, -1)
            sorted_values = np.zeros_like(values)
            for i in range(values.shape[1]):
                sorted_values[0, self.impact_order[i][0]] = values[0, i]
            z = torch.tensor(sorted_values)
            tensor = self.experiment.model.sample_with_value(z, self.device)
            tensor = (tensor + 1.0) / 2.0
            for i in range(len(self.labels)):
                text = str(self.impact_order[i][0]) + "(" + format(self.impact_order[i][1], ".3f") + "): " + str(values[0, i])
                self.texts[i].set(text)
            img = transforms.ToPILImage()(tensor[0].detach().cpu())

            img = img.resize((int(self.width), int(self.height)))
            self.img = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=self.img)
            self.image_panel.image = self.img
        return x
