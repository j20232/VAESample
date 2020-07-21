import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


batch_size = 32
num_workers = 2

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=num_workers)
