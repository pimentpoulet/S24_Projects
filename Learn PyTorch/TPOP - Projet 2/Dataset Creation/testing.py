import torch as t
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import glob
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timeit import default_timer as timer
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

from functions import *


height = 600
width = 600
image = Image.new(mode='L', size=(height, width), color=255)

# Draw some lines
draw = ImageDraw.Draw(image)
y_start = 0
y_end = image.height
step_size = int(image.width / 10)

for x in range(0, image.width, step_size):
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=128)

del draw

image.show()




"""
# save img_array
saving_path_num_3 = os.path.join(saving_path_num_2, f"{num}.png")
plt.imsave(saving_path_num_3, img_array, cmap="gray")

# perform fft of image_array
img_fft = np.fft.fftshift(np.fft.fft2(img_array))

# convert to log
img_log = 1 + np.abs(img_fft)

# normalize image_log
img_save = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))

# save img_save
saving_path_fft_3 = os.path.join(saving_path_fft_2, f"{num}.png")
plt.imsave(saving_path_fft_3, img_save, cmap="gray")
"""
