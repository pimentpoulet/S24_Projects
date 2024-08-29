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
from PIL import Image

from functions import *


# setup paths
train_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\fft_dataset_origin\train"
test_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\fft_dataset_origin\test"


""" VISUALIZE AN IMAGE FROM FFT DATASET """

# 1. Get all image paths
# 2. Pick an image's path
# 3. Get the image class name using pathlib.Path.parent.stem
# 4. Open the image with Python's PIL
# 5. Show the image and print metadata

dir_path = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\fft_dataset")

# 1.
image_path_list = list(dir_path.glob("*/*/*.jpg"))

# 2.
random_image_path = random.choice(image_path_list)
print(f"\nrandom_image_path: {random_image_path}")

# 3.
image_class = random_image_path.parent.stem
print(f"parent: {random_image_path.parent}")
print(f"stem: {image_class}")

# 4.
img = Image.open(random_image_path)

# 5. Show the image using PIL
print(f"\nRandom image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}\n")
# img.show()

# 5. Show the image using matplotlib.pyplot
img_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_array)
plt.title(f"Image class: {image_class} | Image shape: {img_array.shape} -> [height, width, color_channels]")
plt.axis(False)
# plt.show()


""" TURN IMAGES INTO TENSORS """

data_transform = transforms.Compose([
    transforms.ToTensor()
])

img_tensor = data_transform(img)

# plot_transformed_image(image_paths=image_path_list,
#                        transform=data_transform,
#                        n=3)


""" CREATE TRAINING AND TESTING DATASETS """

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform,
                                 target_transform=None)

print(f"{train_data}\n{test_data}")
print(train_data.classes)

print(len(train_data), len(test_data))
class_names = train_data.classes


""" DATA LOADER """

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
