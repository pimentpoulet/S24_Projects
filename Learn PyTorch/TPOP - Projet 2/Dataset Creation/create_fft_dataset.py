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


""" SETUP DATA DIRECTORY """

mnist_path = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\MNIST Dataset")
num_path = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\fft_dataset_generated\numbers"
fft_path = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\fft_dataset_generated\fft"

# train_dir_num = os.path.join(num_path,"train")
# test_dir_num = os.path.join(num_path,"test")


""" CREATE AND SAVE FFT DATASET """

# Define a transform to convert PIL image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.TrivialAugmentWide(num_magnitude_bins=31)
])

start_time = timer()

# iterate over train/test folders of mnist dataset to create
for filename in os.listdir(mnist_path):

    # path to save_folder
    saving_path_num_1 = os.path.join(num_path, filename)
    saving_path_fft_1 = os.path.join(fft_path, filename)

    # create folder
    os.makedirs(saving_path_num_1, exist_ok=True)
    os.makedirs(saving_path_fft_1, exist_ok=True)

    for folder in os.listdir(os.path.join(mnist_path, filename)):

        # get path to the live number folder
        classes = os.path.join(mnist_path, filename, folder)

        # path to save_folder in train/test folders
        saving_path_num_2 = os.path.join(saving_path_num_1, os.path.basename(folder))
        saving_path_fft_2 = os.path.join(saving_path_fft_1, os.path.basename(folder))

        # create number folder
        os.makedirs(saving_path_num_2, exist_ok=True)
        os.makedirs(saving_path_fft_2, exist_ok=True)

        # iterate over images in number folders
        for num, classe in enumerate(os.listdir(classes)):

            # generate dataset
            img = Image.new('RGB', (384, 384), (250, 250, 250))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("OpenSans-Regular.ttf", 307)   # 0.8%
            draw.text((115, -23), str(folder), (0, 0, 0), font=font)    # 0.3% et 0.06%
            img = img.convert('L')

            # turn PIL into tensor, permute shape for matplotlib compatibility and turn into numpy array
            img_tensor = transform(img).permute(1,2,0).squeeze()
            print(img_tensor.shape)
            img_array = img_tensor.numpy()

            # save img_array
            saving_path_num_3 = os.path.join(saving_path_num_2,f"{num}.png")
            plt.imsave(saving_path_num_3, img_array, cmap="gray")

            # perform fft of image_array
            img_fft = np.fft.fftshift(np.fft.fft2(img_array))

            # convert to log
            img_log = 1 + np.abs(img_fft)

            # normalize image_log
            img_save = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))

            # save img_save
            saving_path_fft_3 = os.path.join(saving_path_fft_2,f"{num}.png")
            plt.imsave(saving_path_fft_3, img_save, cmap="gray")

            if filename == "train" and num == 300:
                break
            if filename == "test" and num == 100:
                break


end_time = timer()
print_train_time(start=start_time,
                 end=end_time,
                 device="cpu")
