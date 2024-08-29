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
from PIL import Image, ImageDraw

from functions import *


""" SETUP DATA DIRECTORY """

dir_path = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\MNIST Dataset")
# print("Looping through dir_path:")
# walk_through_dir(dir_path)
# print(f"\ndir_path: {dir_path}")

train_dir = os.path.join(dir_path, "train")
test_dir = os.path.join(dir_path, "test")

# print(f"train_dir: {train_dir}\ntest_dir: {test_dir}")


""" CREATE AND SAVE FFT DATASET """

parent_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\Grid Dataset"

# Define a transform to convert PIL image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=(224, 224))
])

class_names = ['full grid','lines']

start_time = timer()

# iterate over train/test folders
for filename in os.listdir(dir_path):

    # path to live train/test folder
    splits = os.path.join(dir_path, filename)

    # name of live train/test folder
    tt = os.path.basename(splits)

    # path to save_folder
    saving_path_1 = os.path.join(parent_dir, tt)

    # create folder
    os.makedirs(saving_path_1, exist_ok=True)

    # iterate over 0-9 folders in train/test folders
    for classe in class_names:

        # path to save_folder in train/test folders
        saving_path_2 = os.path.join(saving_path_1, classe)

        # create number folder
        os.makedirs(saving_path_2, exist_ok=True)

        # iterate over images in number folders
        for num in range(500):

            if classe == class_names[0]:


                """ DRAW GRID """

                step_count = random.randint(5,75)
                height = 224
                width = 224
                image = Image.new(mode='L', size=(height, width), color=255)

                # Draw some lines
                draw = ImageDraw.Draw(image)
                y_start = 0
                y_end = image.height
                step_size = int(image.width / step_count)

                for x in range(0, image.width, step_size):
                    line = ((x, y_start), (x, y_end))
                    draw.line(line, fill=128)

                x_start = 0
                x_end = image.width

                for y in range(0, image.height, step_size):
                    line = ((x_start, y), (x_end, y))
                    draw.line(line, fill=128)

                del draw


                """ COMPUTE FFT OF GRIDS """

                # turn PIL into tensor, permute shape for matplotlib compatibility and turn into numpy array
                img_tensor = transform(image).permute(1, 2, 0).squeeze()
                img_array = img_tensor.numpy()

                # perform fft of image_array
                img_fft = np.fft.fftshift(np.fft.fft2(img_array))

                # convert to log
                img_log = np.log(1 + np.abs(img_fft))

                # normalize image_log
                img_save = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))

                # plot predictions
                plt.figure(figsize=(10, 5))
                nrows = 1
                ncols = 2
                plt.subplot(nrows,ncols,1)
                plt.imshow(img_array, cmap="gray")
                plt.subplot(nrows, ncols,2)
                plt.imshow(img_save, cmap="gray")
                # plt.show()

                # save origin grid
                saving_path_3 = os.path.join(saving_path_2, f"{num}.png")
                # plt.imsave(saving_path_3, img_save, cmap="gray")


            if classe == class_names[1]:


                """ DRAW Lines """

                height = 224
                width = 224
                image = Image.new(mode='L', size=(height, width), color=255)

                # Draw some lines
                draw = ImageDraw.Draw(image)
                y_start = 0
                y_end = image.height
                step_size = int(image.width / random.randint(5,75))

                for x in range(0, image.width, step_size):
                    line = ((x, y_start), (x, y_end))
                    draw.line(line, fill=128)

                del draw


                """ COMPUTE FFT OF LINES """

                # turn PIL into tensor, permute shape for matplotlib compatibility and turn into numpy array
                img_tensor = transform(image).permute(1, 2, 0).squeeze()
                img_array = img_tensor.numpy()

                # perform fft of image_array
                img_fft = np.fft.fftshift(np.fft.fft2(img_array))

                # convert to log
                img_log = np.log(1 + np.abs(img_fft))

                # normalize image_log
                img_save = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))

                # save origin grid
                saving_path_3 = os.path.join(saving_path_2, f"{num}.png")
                # plt.imsave(saving_path_3, img_save, cmap="gray")

end_time = timer()
print(end_time - start_time)
