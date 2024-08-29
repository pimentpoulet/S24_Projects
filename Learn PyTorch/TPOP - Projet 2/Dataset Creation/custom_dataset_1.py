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


""" SETUP DATA DIRECTORY """

dir_path = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\MNIST_data")
# print("Looping through dir_path:")
# walk_through_dir(dir_path)
# print(f"\ndir_path: {dir_path}")

train_dir = os.path.join(dir_path,"train")
test_dir = os.path.join(dir_path,"test")

# print(f"train_dir: {train_dir}\ntest_dir: {test_dir}")


""" CREATE AND SAVE FFT DATASET """

parent_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\fft_dataset_origin"

# Define a transform to convert PIL image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=(224, 224))
])


start_time = timer()

# iterate over train/test folders
for filename in os.listdir(dir_path):
    
    # path to live train/test folder
    splits = os.path.join(dir_path, filename)

    # name of live train/test folder
    tt = os.path.basename(splits)

    # path to save_folder
    saving_path_1 = os.path.join(parent_dir,tt)

    # create folder
    os.makedirs(saving_path_1, exist_ok=True)

    # iterate over 0-9 folders in train/test folders
    for split in os.listdir(splits):

        # get path to the live number folder
        classes = os.path.join(splits, split)

        # number we're at
        chiffre = os.path.basename(split)

        # path to save_folder in train/test folders
        saving_path_2 = os.path.join(saving_path_1,chiffre)

        # create number folder
        os.makedirs(saving_path_2, exist_ok=True)

        # iterate over images in number folders
        for num, classe in enumerate(os.listdir(classes)):

            # path to live image in number folder
            f = os.path.join(classes, classe)

            # transform image to tensor
            image = Image.open(f)
            image_tensor = transform(image).squeeze()

            # transform tensor to numpy array
            image_array = image_tensor.numpy()

            # perform fft of image_array
            image_fft = np.fft.fftshift(np.fft.fft2(image_array))

            # convert to log
            image_log = np.log(1 + np.abs(image_fft))

            # normalize image_log
            image_to_save = (image_log - np.min(image_log))/(np.max(image_log) - np.min(image_log))

            # compute saving_path_3
            saving_path_3 = os.path.join(saving_path_2,f"{num}.jpg")

            # save image
            plt.imsave(saving_path_3, image_to_save, cmap="gray")

        print(f"Transfer for {chiffre} folder completed. {len(os.listdir(classes))} images transformed and downloaded.")
    print(f"\nTransfer for {tt} split completed. {len(os.listdir(splits))} folders transformed and downloaded.\n")


end_time = timer()
print(end_time - start_time)
