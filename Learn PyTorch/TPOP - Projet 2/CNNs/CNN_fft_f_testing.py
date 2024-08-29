import torch as t
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.io
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import glob
import os

from torch import nn
from torchinfo import summary
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timeit import default_timer as timer
from pathlib import Path
from PIL import Image

from CNN_fft_f import CNN_FFT
from functions import *


""" AGNOSTIC CODE """

device = "cuda" if t.cuda.is_available() else "cpu"


""" LOAD CUSTOM IMAGE """

data_transform = transforms.Compose([
    transforms.Resize(size=(32, 32), antialias=True),
    transforms.ConvertImageDtype(t.float32)
])

custom_image_path = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\tpop_donnees_fourier\nouvelles_donnees\lines\lignes_verticales.png"
custom_image_float32_resized = data_transform(torchvision.io.read_image(custom_image_path))

# plt.imshow(custom_image_float32_resized.permute(1, 2, 0), cmap="gray")
# plt.show()

custom_image = (custom_image_float32_resized.to(device)).unsqueeze(0)


""" LOAD CUSTOM TESTING DATA """

custom_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\tpop_donnees_fourier\nouvelles_donnees"

data_transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor()
])

custom_data = datasets.ImageFolder(root=custom_dir,
                                   transform=data_transform)
class_names = ['full grid', 'lines']


""" LOAD THE MODEL """

# create model save
MODEL_PATH = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\models - State_dicts")
MODEL_NAME = "CNN_fft.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# instantiate a new instance of our model class
loaded_CNN_fft_f = CNN_FFT(input_shape=3,
                           hidden_units=3,
                           output_shape=2).to(device)

# Load the saved state_dict() of LinearRegressionModel_1 (this will update the new instance with updated parameters)
loaded_CNN_fft_f.load_state_dict(t.load(f=MODEL_SAVE_PATH))

# put the loaded model on the GPU :
loaded_CNN_fft_f.to(device)

# print(loaded_CNN_fft_f.state_dict())


""" SUMMARIZE MODEL """

# summary(loaded_CNN_fft_f, input_size=[1, 3, 32, 32])


""" TEST 1 IMAGE """

loaded_CNN_fft_f.eval()
with t.inference_mode():
    custom_preds = loaded_CNN_fft_f(custom_image)

custom_probs = t.softmax(custom_preds, dim=1)
custom_labels = t.argmax(custom_probs, dim=1).cpu()
print(f"\nSingle image probabilities: {custom_probs}\nPrediction: {class_names[custom_labels]}")


""" MAKES PREDICTIONS """

test_samples = []
test_labels = []

for sample, label in random.sample(list(custom_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model=loaded_CNN_fft_f,
                              data=test_samples,
                              device=device)
pred_classes = pred_probs.argmax(dim=1)


""" VISUALIZE RESULTS """

# plot predictions
plt.figure(figsize=(16, 16))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):

    # create subplot
    plt.subplot(nrows, ncols, i + 1)

    # plot the target image
    plt.imshow((sample.squeeze()).permute(1, 2, 0), cmap="gray")

    # find the prediction label
    pred_label = class_names[pred_classes[i]]

    # get the truth label
    truth_label = class_names[test_labels[i]]

    # create title
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # check for equality between pred and trutch
    if pred_label == truth_label:
        plt.title(title_text, fontsize=18, c="g")
    else:
        plt.title(title_text, fontsize=18, c="r")
    plt.axis(False)
plt.show()
