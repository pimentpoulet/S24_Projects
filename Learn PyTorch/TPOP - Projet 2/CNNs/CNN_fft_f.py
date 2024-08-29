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

from torch import nn
from torchinfo import summary
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timeit import default_timer as timer
from pathlib import Path
from PIL import Image

from functions import *


""" GENERAL SOURCES """

"https://en.moonbooks.org/Articles/Create-number-images-from-0-to-9-with-the-python-module-Pillow/"
"https://github.com/teavanist/MNIST-JPG"
"https://poloclub.github.io/cnn-explainer/#article-pooling"
"https://www.youtube.com/watch?v=Z_ikDlimN6A&list=RDCMUCr8O8l5cCX85Oem1d18EezQ&start_radio=1"
"https://stackoverflow.com/questions/9506841/using-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image"
"https://www.kaggle.com/code/mrumuly/hopfield-network-mnist"
"https://horace.io/brrr_intro.html"
"https://randomgeekery.org/post/2017/11/drawing-grids-with-python-and-pillow/"


""" SETUP PATHS """

train_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\Grid Dataset\train"
test_dir = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\Grid Dataset\test"


""" CREATE TRAINING AND TESTING DATASETS """

data_transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)
class_names = train_data.classes


""" DATA LOADER """

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


""" AGNOSTIC CODE """

device = "cuda" if t.cuda.is_available() else "cpu"
print(device)

"""  CNN  """


class CNN_FFT(nn.Module):
    """
    Model architecture based TinyVGG's architecture
    """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )


    def forward(self, x: t.Tensor):
        return self.classifier(self.conv_block_1(x))  # self.conv_block_2(self.conv_block_1(x)))


CNN_fft = CNN_FFT(input_shape=3,                               # input_shapes is the number of color channels [1]
                  hidden_units=3,                              # number of neurons
                  output_shape=len(class_names)).to(device)    # number of classes in dataset


""" SUMMARIZE MODEL """

print()
summary(CNN_fft, input_size=[1, 3, 32, 32])
print()


""" SETUP LOSS FUNCTION AND OPTIMIZER """

loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(params=CNN_fft.parameters(),
                         lr=0.005)


""" PROTECT MAIN SCRIPT """

if __name__ == "__main__":
    train_time_start_on_gpu = timer()


    """ TRAINING AND TESTING LOOP """

    EPOCHS = 1
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}\n------")

        CNN_fft_results = train(model=CNN_fft,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                epochs=EPOCHS)


    """ AI DURATION """

    train_time_end_on_gpu = timer()
    total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=str(next(CNN_fft.parameters()).device))

    """ MAKE PREDICTIONS """

    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=16):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = make_predictions(model=CNN_fft,
                                  data=test_samples,
                                  device=device)
    pred_classes = pred_probs.argmax(dim=1)


    """ VISUALIZE RESULTS """

    # plot predictions
    plt.figure(figsize=(16, 16))
    nrows = 4
    ncols = 4
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
        title_text = f"Pred: {pred_label}\nTruth: {truth_label}"

        # check for equality between pred and trutch
        if pred_label == truth_label:
            plt.title(title_text, fontsize=12, c="g")
        else:
            plt.title(title_text, fontsize=12, c="r")

        plt.axis(False)

    plot_loss_curves(CNN_fft_results)


""" SAVE MODEL """

# create model directory path
MODEL_PATH = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\models - State_dicts")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# create model save
MODEL_NAME = "CNN_fft.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
t.save(obj=CNN_fft.state_dict(),
       f=MODEL_SAVE_PATH)
