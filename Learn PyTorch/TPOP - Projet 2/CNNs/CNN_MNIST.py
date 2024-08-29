import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
import random

from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path

from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from functions import *


""" IMPORT MNIST DATASET """

train_data = tv.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = tv.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

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


"""  CNN  """


class CNN_MNIST(nn.Module):
    """
    Model architecture based TinyVGG's architecture
    """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
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
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )


    def forward(self, x: t.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


t.manual_seed(37)
CNN_MNIST = CNN_MNIST(input_shape=1,                               # input_shapes is the number of color channels [1]
                      hidden_units=10,                             # number of neurons
                      output_shape=len(class_names)).to(device)    # number of classes in dataset


""" SETUP LOSS FUNCTION AND OPTIMIZER """

loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(params=CNN_MNIST.parameters(),
                        lr=0.05)


""" TRAINING AND TESTING LOOP """

t.manual_seed(42)
train_time_start_on_gpu = timer()

EPOCHS = 2
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}\n------")

    # call training function
    train_step(model=CNN_MNIST,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)

    # call testing function
    test_step(model=CNN_MNIST,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)


""" AI DURATION """

# calculate training time
train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=str(next(CNN_MNIST.parameters()).device))

model_1_results = eval_model(model=CNN_MNIST,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)


""" MAKE PREDICTIONS """

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=16):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model=CNN_MNIST,
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
    plt.subplot(nrows, ncols, i+1)

    # plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # find the prediction label
    pred_label = class_names[pred_classes[i]]

    # get the truth label
    truth_label = class_names[test_labels[i]]

    # create title
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # check for equality between pred and trutch
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
    else:
        plt.title(title_text, fontsize=10, c="r")

    plt.axis(False)

plt.show()


""" SAVE MODEL """

# create model directory path
MODEL_PATH = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch/models - State_dicts")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# create model save
MODEL_NAME = "CNN_MNIST.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
t.save(obj=CNN_MNIST.state_dict(),
       f=MODEL_SAVE_PATH)
